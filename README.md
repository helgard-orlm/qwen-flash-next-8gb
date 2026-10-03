# Qwen3.8-Flash-Next (133 GB NVFP4) on an 8 GB GPU — experts streamed from NVMe

A from-scratch inference engine that runs `nvidia/Qwen3.8-Flash-Next-NVFP4` (48-layer MoE, 512 experts, top-10)
on a single **RTX 5060 8 GB** at **~9–12 tokens/s** in chat. Only the dense part of the model lives on the GPU;
the 63 GiB of experts are read on demand from an NVMe drive, through a RAM cache, one layer at a time.

Output is checked against the Hugging Face reference implementation (`transformers` 5.18, `qwen4_exp`)
fed with the same weights — see [Correctness](#correctness).

Batch size 1, text only (vision and MTP heads are not wired in). Code comments are in Russian.

## Hardware it was built on

| part | value |
|---|---|
| GPU | RTX 5060, 8 GB (Blackwell, sm_120), PCIe 5.0 ×8 — measured host→GPU 28.6 GB/s |
| CPU | Intel Core Ultra 5 225F (6 P-cores + 4 E-cores) |
| RAM | 31 GiB |
| model disk | PCIe Gen5 NVMe, ×4 lanes from the CPU, used only for model files |

## The method

**Why this model fits the idea.** Each expert is small: 3 × 2560×640 in NVFP4 = 2.70 MiB with scales, 10 per layer,
48 layers ⇒ **1.27 GiB of experts per token**. For comparison, DeepSeek-V4.1-Flash needs ~4.2 GiB/token
on the same scheme. All experts together are 63.3 GiB; a 16 GB RAM cache holds about a quarter of them.

1. **Split the model.** Everything that is not an expert (embeddings, DeltaNet, attention, router, shared expert,
   lm_head; ~7.2 GB BF16) is extracted once into `nonexpert.safetensors`. The large DeltaNet matrices are stored
   on the GPU in FP8 E4M3 with one scale per row, so the resident part fits: **6.13 GiB VRAM** for the whole engine.
2. **Repack experts for the disk** (`repack_experts.py`). One contiguous 2,764,800-byte block per expert
   (675 pages of 4 KiB): `gate_w | up_w | down_w | gate_s | up_s | down_s`. One `pread` with `O_DIRECT`
   per expert into page-aligned pinned memory, then one copy to the GPU.
3. **Expert cache** (`ExpertCache` in `qwen_engine.py`): RAM LRU in pinned memory + a pool of reader threads +
   two banks of GPU slots, so copies for layer L+1 can run while layer L computes.
4. **Prefetch into the gap** (`QW_PF=2`): after the reads of a layer are issued, guess the next layer's top
   experts and read them while the GPU is busy (91% guessed). It must *not* run at the same time as the
   layer's required reads — sharing the disk made it slower.
5. **Speculative copies of the next layer** (`QW_SPEC=10`): run the router of layer L+1 on the MoE input of
   layer L (top-10 hit rate 65%) and start the host→GPU copies early.
6. **Own kernels (Triton)**:
   - `qmoe.py` — NVFP4 SwiGLU expert on one token, using the hardware `cvt.rn.f16x2.e2m1x2` instruction
     (sm_120a), W4A16, fp32 accumulation; 3.6e-7 from the torch reference, 115 µs/layer;
   - `fp8gemv.py` — FP8-row-scaled matrix × vector; the previous unpack path cost ~106 ms per token.
7. **CUDA graph for one-token DeltaNet** (`qwen_delta_graph.py`, written by Codex/GPT): 36 DeltaNet layers
   captured into graphs that share **one** memory pool (36 private pools ran out of memory).
   Graphs are dropped before any prompt ≥1024 tokens to give the prompt its activation memory.
8. **Pin the decode thread** to P-core 0 (`QW_PIN=0`). On a hybrid CPU the unpinned thread jumps between
   P and E cores: 93–108 ms/token vs a stable 81.6 ms when pinned.
9. **Prompt in large pieces.** Reading the prompt in 512-token chunks re-read every expert for every chunk
   (66 GB per chunk). Now the whole prompt (or a piece of it) goes through each layer at once, experts are
   read in batches of 32: a 2451-token prompt went from 71.6 s to 23.3 s. Piece size is chosen so that
   `piece × (position + piece) ≤ 3072²`, which keeps prompts up to the 8192 limit inside 8 GB.
10. **PLE n-gram table** (51 B parameters, FP8, 53.7 GB file): only the rows for the current token's n-grams
    are read from NVMe — 0.3–0.5 ms/token.
11. **Server** (`qwen_server.py`): OpenAI-compatible `/v1/chat/completions` with streaming, cancel, and
    dialogue snapshots in RAM (recurrent DeltaNet state + KV + indexer keys), so a follow-up message only
    processes the new tail. Two model ids: `qwen3.8-flash-next` (greedy) and `qwen3.8-flash-next-think`
    (thinking on, `reasoning_effort` default `medium`, sampling from the authors' generation config).

### Where the time goes (all experts in RAM, before the CUDA graph)

80.7–88 ms per token ≈ **39 ms of Python issuing GPU work** + **47 ms moving 1.33 GB of experts over PCIe**,
and the two ran one after the other. The 52 CPU↔GPU syncs per token are cheap by themselves — they only
mattered because they delayed the start of the expert copies. Hence items 5, 7 and 8.

## Results

Speeds are tokens/s of the answer, single user.

| configuration | tok/s |
|---|---|
| first version, experts from disk only, no cache | 2.68 |
| disk only, prefetch P=2 | 4.72–4.75 |
| RAM cache 16 GB + P=2 | 8.1 |
| live chat, before speed-up (1000-token answer) | 9.71 |
| **live chat, with items 5 + 7 + 8 (1000-token answer)** | **11.65** |

Live API before → after the speed-up, one run each: Russian text 9.18 → 11.31, code 9.41 → 10.93,
thinking mode 10.00 → 10.89, long context 7.64 → 8.23.
Prompt: 24 tokens 4.5 s; 2451 tokens 23.3 s; 7182 tokens 98 s.

## Correctness

Reference = `transformers` 5.18 `qwen4_exp` with the **same** FP8/NVFP4 weights (`tools/ref_check.py`).

- next-token argmax with PLE: **32/32**, mean |Δlogit| 0.077;
- long prompt of 2451 tokens (beyond 2048, where the attention indexer starts selecting blocks): **11/11**,
  and the needle-in-a-haystack fact is answered correctly;
- perplexity (`tools/ppl_ablate.py`): EN 2.21, RU 2.22. Control runs that must break: nibbles swapped
  2016 / 33,269; experts removed 588 / 5,351 — so the unpacking is verified independently of the reference;
- the speed-up (graphs + speculation + pinning) was accepted only with `equal_reference: true` on all runs,
  plus a 10-answer deterministic API acceptance test (2/2 PASS).

## Tried and rejected

- **Fused DeltaNet recurrence** in one kernel (4.3 → 0.43 ms for 36 layers): 5/256 argmax differ → rejected.
- **Copying top-16 predicted experts** instead of top-10: more bytes over the bus than it saves.
- **Fewer CPU↔GPU syncs** as a goal: measured, not the bottleneck by themselves.

## Running

```bash
pip install -r requirements.txt
export QW_ORIG=/path/to/hf-snapshot QW_DIR=/path/on/nvme/qwen38
./setup.sh      # downloads the snapshot if QW_ORIG is empty, links/copies configs + PLE table,
                # repacks and checks experts, extracts the non-expert weights
QW_RAM_GB=16 QW_PF=2 QW_SPEC=10 QW_PIN=0 QW_DELTA_GRAPH=1 python qwen_server.py   # :9805
```

`setup.sh` checks free space before it starts and handles both layouts:

| layout | example | peak space | after deleting `QW_ORIG` |
|---|---|---|---|
| **two disks** | snapshot on an HDD, `QW_DIR` on NVMe | HDD 133 GB + NVMe 132 GB | NVMe 132 GB |
| **one disk** | both on the same NVMe | 211 GB (PLE table and configs are hard-linked, not copied) | 132 GB |

After setup the snapshot is not needed any more; the script prints how much deleting it frees.
The 9.9 GB of non-expert weights are extracted by `setup.sh` (older versions did it on the first server start,
which still needed the snapshot).

RESULTS_PLACEHOLDER

Then point any OpenAI-compatible client (we use Open WebUI) at `http://<host>:9805/v1`.
The server unloads Ollama models from the GPU on start (`127.0.0.1:11434`), because they share the card.
Tools and tests import the engine from the repo root: `PYTHONPATH=. python tools/bench.py --new 64`.

| env | meaning | default |
|---|---|---|
| `QW_DIR` | working directory on NVMe | `/fast/qwen38` |
| `QW_ORIG` | original HF snapshot (setup only) | — |
| `QW_RAM_GB` | expert RAM cache | 16 |
| `QW_PF` | experts prefetched into the disk gap | 2 |
| `QW_SPEC` | next-layer experts copied to GPU early | 0 (use 10) |
| `QW_PIN` | CPU core of the decode thread | unpinned (use 0) |
| `QW_DELTA_GRAPH` | CUDA graphs for one-token DeltaNet | 0 (use 1) |
| `QW_MAX_SEQ` | context limit | 8192 |

## Notes

- `qwen_delta_graph.py` is Codex's module plus one fix made in production: `capture_error_mode="thread_local"`.
  In the default `global` mode the expert-reader threads call `event.synchronize()` during capture, which
  raises `cudaErrorStreamCaptureInvalidated` after long prompts. All engine/server/kernel files here are
  byte-identical to the running service, except that hard-coded paths became `QW_DIR` / `QW_ORIG`.
- Runtime we use: torch 2.11.0+cu128, triton 3.6.0, CUDA toolkit 13.2 (`CUDA_HOME`) for Triton,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- The NVFP4 kernel needs the `e2m1` conversion instruction of Blackwell (sm_120a).
- Model weights are not included; they are under NVIDIA's license on the model card.

## Credits

Direction and decisions: helgard. Engine, kernels, server: Claude (Anthropic). DeltaNet CUDA graph and the
speculation/pinning A/B: Codex (OpenAI). Reference implementation: Hugging Face `transformers`.

License: Apache-2.0 (code in this repository).
