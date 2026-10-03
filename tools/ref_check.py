"""ЭТАЛОН: модули transformers 5.18 (qwen4_exp) по одному слою, вся последовательность сразу (путь без кэша),
веса = те же файлы. Сравнение с движком: движок жадно генерирует N токенов (кэш, шаг по одному), эталон
считает логиты на prompt+gen[:-1] учительским прогоном и проверяет argmax на каждой позиции генерации.
Подмены в эталоне (только источник весов, алгоритм модулей нетронут):
  - эксперты: тот же цикл Qwen4ExpTextExperts.forward, веса gate_up/down = распаковка блока NVFP4 -> bf16;
  - таблица n-грамм: строки FP8 из файла (вместо nn.Embedding на 320 млн строк), буферы хеша считает код эталона.
Запуск: PYTHONPATH=/fast/qwen38/pylib python ref_check.py --new 32 [--no_ple] [--fp8 linear_attn]"""
import argparse, json, sys, time
import torch, torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers import AutoTokenizer
import qwen_engine as QE
import qmoe

ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=32)
ap.add_argument("--no_ple", action="store_true")
ap.add_argument("--fp8", default="linear_attn")
ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
ap.add_argument("--raw", action="store_true", help="без шаблона чата")
ap.add_argument("--chunk", type=int, default=512, help="промт движку кусками (как в службе)")
ap.add_argument("--long", type=int, default=0, help="длинный промт ~N токенов (проверка отбора индексатора > 2048)")
ap.add_argument("--out", default="ref_check.json")
ap.add_argument("--ref_sim_fp8", action="store_true", help="эталону те же FP8-округлённые веса, что у движка (опыт: отделить FP8 от ошибок кода)")
a = ap.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained(QE.SRC)
if a.long:
    import random
    random.seed(7)
    things = ["lighthouse", "river", "violin", "glacier", "market", "comet", "library", "orchard", "bridge", "desert"]
    acts = ["was repaired by", "was painted by", "was described by", "was visited by", "was measured by"]
    who = ["a sailor", "an engineer", "a child", "the mayor", "a botanist", "two students", "an old pilot"]
    facts, i = [], 0
    while len(tok("\n".join(facts))["input_ids"]) < a.long:
        i += 1
        facts.append(f"Fact {i}: the {random.choice(things)} number {random.randint(10, 999)} "
                     f"{random.choice(acts)} {random.choice(who)} in {random.randint(1500, 2020)}.")
    key = facts[37]
    a.prompt = ("Read the facts and answer.\n" + "\n".join(facts) +
                f"\nQuestion: according to Fact 38, who and in what year? Answer briefly.")
    print("ожидаемый ответ (Fact 38):", key, flush=True)
if a.raw:
    ids = tok(a.prompt)["input_ids"]
else:
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
assert isinstance(ids, list) and all(isinstance(t, int) for t in ids), type(ids)
print("промт токенов:", len(ids), "| подано:", repr(tok.decode(ids))[:300], "...", repr(tok.decode(ids[-40:])), flush=True)

# ---------------- движок
t0 = time.time()
eng = QE.QwenEngine(dev, fp8=tuple(x for x in a.fp8.split(",") if x), ngram=not a.no_ple)
print(f"движок загружен {time.time()-t0:.0f} с, VRAM {eng.vram_report():.2f} ГиБ", flush=True)
t0 = time.time()
for c0 in range(0, len(ids), a.chunk):
    lg = eng.forward(ids[c0:c0 + a.chunk])
tp = time.time() - t0
gen, eng_top, eng_lg = [], [], []
tg = time.time()
for i in range(a.new):
    t = int(lg.argmax()); gen.append(t)
    eng_top.append(lg.topk(5).indices.tolist()); eng_lg.append(lg.cpu())
    if t in (248046, 248044):
        break
    lg = eng.forward([t])
tg = time.time() - tg
print(f"движок: промт {tp:.1f} с, {len(gen)} ток за {tg:.1f} с = {len(gen)/tg:.2f} ток/с; чтений {eng.store.nread}, "
      f"диск {eng.store.read_s:.1f} с, PLE строки {eng.ple_read_s*1e3:.0f} мс всего", flush=True)
print("ДВИЖОК:", repr(tok.decode(gen)), flush=True)
# тот же движок учительским прогоном всей последовательности сразу (путь промта) — сверка шагового пути с кэшем
eng.reset()
seq_e = ids + gen[:-1]
parts = []
for c0 in range(0, len(seq_e), a.chunk):
    piece = seq_e[c0:c0 + a.chunk]
    lo = max(0, len(ids) - 1 - c0)                 # нужны логиты с позиции len(ids)-1
    if lo < len(piece):
        parts.append(eng.forward(piece, all_logits=True, logits_from=lo).cpu())
    else:
        eng.forward(piece)
eng_full = torch.cat(parts, 0)
eng_step = torch.stack(eng_lg)
d_sf = (eng_full - eng_step).abs()
print(f"движок шагами vs целиком: max|Δлогит| {d_sf.max():.3f}, средн {d_sf.mean():.4f}, argmax совпал "
      f"{int((eng_full.argmax(-1) == eng_step.argmax(-1)).sum())}/{len(gen)}", flush=True)
ple_rows = eng.ngt
import gc
del eng, lg; gc.collect(); torch.cuda.empty_cache()
print(f"после выгрузки движка занято {torch.cuda.memory_allocated()/2**30:.2f} ГиБ", flush=True)

# ---------------- эталон
cfgj = json.load(open(f"{QE.SRC}/config.json"))["text_config"]
cfg = Qwen4ExpTextConfig(**cfgj)
cfg._attn_implementation = "sdpa"
seq = ids + gen[:-1]
T = len(seq)
W = safe_open(QE.nonexpert_path(), "pt")
P = QE.PFX
fx = open(f"{QE.FAST}/experts.bin", "rb")
import numpy as np
scal = np.load(f"{QE.FAST}/experts_scal.npy")
Lcur = [0]


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """= Qwen4ExpTextExperts.forward эталона, веса эксперта берутся из блока NVFP4"""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        e = int(expert_idx)
        fx.seek((Lcur[0] * 512 + e) * qmoe.BLOCK)
        blk = torch.frombuffer(bytearray(fx.read(qmoe.BLOCK)), dtype=torch.uint8)
        g, u, d = qmoe.block_deq(blk, scal[Lcur[0], e, :3])
        gate_up = torch.cat([g, u], 0).to(dev, torch.bfloat16)
        down = d.to(dev, torch.bfloat16)
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, gate_up).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, down)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


class TableEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1), requires_grad=False)      # эталон смотрит .weight.device

    def forward(self, idx):
        return ple_rows.rows(idx.cpu()).to(dev, torch.bfloat16)


def load_into(mod, prefix):
    sd = {}
    for n, _ in list(mod.named_parameters()) + list(mod.named_buffers()):
        k = prefix + n
        if k in W.keys():
            sd[n] = W.get_tensor(k)
    missing = [n for n, _ in mod.named_parameters() if n not in sd]
    mod.load_state_dict(sd, strict=False)
    return missing


t0 = time.time()
with torch.no_grad():
    idt = torch.tensor([seq], device=dev)
    emb = W.get_tensor(P + "embed_tokens.weight")[torch.tensor(seq)].to(dev).unsqueeze(0)
    rot = M.Qwen4ExpTextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(emb, pos)
    mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]
    h = emb.repeat(1, 1, cfg.hc_count)
    for L in range(cfg.num_hidden_layers):
        Lcur[0] = L
        with torch.device("meta"):
            layer = M.Qwen4ExpTextDecoderLayer(cfg, L)
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = TableEmb()
        # полные матрицы 512 экспертов (3.1+1.6 ГиБ) не создаём: веса эксперта читаются в experts_forward
        layer.mlp.experts.gate_up_proj = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        layer.mlp.experts.down_proj = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        layer = layer.to_empty(device=dev).to(torch.bfloat16)
        miss = load_into(layer, P + f"layers.{L}.")
        miss = [m for m in miss if not m.startswith("mlp.experts.") and "ngram_embedding" not in m]
        assert not miss, (L, miss)
        layer.mlp.experts.forward = experts_forward.__get__(layer.mlp.experts)
        if a.ref_sim_fp8:
            fp8set = tuple(x for x in a.fp8.split(",") if x)
            nsim = 0
            for n, p_ in layer.named_parameters():
                full = P + f"layers.{L}." + n
                if p_.dim() == 2 and n.endswith(".weight") and any(t in full for t in fp8set) and \
                        any(s in n for s in ("in_proj_qkv", "in_proj_z", "out_proj")):
                    q8, s8 = QE.to_fp8_rows(p_.data)
                    p_.data.copy_((q8.to(torch.bfloat16) * s8).to(torch.bfloat16)); nsim += 1
            if L == 0: print("эталон: FP8-округлено матриц в слое 0:", nsim, flush=True)
        if layer.ple is not None:
            if a.no_ple:
                layer.ple = None
            else:
                pe = layer.ple.ple_embedding          # буферы загружены из файла; код эталона обязан дать то же
                assert torch.equal(pe.layer_multipliers.cpu(), M._build_layer_multipliers(
                    pe.unigram_vocab_size, pe.ngram_size, pe.ple_layer_index, pe.seed))
                assert pe.ngram_heads_vocab_sizes.tolist() == pe.head_vocab_sizes
                assert pe.ngram_heads_offsets.tolist() == pe.head_offsets
        layer.eval()
        h = layer(h, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
                  past_key_values=None, ple_input_ids=idt)
        del layer
    with torch.device("meta"):
        mix = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    mix = mix.to_empty(device=dev).to(torch.bfloat16)
    assert not load_into(mix, P + "hyper_connection_mixer.")
    hm = mix(h)
    lm = W.get_tensor("lm_head.weight").to(dev)
    logits = nn.functional.linear(hm[0, len(ids) - 1:], lm).float()
print(f"эталон {time.time()-t0:.0f} с", flush=True)
ref_arg = logits.argmax(-1).tolist()
lc = logits.cpu()
d = (lc - eng_step).abs()
print(f"движок(шаги) vs эталон: max|Δлогит| {d.max():.3f}, средн {d.mean():.4f}; логиты эталона: размах {lc.std():.2f}", flush=True)
t2 = lc.topk(2, -1)
for i, (rr, g) in enumerate(zip(ref_arg, gen)):
    if rr != g:
        print(f"  поз {i}: эталон {tok.decode([rr])!r} {lc[i, rr]:.3f} / движок {tok.decode([g])!r} у эталона {lc[i, g]:.3f} "
              f"(зазор {lc[i, rr] - lc[i, g]:.3f}); у движка {eng_step[i, g]:.3f} vs {eng_step[i, rr]:.3f}", flush=True)
mg = (t2.values[:, 0] - t2.values[:, 1])
print(f"зазор top1-top2 эталона: медиана {mg.median():.3f}, мин {mg.min():.3f}", flush=True)
ok = sum(int(r == g) for r, g in zip(ref_arg, gen))
first_bad = next((i for i, (r, g) in enumerate(zip(ref_arg, gen)) if r != g), None)
in5 = sum(int(r in t5) for r, t5 in zip(ref_arg, eng_top))
print(f"СОВПАЛО argmax {ok}/{len(gen)}; эталонный top1 в top5 движка {in5}/{len(gen)}; первое расхождение {first_bad}")
print("ЭТАЛОН жадно по тем же префиксам:", repr(tok.decode(ref_arg)))
json.dump(dict(ids=ids, gen=gen, ref=ref_arg, ok=ok, n=len(gen), first_bad=first_bad, fp8=a.fp8, no_ple=a.no_ple),
          open(a.out, "w"))
