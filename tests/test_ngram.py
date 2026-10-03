import json, torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
import qwen_engine as QE
tc = json.load(open(f"{QE.SRC}/config.json"))["text_config"]
cfg = Qwen4ExpTextConfig(**tc)
with torch.device("meta"):
    ref = M.Qwen4ExpTextNGramEmbedding(cfg, tc["ple_embed_dim"], 1, 0)
class _Id(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.empty(1))
    def forward(self, ids): return ids.unsqueeze(-1)
ref.ngram_embedding = _Id()
ref = ref.to_empty(device="cpu")
ref.layer_multipliers = M._build_layer_multipliers(ref.unigram_vocab_size, ref.ngram_size, 0, ref.seed)
ref.ngram_heads_vocab_sizes = torch.tensor(ref.head_vocab_sizes); ref.ngram_heads_offsets = torch.tensor(ref.head_offsets)
mine = QE.NGramIds(tc)
assert torch.equal(mine.mult, ref.layer_multipliers) and mine.sizes.tolist() == ref.head_vocab_sizes
torch.manual_seed(0)
eos = mine.eos
for trial in range(20):
    T = 40
    ids = torch.randint(0, tc["vocab_size"], (T,))
    ids[torch.randint(0, T, (3,))] = eos
    r = ref(ids[None], None)[0]                     # [T, 16]
    full = mine([eos, eos], ids.tolist())
    assert torch.equal(r, full), trial
    # пошагово
    hist = []
    for t in range(T):
        prev = hist[-2:]; prev = [eos] * (2 - len(prev)) + prev
        st = mine(prev, [int(ids[t])])
        assert torch.equal(st[0], r[t]), (trial, t)
        hist.append(int(ids[t]))
print("NGRAM OK: 20 последовательностей с eos внутри, целиком и пошагово = эталон; строк всего", mine.total)
