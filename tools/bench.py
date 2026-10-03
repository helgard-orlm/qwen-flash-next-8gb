"""Замер скорости движка: промт + N шагов; время шага разложено на ожидание диска (store.read_s) и остальное.
Запуск: PYTHONPATH=/fast/qwen38/pylib python bench.py --new 64 [--no_ple] [--ram_gb 16] [--pf 2]"""
import argparse, json, time, torch
from transformers import AutoTokenizer
import qwen_engine as QE

ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=64)
ap.add_argument("--no_ple", action="store_true")
ap.add_argument("--fp8", default="linear_attn")
ap.add_argument("--prompt", default="Напиши короткий рассказ о маяке на краю света.")
ap.add_argument("--out", default="bench.json")
ap.add_argument("--opts", default="{}", help="JSON с параметрами движка (ram_gb, pf, ...)")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(QE.SRC)
text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                               enable_thinking=False, tokenize=False)
ids = tok(text, add_special_tokens=False)["input_ids"]
eng = QE.QwenEngine("cuda", fp8=tuple(x for x in a.fp8.split(",") if x), ngram=not a.no_ple, **json.loads(a.opts))
print(f"VRAM {eng.vram_report():.2f} ГиБ", flush=True)
res = []
for rep in range(2):                     # второй проход — с прогретым RAM-кэшем (если он есть)
    eng.reset()
    t = time.time(); lg = eng.forward(ids); torch.cuda.synchronize(); tp = time.time() - t
    r0, n0, p0 = eng.store.read_s, eng.store.nread, eng.ple_read_s
    st0 = dict(getattr(eng.store, "stat", {}))
    gen = []; t = time.time()
    for i in range(a.new):
        x = int(lg.argmax()); gen.append(x)
        if x in (248046, 248044):
            break
        lg = eng.forward([x])
    torch.cuda.synchronize(); tg = time.time() - t
    n = len(gen)
    rd = eng.store.read_s - r0
    st = {k: v - st0.get(k, 0) for k, v in getattr(eng.store, "stat", {}).items()}
    r = dict(rep=rep, prompt_tok=len(ids), prompt_s=round(tp, 2), n=n, tok_s=round(n / tg, 3),
             s_per_tok=round(tg / n, 4), disk_wait_per_tok=round(rd / n, 4),
             reads_per_tok=round((eng.store.nread - n0) / n, 1), ple_ms_per_tok=round((eng.ple_read_s - p0) / n * 1e3, 2),
             **{k: (round(v / n, 2) if isinstance(v, (int, float)) else v) for k, v in st.items()})
    res.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    print("ТЕКСТ:", repr(tok.decode(gen)), flush=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False)
