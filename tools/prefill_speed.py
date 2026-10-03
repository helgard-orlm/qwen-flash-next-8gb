"""Скорость промта: длинный (~2450 ток) и короткий; кусками vs целиком; ответ обязан совпасть."""
import random, time, torch, json, sys
from transformers import AutoTokenizer
import qwen_engine as QE
tok = AutoTokenizer.from_pretrained(QE.SRC)
random.seed(7)
things = ["lighthouse", "river", "violin", "glacier", "market", "comet", "library", "orchard", "bridge", "desert"]
acts = ["was repaired by", "was painted by", "was described by", "was visited by", "was measured by"]
who = ["a sailor", "an engineer", "a child", "the mayor", "a botanist", "two students", "an old pilot"]
facts, i = [], 0
while len(tok("\n".join(facts))["input_ids"]) < 2400:
    i += 1
    facts.append(f"Fact {i}: the {random.choice(things)} number {random.randint(10, 999)} "
                 f"{random.choice(acts)} {random.choice(who)} in {random.randint(1500, 2020)}.")
long_p = ("Read the facts and answer.\n" + "\n".join(facts) + "\nQuestion: according to Fact 38, who and in what year? Answer briefly.")
eng = QE.QwenEngine("cuda", ram_gb=float(sys.argv[1]) if len(sys.argv) > 1 else 0.0, pf=2)
for name, p in (("короткий", "Напиши короткий рассказ о маяке на краю света."), ("длинный", long_p)):
    text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    for ch in (512, 4096):
        eng.reset(); n0 = eng.store.nread
        torch.cuda.synchronize(); t = time.time()
        for c0 in range(0, len(ids), ch):
            lg = eng.forward(ids[c0:c0 + ch])
        torch.cuda.synchronize(); tp = time.time() - t
        g = []
        for _ in range(12):
            x = int(lg.argmax()); g.append(x)
            if x in (248046, 248044): break
            lg = eng.forward([x])
        print(f"{name} {len(ids)} ток, куски {ch}: промт {tp:.1f} с ({len(ids)/tp:.0f} ток/с), чтений {eng.store.nread - n0}, "
              f"пик VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} ГиБ | {tok.decode(g)!r}", flush=True)
        torch.cuda.reset_peak_memory_stats()
