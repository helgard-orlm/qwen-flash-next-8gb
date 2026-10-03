"""Независимая проверка распаковки экспертов: перплексия учительским прогоном на известном тексте.
Режимы: верно / SWAP (полубайты наоборот) / без экспертов. Верная распаковка обязана быть заметно лучше обоих."""
import argparse, json, math, time, torch
from transformers import AutoTokenizer
import qwen_engine as QE, qmoe
ap = argparse.ArgumentParser(); ap.add_argument("--no_ple", action="store_true"); ap.add_argument("--fp8", default="linear_attn")
a = ap.parse_args()
TEXTS = {
 "en": "The water cycle describes how water moves through the environment. Water evaporates from oceans, lakes and rivers when it is heated by the sun. The vapour rises into the atmosphere, cools, and condenses into tiny droplets that form clouds. When the droplets combine and become heavy enough, they fall back to the ground as rain or snow. Some of this water flows over the surface into streams and rivers, while some soaks into the soil and becomes groundwater.",
 "ru": "Москва — столица России, крупнейший по численности населения город страны. Город расположен на реке Москве в центре Восточно-Европейской равнины. Впервые Москва упоминается в летописи 1147 года, когда князь Юрий Долгорукий пригласил туда своего союзника. В XIV веке город стал центром Московского княжества, а позднее — столицей единого Русского государства.",
}
tok = AutoTokenizer.from_pretrained(QE.SRC)
eng = QE.QwenEngine("cuda", fp8=tuple(x for x in a.fp8.split(",") if x), ngram=not a.no_ple)
res = {}
for mode in ("верно", "SWAP", "без_экспертов"):
    qmoe.SWAP = mode == "SWAP"; eng.no_experts = mode == "без_экспертов"
    for name, txt in TEXTS.items():
        ids = tok(txt)["input_ids"]
        eng.reset(); t = time.time()
        lg = eng.forward(ids, all_logits=True)
        lp = torch.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:], device=lg.device)[:, None])
        ppl = math.exp(-lp.mean().item())
        res[f"{mode}/{name}"] = ppl
        print(f"{mode:14s} {name}: перплексия {ppl:10.2f}  ({len(ids)} ток, {time.time()-t:.0f} с)", flush=True)
qmoe.SWAP = False
json.dump(res, open("ppl_ablate.json", "w"), ensure_ascii=False)
