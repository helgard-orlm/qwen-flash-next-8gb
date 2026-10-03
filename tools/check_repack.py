"""Сверка перепаковки: 200 случайных экспертов (+ углы) — каждый кусок блока бит-в-бит против safetensors."""
import json, os, struct, random, glob
import numpy as np
SRC = os.environ.get("QW_ORIG", "/backup/ai-models/qwen38-flash-next-nvfp4")
BLOCK = 2764800
PARTS = ["gate_proj.weight", "up_proj.weight", "down_proj.weight",
         "gate_proj.weight_scale", "up_proj.weight_scale", "down_proj.weight_scale"]
SIZES = [819200] * 3 + [102400] * 3
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
hdrs = {}
def hdr(f):
    if f not in hdrs:
        with open(f"{SRC}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdrs[f] = (8 + n, json.loads(fh.read(n)))
    return hdrs[f]
def get(k):
    f = wm[k]; base, h = hdr(f); a, b = h[k]["data_offsets"]
    with open(f"{SRC}/{f}", "rb") as fh:
        fh.seek(base + a); return fh.read(b - a)
scal = np.load(os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts_scal.npy"))
random.seed(1)
pick = [(0, 0), (47, 511), (0, 511), (47, 0)] + [(random.randrange(48), random.randrange(512)) for _ in range(200)]
bad = 0
with open(os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts.bin"), "rb") as fx:
    for L, e in pick:
        fx.seek((L * 512 + e) * BLOCK); blk = fx.read(BLOCK); o = 0
        for p, s in zip(PARTS, SIZES):
            if blk[o:o + s] != get(f"model.language_model.layers.{L}.mlp.experts.{e}.{p}"): bad += 1; print("BAD", L, e, p)
            o += s
        for i, p in enumerate(["gate_proj.weight_scale_2", "up_proj.weight_scale_2", "down_proj.weight_scale_2",
                               "gate_proj.input_scale", "up_proj.input_scale", "down_proj.input_scale"]):
            if struct.pack("<f", scal[L, e, i]) != get(f"model.language_model.layers.{L}.mlp.experts.{e}.{p}"): bad += 1; print("BADS", L, e, p)
print("CHECK", len(pick), "экспертов, расхождений", bad)
