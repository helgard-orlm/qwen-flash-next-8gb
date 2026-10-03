import os
import json, numpy as np, torch
from safetensors import safe_open
import qmoe
SRC = os.environ.get("QW_ORIG", "/backup/ai-models/qwen38-flash-next-nvfp4")
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
scal = None
fx = open(os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts.bin"), "rb")
import struct
for L in (0, 20):
    sh = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        k = f"model.language_model.layers.{L}.mlp.shared_expert.{p}.weight"
        with safe_open(f"{SRC}/{wm[k]}", "pt") as f: sh[p] = f.get_tensor(k).float()
    for e in (0, 123):
        s2 = []
        for p in ("gate_proj", "up_proj", "down_proj"):
            k = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}.weight_scale_2"
            with safe_open(f"{SRC}/{wm[k]}", "pt") as f: s2.append(float(f.get_tensor(k)))
        fx.seek((L * 512 + e) * qmoe.BLOCK)
        blk = torch.frombuffer(bytearray(fx.read(qmoe.BLOCK)), dtype=torch.uint8)
        g, u, d = qmoe.block_deq(blk, s2)
        print(f"L{L} e{e}: std gate/up/down эксперта {g.std():.4f}/{u.std():.4f}/{d.std():.4f} | общий {sh['gate_proj'].std():.4f}/{sh['up_proj'].std():.4f}/{sh['down_proj'].std():.4f} | ws2 {s2}")
