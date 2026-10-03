import os
import numpy as np, torch, time
from qmoe import *
scal = None
fx = open(os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts.bin"), "rb")
torch.manual_seed(0)
L, ex = 0, [3, 77, 100, 200, 255, 300, 401, 450, 500, 511]
S = 12
slots = torch.zeros(S, BLOCK, dtype=torch.uint8)
import struct, json
# скаляры из safetensors напрямую (npy ещё не записан)
SRC = os.environ.get("QW_ORIG", "/backup/ai-models/qwen38-flash-next-nvfp4")
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
from safetensors import safe_open
s2 = torch.zeros(len(ex), 3)
for j, e in enumerate(ex):
    fx.seek((L * 512 + e) * BLOCK); slots[j + 2] = torch.frombuffer(bytearray(fx.read(BLOCK)), dtype=torch.uint8)
    for i, p in enumerate(["gate_proj", "up_proj", "down_proj"]):
        k = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}.weight_scale_2"
        with safe_open(f"{SRC}/{wm[k]}", "pt") as f: s2[j, i] = f.get_tensor(k).float()
x = (torch.randn(2560) * 0.5).to(torch.bfloat16)
coef = torch.rand(len(ex))
# эталон fp32 на CPU
ref = torch.zeros(2560)
for j in range(len(ex)):
    g, u, d = block_deq(slots[j + 2], s2[j])
    h = torch.nn.functional.silu(g @ x.float()) * (u @ x.float())
    ref += coef[j] * (d @ h)
dev = "cuda"
sl = slots.to(dev); sidx = torch.arange(2, 2 + len(ex), dtype=torch.int32, device=dev)
hb = torch.empty(len(ex), 640, device=dev); ob = torch.empty(len(ex), 2560, device=dev)
y = moe_decode(sl, sidx, s2.to(dev), coef.to(dev), x.to(dev), hb, ob)
torch.cuda.synchronize()
err = (y.cpu() - ref).abs().max().item(); rel = err / ref.abs().max().item()
print("max|y-ref|", err, "отн.", rel, "норма ref", ref.norm().item())
for _ in range(3): moe_decode(sl, sidx, s2.to(dev), coef.to(dev), x.to(dev), hb, ob)
torch.cuda.synchronize(); t = time.time()
for _ in range(200): moe_decode(sl, sidx, s2.to(dev), coef.to(dev), x.to(dev), hb, ob)
torch.cuda.synchronize(); dt = (time.time() - t) / 200
print(f"K=10: {dt*1e6:.0f} мкс/слой, {10*BLOCK/dt/1e9:.0f} ГБ/с; 48 слоёв = {48*dt*1e3:.1f} мс/ток")
