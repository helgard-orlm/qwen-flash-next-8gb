"""Перепаковка экспертов Qwen3.8-Flash-Next NVFP4 в один кусок на эксперта (для чтения с /fast одним pread).
Блок (L*512+e), 2 764 800 Б = 675 страниц по 4 КиБ:
  gate_w [640,1280]u8 | up_w [640,1280]u8 | down_w [2560,320]u8 | gate_s [640,160]e4m3 | up_s | down_s [2560,40]e4m3
Скаляры -> experts_scal.npy float32 [48,512,6]: ws2 gate/up/down, input_scale gate/up/down.
Источник читается по возрастанию смещений в каждом файле (HDD /backup)."""
import json, os, re, struct, sys, time, glob
import numpy as np

SRC = os.environ.get("QW_ORIG", "/backup/ai-models/qwen38-flash-next-nvfp4")
DST = os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts.bin")
NL, NE = 48, 512
PARTS = [("gate_proj.weight", 819200), ("up_proj.weight", 819200), ("down_proj.weight", 819200),
         ("gate_proj.weight_scale", 102400), ("up_proj.weight_scale", 102400), ("down_proj.weight_scale", 102400)]
OFF = {}
o = 0
for n, s in PARTS:
    OFF[n] = (o, s); o += s
BLOCK = o
assert BLOCK == 2764800 and BLOCK % 4096 == 0
SCAL = {"gate_proj.weight_scale_2": 0, "up_proj.weight_scale_2": 1, "down_proj.weight_scale_2": 2,
        "gate_proj.input_scale": 3, "up_proj.input_scale": 4, "down_proj.input_scale": 5}
pat = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")

scal = np.full((NL, NE, 6), np.nan, dtype=np.float32)
fd = os.open(DST, os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, NL * NE * BLOCK)
seen = 0
t0 = time.time(); nbytes = 0
for f in sorted(glob.glob(f"{SRC}/model-0*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    items = []
    for k, v in hdr.items():
        m = pat.match(k)
        if not m:
            continue
        L, e, part = int(m.group(1)), int(m.group(2)), m.group(3)
        a, b = v["data_offsets"]
        items.append((a, b, L, e, part, v["dtype"]))
    items.sort()
    src = os.open(f, os.O_RDONLY)
    for a, b, L, e, part, dt in items:
        buf = os.pread(src, b - a, base + a)
        assert len(buf) == b - a
        if part in SCAL:
            assert dt == "F32" and len(buf) == 4
            scal[L, e, SCAL[part]] = struct.unpack("<f", buf)[0]
        else:
            po, ps = OFF[part]
            assert ps == len(buf), (part, len(buf))
            os.pwrite(fd, buf, (L * NE + e) * BLOCK + po)
            seen += 1; nbytes += len(buf)
    os.close(src)
    el = time.time() - t0
    print(f"{os.path.basename(f)}: частей {seen}/{NL*NE*6}, {nbytes/2**30:.1f} ГиБ, {nbytes/2**20/max(el,1e-9):.0f} МиБ/с", flush=True)
os.fsync(fd); os.close(fd)
assert seen == NL * NE * 6, seen
assert not np.isnan(scal).any(), int(np.isnan(scal).sum())
np.save(os.path.join(os.environ.get("QW_DIR", "/fast/qwen38"), "experts_scal.npy"), scal)
print("REPACK_DONE", seen, f"{time.time()-t0:.0f} с", flush=True)
