"""Qwen3.8-Flash-Next (qwen4_exp, NVFP4 от NVIDIA) на gpu104: эксперты с /fast, остальное на карте. v1 — правильность.
Счёт перенесён из эталона transformers 5.18 `models/qwen4_exp/modeling_qwen4_exp.py` (одна последовательность, batch=1).
Отличия от эталона (осознанные):
  - эксперты W4A16: распакованный NVFP4 × bf16-активация, накопление fp32 (qmoe.py); эталон-сверка делает то же через torch;
  - --fp8 <группы>: большие матрицы хранятся FP8 E4M3 с масштабом на строку (иначе постоянная часть не влезает в 8 ГБ);
  - внимание: отбор индексатора (top-512 блоков по 4 токена) как в эталоне; до 2048+3 токенов он = полное причинное.
Блоки экспертов: /fast/qwen38/experts.bin (repack_experts.py), скаляры experts_scal.npy."""
import collections, json, math, mmap, os, struct, sys, threading, time
import concurrent.futures as cf
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
import qmoe
import fp8gemv

SRC = os.environ.get("QW_DIR", "/fast/qwen38")   # рабочий каталог на быстром NVMe (experts.bin, nonexpert, конфиги)
ORIG = os.environ.get("QW_ORIG", "/backup/ai-models/qwen38-flash-next-nvfp4")   # исходный снимок HF, только для одноразового извлечения
FAST = SRC
PFX = "model.language_model."
BLK = 4096
_KEEP = []


def pinned_aligned(nbytes):
    """закреплённая память, выровненная по странице (pin_memory() не выровнен — O_DIRECT падает, урок 14.09)"""
    nr = (nbytes + BLK - 1) // BLK * BLK
    mm = mmap.mmap(-1, nr)
    t = torch.from_numpy(np.frombuffer(mm, dtype=np.uint8, count=nbytes))
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nr, 0)
    assert int(rc) == 0, f"cudaHostRegister {rc}"
    _KEEP.append(mm)
    return t


# ============================================================ постоянная часть (не эксперты)
def nonexpert_path():
    return f"{FAST}/nonexpert.safetensors"


def extract_nonexpert():
    """один раз: все не-эксперты текстовой модели (bf16, без PLE-таблицы, без зрения/MTP) -> /fast одним файлом"""
    wm = json.load(open(f"{ORIG}/model.safetensors.index.json"))["weight_map"]
    keep = [k for k in wm if (k.startswith(PFX) or k == "lm_head.weight") and ".mlp.experts." not in k
            and "ngram_embedding" not in k]
    out = {}
    byfile = {}
    for k in keep:
        byfile.setdefault(wm[k], []).append(k)
    for f, ks in sorted(byfile.items()):
        with safe_open(f"{ORIG}/{f}", "pt") as fh:
            for k in ks:
                out[k] = fh.get_tensor(k).contiguous()
    save_file(out, nonexpert_path())
    return len(out)


@torch.no_grad()
def to_fp8_rows(w):
    """bf16 [N,K] -> (e4m3 [N,K], масштаб fp32 [N,1]), amax строки -> 448"""
    w = w.float()
    s = (w.abs().amax(1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (w / s).to(torch.float8_e4m3fn), s


class Lin:
    """линейный слой без смещения: bf16 на карте или FP8 со строковым масштабом (распаковка в bf16 на лету)"""

    def __init__(self, w, fp8, dev):
        self.fp8 = fp8
        if fp8:
            q, s = to_fp8_rows(w)
            self.q, self.s = q.to(dev), s.to(dev)
            self.s1 = self.s.view(-1).contiguous()
        else:
            self.w = w.to(dev, torch.bfloat16)

    def __call__(self, x):
        if self.fp8:
            if x.shape[0] == 1:                                  # шаг одного токена: ядро fp8 x вектор (читает только fp8)
                return fp8gemv.gemv(self.q, self.s1, x[0].contiguous())[None]
            return F.linear(x, (self.q.to(torch.bfloat16) * self.s).to(torch.bfloat16))
        return F.linear(x, self.w)

    def nbytes(self):
        return self.q.numel() + self.s.numel() * 4 if self.fp8 else self.w.numel() * 2


# ============================================================ мелкие части, дословно как в эталоне
def rmsnorm(x, w, eps, group=None):
    """Qwen4ExpTextRMSNorm: (x/rms) * (1 + w), считается во fp32, результат в dtype x"""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        o = o.flatten(-2)
    return (o * (1.0 + w.float())).type_as(x)


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x [..., T, H, D] или совместимая форма; cos/sin уже приведены к форме x[..., :rd]"""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None):
    """= torch_chunk_gated_delta_rule эталона (use_qk_l2norm_in_kernel=True, output_final_state=True)"""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query = l2norm(query); key = l2norm(key)
    query = query * query.shape[-1] ** -0.5
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    num_chunks = (sequence_length + pad_size) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    pw = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pw
    intra = (query @ key.transpose(-1, -2)) * pw
    dkb = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    S = (torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=new_values.dtype, device=new_values.device)
         if initial_state is None else initial_state.to(new_values))
    out = torch.zeros_like(new_values)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cum[:, :, i] @ S
        out[:, :, i] = query[:, :, i] @ S + intra[:, :, i] @ v_new
        S = S * cd[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    return out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format), S


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state):
    """= torch_recurrent_gated_delta_rule эталона для одного токена; S меняется на месте не обязательно — возвращается"""
    initial_dtype = query.dtype
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query = l2norm(query); key = l2norm(key)
    query = query / (query.shape[-1] ** 0.5)
    S = initial_state
    out = torch.zeros_like(value)
    for i in range(query.shape[2]):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        S = S * decay[:, :, i].exp()[..., None, None]
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (S * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(initial_dtype), S


# ============================================================ PLE: хеш n-грамм (дословно эталон)
_MASK64 = (1 << 64) - 1
_GAMMA, _M1, _M2, _P1 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(v):
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramIds:
    """номера строк таблицы n-грамм для токенов (CPU, int64 — переполнение как у torch)"""

    def __init__(self, tc, ple_layer_index=0):
        self.n = tc["ngram_size"]
        self.ctx = self.n - 1
        self.hpn = tc["heads_per_ngram"]
        self.heads = (self.n - 1) * self.hpn
        self.eos = tc["eos_token_id"][0] if isinstance(tc["eos_token_id"], list) else tc["eos_token_id"]
        sizes, offs, tot = [], [], 0
        for h in range(self.heads):
            s = _nth_prime_after(tc["ngram_vocab_size_base"] - 1, ple_layer_index * self.heads + h + 1)
            sizes.append(s); offs.append(tot); tot += s
        self.total = tot
        V = tc["vocab_size"]
        mmax = ((1 << 63) - 1) // max(V, 1)
        half = max(1, mmax // 2)
        base = tc["seed"] + _P1 * ple_layer_index
        self.mult = torch.tensor([2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1
                                  for i in range(self.n)], dtype=torch.long)
        self.sizes = torch.tensor(sizes, dtype=torch.long)
        self.offs = torch.tensor(offs, dtype=torch.long)

    def _shift(self, ids, shift):
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = torch.arange(T, dtype=torch.long)
        eosp = torch.where(ids == self.eos, pos, -1)
        prev_incl = torch.cummax(eosp, dim=1).values
        prev = torch.cat([eosp.new_full((B, 1), -1), prev_incl[:, :-1]], dim=1)
        pis = pos.unsqueeze(0) - (prev + 1)
        src = pos - shift
        sh = ids.gather(1, src.clamp_min(0).unsqueeze(0).expand(B, -1))
        valid = (pis >= shift) & (src.unsqueeze(0) >= 0)
        return torch.where(valid, sh, ids.new_full((), self.eos))

    def __call__(self, prev_ctx, ids):
        """prev_ctx: list[int] длины ctx (последние токены до ids; в начале — eos) · ids: list[int] -> [T, heads] long"""
        hist = torch.tensor([list(prev_ctx) + list(ids)], dtype=torch.long)
        sh = [self._shift(hist, s) for s in range(self.n)]
        blocks = []
        for ng in range(2, self.n + 1):
            a, b = (ng - 2) * self.hpn, (ng - 1) * self.hpn
            mixed = sh[0] * self.mult[0]
            for p in range(1, ng):
                mixed = torch.bitwise_xor(mixed, sh[p] * self.mult[p])
            r = torch.remainder(mixed.unsqueeze(-1), self.sizes[a:b].view(1, 1, -1))
            blocks.append(r + self.offs[a:b].view(1, 1, -1))
        return torch.cat(blocks, -1)[0, -len(ids):]


class NGramTable:
    """строки таблицы (FP8) читаются выборочно из model-fp8-mtp-ple.safetensors; раскладка проверяется при открытии"""

    def __init__(self, path, n_parts, dim, total_rows):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        pre = PFX + "layers.1.ple.ple_embedding.ngram_embedding."
        self.parts = []
        for i in range(n_parts):
            v = hdr[f"{pre}shard_{i}.weight"]
            assert v["dtype"] == "F8_E4M3" and v["shape"][1] == dim, v
            self.parts.append((base + v["data_offsets"][0], v["shape"][0]))
        self.scales = {k: v for k, v in hdr.items() if k.startswith(pre) and "scale" in k}
        rows = sum(p[1] for p in self.parts)
        assert rows >= total_rows, (rows, total_rows)
        self.starts = np.cumsum([0] + [p[1] for p in self.parts])
        self.dim = dim
        self.fd = os.open(path, os.O_RDONLY)
        self.path = path
        self.hdr_base = base
        self.hdr = hdr
        self.scale = self._load_scale()

    def _load_scale(self):
        """масштаб FP8 таблицы: modelopt FP8 (без групп) = один weight_scale на тензор-кусок -> храним по кускам"""
        sc = []
        for i in range(len(self.parts)):
            k = PFX + f"layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight_scale"
            v = self.hdr.get(k)
            if v is None:
                sc.append(None); continue
            a, b = v["data_offsets"]
            raw = os.pread(self.fd, b - a, self.hdr_base + a)
            assert v["dtype"] == "F32" and len(raw) == 4, v
            sc.append(struct.unpack("<f", raw)[0])
        if all(s is None for s in sc):
            k = PFX + "layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
            v = self.hdr.get(k)
            assert v is not None, "не нашёл масштаб FP8 таблицы n-грамм: " + str(list(self.scales)[:5])
            a, b = v["data_offsets"]
            raw = os.pread(self.fd, b - a, self.hdr_base + a)
            # ⛔02.10: масштаб здесь ОДИН BF16 (2 байта) — читал 4 байта как f32 => мусор => «!!!!»; тип — из заголовка
            if v["dtype"] == "BF16":
                assert len(raw) == 2, v
                s = struct.unpack("<f", b"\x00\x00" + raw)[0]
            elif v["dtype"] == "F32":
                assert len(raw) == 4, v
                s = struct.unpack("<f", raw)[0]
            else:
                raise AssertionError(v)
            sc = [s] * len(self.parts)
        print(f"таблица n-грамм: масштаб FP8 {sc[0]:.6g}", flush=True)
        assert all(s is not None for s in sc), sc
        return sc

    def rows(self, idx):
        """idx [T, H] long -> fp32 [T, H, dim]"""
        flat = idx.flatten().tolist()
        out = np.empty((len(flat), self.dim), dtype=np.uint8)
        sc = np.empty(len(flat), dtype=np.float32)
        for i, r in enumerate(flat):
            p = int(np.searchsorted(self.starts, r, side="right") - 1)
            off, _ = self.parts[p]
            buf = os.pread(self.fd, self.dim, off + (r - int(self.starts[p])) * self.dim)
            out[i] = np.frombuffer(buf, dtype=np.uint8)
            sc[i] = self.scale[p]
        t = torch.from_numpy(out).view(torch.float8_e4m3fn).float() * torch.from_numpy(sc)[:, None]
        return t.view(*idx.shape, self.dim)


# ============================================================ эксперты с диска (v1: синхронно)
# #324: Core Ultra 5 225F гибридный (0-5 производительные, 6-9 энергоэффективные). Шаг упирается в Python одного потока:
# не закреплённый, он скачет по ядрам ⇒ 93-108 мс/слово; на ядре 0 ровно 81.6. QW_PIN=ядро генерирующего потока
# (ставит сервер), потоки чтения — на все остальные (иначе унаследуют одно ядро от создавшего их потока).
PIN = os.environ.get("QW_PIN")


def io_affinity():
    if PIN not in (None, ""):
        os.sched_setaffinity(0, set(range(os.cpu_count())) - {int(PIN)})

class ExpertCache:
    """v2: эксперты с /fast — RAM-кэш LRU (закреплённая память) + чтение промахов в пул потоков (O_DIRECT, один pread
    на эксперта) + копия на карту в отдельном потоке CUDA; две банки слотов (чётный/нечётный слой); предвыборка P
    экспертов следующего слоя в «пузырь» (как DeepSeek v4). Шаг: start() -> [счёт общего эксперта] -> finish() -> ядро -> done()."""

    def __init__(self, K, dev, ram_gb=0.0, threads=32, pf=0, spec=0):
        self.K, self.pf, self.dev, self.spec = K, pf, dev, spec
        self.fd = os.open(f"{FAST}/experts.bin", os.O_RDONLY | os.O_DIRECT)
        self.n_ram = int(ram_gb * 2**30 // qmoe.BLOCK)
        nb = 2 * K + 2 * pf + 4
        self.buf = pinned_aligned((self.n_ram + nb) * qmoe.BLOCK).view(self.n_ram + nb, qmoe.BLOCK)
        self.where, self.lru, self.free = {}, collections.OrderedDict(), list(range(self.n_ram))
        self.bfree = collections.deque(range(self.n_ram, self.n_ram + nb))
        self.row_ev, self.pend, self.inflight = {}, {}, {}
        # #324: + две банки по spec слотов — эксперты след. слоя, скопированные на карту ЗАРАНЕЕ по прогнозу роутера
        # (шина 47 мс/слово шла ПОСЛЕ счёта; так копия идёт, пока Python выдаёт ядра текущего слоя)
        self.slots = torch.empty(2 * K + 2 * spec, qmoe.BLOCK, dtype=torch.uint8, device=dev)
        self.spec_map = {}                                         # слой -> {эксперт: слот}, забирается в start()
        self.sidx_h = torch.empty(64, K, dtype=torch.int32).pin_memory()   # кольцо по слою: перезапись через слово,
        self.sidx_d = torch.empty(64, K, dtype=torch.int32, device=dev)    # а за слово 48 синхронизаций — копия давно дошла
        self.sidx = [torch.arange(b * K, (b + 1) * K, dtype=torch.int32, device=dev) for b in (0, 1)]
        self.bank_ev = [None, None]
        self.cs = torch.cuda.Stream()
        self.pool = cf.ThreadPoolExecutor(threads, initializer=io_affinity)
        self.lock = threading.RLock()          # RLock: обработчик конца чтения может вызваться внутри drop_pf под замком
        scal = np.load(f"{FAST}/experts_scal.npy")
        self.s2 = torch.from_numpy(scal[:, :, :3].copy()).to(dev)          # ws2 gate/up/down [48,512,3]
        self.scratch = pinned_aligned(qmoe.BLOCK)
        self.stat = collections.Counter()
        self.read_s = 0.0
        self.nread = 0
        print(f"эксперты: RAM-кэш {self.n_ram} ({self.n_ram * qmoe.BLOCK / 2**30:.1f} ГиБ, "
              f"{self.n_ram / (48 * 512) * 100:.0f}% всех), предвыборка P={pf}, потоков {threads}", flush=True)

    # ---- строки RAM
    def _alloc(self, k, protect):
        """под замком: строка для нового ключа (кэш с вытеснением LRU или строка отражения)"""
        if self.n_ram:
            if self.free:
                r = self.free.pop()
            else:
                victim = next(v for v in self.lru if v not in protect and v not in self.pend
                              and not (v in self.inflight and not self.inflight[v].done()))
                r = self.where.pop(victim); del self.lru[victim]
            self.where[k] = r; self.lru[k] = None
            return r
        assert self.bfree, "нет свободной строки отражения"
        return self.bfree.popleft()

    def _read(self, r, k):
        ev = self.row_ev.pop(r, None)
        if ev is not None:
            ev.synchronize()                      # строку ещё копируют на карту
        got = os.preadv(self.fd, [memoryview(self.buf[r].numpy())], (k[0] * 512 + k[1]) * qmoe.BLOCK)
        assert got == qmoe.BLOCK, got

    def _release(self, r):
        if r >= self.n_ram:
            with self.lock:
                self.bfree.append(r)

    def _copy(self, slot, r):
        with torch.cuda.stream(self.cs):
            self.slots[slot].copy_(self.buf[r], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.cs)
        self.row_ev[r] = ev

    # ---- шаг одного токена
    def start(self, L, experts):
        bank = L % 2
        base = bank * self.K
        if self.bank_ev[bank] is not None:
            self.cs.wait_event(self.bank_ev[bank])                 # слоты банки свободны (ядро слоя L-2 досчитало)
        prot = {(L, e) for e in experts}
        todo = []
        sm = self.spec_map.pop(L, None) if self.spec else None
        slot = [base + j for j in range(len(experts))]
        with self.lock:
            for j, e in enumerate(experts):
                k = (L, e)
                if sm and e in sm:                                 # уже на карте (копия слоем раньше)
                    slot[j] = sm[e]; self.stat["spec_hit"] += 1
                    if k in self.lru:
                        self.lru.move_to_end(k)
                elif k in self.pend:
                    r, f = self.pend.pop(k); self.stat["pf_hit"] += 1
                    todo.append((j, r, f))
                elif k in self.inflight and not self.inflight[k].done():
                    self.stat["late_hit"] += 1                       # догадка слоя-раньше ещё дочитывается
                    self.lru.move_to_end(k)
                    todo.append((j, self.where[k], self.inflight[k]))
                elif k in self.where:
                    self.lru.move_to_end(k); self.stat["ram_hit"] += 1
                    todo.append((j, self.where[k], None))
                else:
                    r = self._alloc(k, prot); self.stat["disk"] += 1
                    todo.append((j, r, self.pool.submit(self._read, r, k)))
        for j, r, f in todo:                                       # попадания — на карту сразу
            if f is None:
                self._copy(base + j, r)
        self.nread += sum(1 for t in todo if t[2] is not None)
        if self.spec:
            self.sidx_h[L].copy_(torch.tensor(slot, dtype=torch.int32))
            self.sidx_d[L].copy_(self.sidx_h[L], non_blocking=True)
            return (L, bank, todo, prot, self.sidx_d[L])
        return (L, bank, todo, prot, self.sidx[bank])

    def finish(self, h):
        L, bank, todo, prot, sidx = h
        base = bank * self.K
        t = time.time()
        waits = {f: (j, r) for j, r, f in todo if f is not None}
        for f in cf.as_completed(list(waits)):                     # копия каждого — как только прочитан
            f.result()
            j, r = waits[f]
            self._copy(base + j, r)
            self._release(r)
        self.read_s += time.time() - t
        ev = torch.cuda.Event(); ev.record(self.cs)
        torch.cuda.current_stream().wait_event(ev)
        return sidx

    def done(self, L):
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
        self.bank_ev[L % 2] = ev

    def speculate(self, L, cands):
        """#324: до spec первых кандидатов слоя L, уже лежащих в RAM, — копия на карту сейчас (в поток копий ПОСЛЕ
        копий текущего слоя ⇒ его счёт не задерживает). Летящие/читаемые с диска пропускаются."""
        b = L % 2
        base = 2 * self.K + b * self.spec
        if self.bank_ev[b] is not None:
            self.cs.wait_event(self.bank_ev[b])                    # слот-банку читало ядро слоя L-2
        m = {}
        with self.lock:
            for e in cands:
                if len(m) >= self.spec:
                    break
                k = (L, e)
                r = self.where.get(k)
                if r is None or k in self.pend or (k in self.inflight and not self.inflight[k].done()):
                    continue
                m[e] = base + len(m)
                self._copy(m[e], r)
        self.spec_map[L] = m
        self.stat["spec_copy"] += len(m)

    def prefetch(self, L, cands, protect=()):
        """заранее прочитать до P экспертов слоя L (кандидаты по убыванию); уже лежащие в RAM/летящие не считаются"""
        n = 0
        with self.lock:
            for e in cands:
                if n >= self.pf:
                    break
                k = (L, e)
                if k in self.where or k in self.pend:
                    continue
                if not self.n_ram and not self.bfree:
                    break
                r = self._alloc(k, set(protect)); self.stat["pf_read"] += 1
                self.pend[k] = (r, self.pool.submit(self._read, r, k)); n += 1
        self.nread += n

    def drop_pf(self, L):
        """неугаданные заранее прочитанные эксперты слоя L — БЕЗ ожидания чтения (ожидание стопорило шаг):
        RAM-режим — запись остаётся в кэше, пока летит помечена inflight (start/вытеснение её дождутся);
        режим отражения — строка вернётся в свободные по окончании чтения"""
        with self.lock:
            ks = [k for k in self.pend if k[0] == L]
            for k in ks:
                r, f = self.pend.pop(k); self.stat["pf_waste"] += 1
                if self.n_ram:
                    self.inflight[k] = f
                else:
                    f.add_done_callback(lambda _f, r=r: self._release(r))
            for k in [k for k, f in self.inflight.items() if f.done()]:
                del self.inflight[k]

    # ---- путь промта
    PB = 32

    def prefill_batch(self, L, experts):
        """до PB экспертов слоя: строки RAM-кэша (попадания) или своё кольцо промта (промахи, читаются параллельно);
        в кэш промахи промта не кладутся — промт не должен вымывать кэш шага"""
        if not hasattr(self, "pbuf"):
            self.pbuf = pinned_aligned(self.PB * qmoe.BLOCK).view(self.PB, qmoe.BLOCK)
        out, jobs = [], []
        with self.lock:
            for i, e in enumerate(experts):
                k = (L, e)
                f = self.pend.get(k, (None, None))[1] or self.inflight.get(k)
                r = self.where.get(k)
                if r is not None and f is None:
                    ev = self.row_ev.get(r)
                    out.append(("ram", r, ev))
                else:
                    out.append(("own", i, f))
                    jobs.append((i, k))
        t = time.time()
        list(self.pool.map(lambda ik: self._pread_into(self.pbuf[ik[0]], ik[1]), jobs))
        self.read_s += time.time() - t
        res = []
        for kind, r, extra in out:
            if kind == "ram":
                if extra is not None:
                    extra.synchronize()
                res.append(self.buf[r])
            else:
                res.append(self.pbuf[r])
        return res

    def _pread_into(self, dst, k):
        got = os.preadv(self.fd, [memoryview(dst.numpy())], (k[0] * 512 + k[1]) * qmoe.BLOCK)
        assert got == qmoe.BLOCK, got

    def block_host(self, L, e):
        k = (L, e)
        with self.lock:
            f = self.pend.get(k, (None, None))[1] or self.inflight.get(k)
            r = self.where.get(k)
        if f is not None:
            f.result()
        if r is not None:
            ev = self.row_ev.get(r)
            if ev is not None:
                ev.synchronize()
            return self.buf[r]
        got = os.preadv(self.fd, [memoryview(self.scratch.numpy())], (L * 512 + e) * qmoe.BLOCK)
        assert got == qmoe.BLOCK
        return self.scratch


# ============================================================ модель
class QwenEngine:
    def __init__(self, dev="cuda", fp8=("linear_attn",), max_ctx=8192, ngram=True, ram_gb=0.0, pf=0, threads=32, spec=0):
        self.dev = dev
        cfg = json.load(open(f"{SRC}/config.json"))
        tc = self.tc = cfg["text_config"]
        self.H = tc["hidden_size"]; self.hc = tc["hc_count"]; self.eps = tc["rms_norm_eps"]
        self.NL = tc["num_hidden_layers"]; self.types = tc["layer_types"]
        self.topk = tc["num_experts_per_tok"]
        self.max_ctx = max_ctx
        if not os.path.exists(nonexpert_path()):
            print("извлекаю постоянную часть на /fast ...", flush=True)
            print("тензоров:", extract_nonexpert(), flush=True)
        self.fp8 = set(fp8)
        W = {}
        with safe_open(nonexpert_path(), "pt") as fh:
            for k in fh.keys():
                W[k] = fh.get_tensor(k)
        self.embed = W.pop(PFX + "embed_tokens.weight")          # [V, H] bf16, остаётся в RAM
        g = lambda k: W[PFX + k]
        big = lambda k: Lin(W[PFX + k], self._is_fp8(k), dev)
        small = lambda k: W[PFX + k].to(dev)
        self.lm_head = Lin(W["lm_head.weight"], "lm_head" in self.fp8, dev)
        self.mixer = self._hc("hyper_connection_mixer", small, big, combine=False)
        self.layers = []
        for i in range(self.NL):
            p = f"layers.{i}."
            Ld = {"type": self.types[i],
                  "ahc": self._hc(p + "attn_hyper_connection", small, big),
                  "mhc": self._hc(p + "mlp_hyper_connection", small, big),
                  "router": small(p + "mlp.gate.weight"),
                  "sh_gate": big(p + "mlp.shared_expert.gate_proj.weight"),
                  "sh_up": big(p + "mlp.shared_expert.up_proj.weight"),
                  "sh_down": big(p + "mlp.shared_expert.down_proj.weight"),
                  "sh_g": small(p + "mlp.shared_expert_gate.weight")}
            if self.types[i] == "linear_attention":
                a = p + "linear_attn."
                Ld.update(qkv=big(a + "in_proj_qkv.weight"), z=big(a + "in_proj_z.weight"),
                          pb=small(a + "in_proj_b.weight"), pa=small(a + "in_proj_a.weight"),
                          conv=small(a + "conv1d.weight").squeeze(1), A_log=small(a + "A_log"),
                          dt_bias=small(a + "dt_bias"), gnorm=small(a + "norm.weight"), out=big(a + "out_proj.weight"))
            else:
                a = p + "self_attn."
                Ld.update(q=big(a + "q_proj.weight"), k=big(a + "k_proj.weight"), v=big(a + "v_proj.weight"),
                          o=big(a + "o_proj.weight"), qn=small(a + "q_norm.weight"), kn=small(a + "k_norm.weight"),
                          iqk=big(a + "indexer.index_qk_proj.weight"),
                          iqn=small(a + "indexer.q_layernorm.weight"), ikn=small(a + "indexer.k_layernorm.weight"))
            if (PFX + p + "ple.key_proj.weight") in W:
                e = p + "ple."
                Ld["ple"] = dict(key=big(e + "key_proj.weight"), val=big(e + "value_proj.weight"),
                                 nk=small(e + "norm_key.weight"), nq=small(e + "norm_query.weight"),
                                 nc=small(e + "norm_conv.weight"), conv=small(e + "conv1d.weight").squeeze(1))
            self.layers.append(Ld)
        ckb = {n: W[PFX + "layers.1.ple.ple_embedding." + n] for n in
               ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")}
        del W
        # RoPE (текст: все три оси mRoPE совпадают ⇒ обычный RoPE на первых 64 измерениях головы)
        rp = tc["rope_parameters"]
        rd = int(tc["head_dim"] * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
        self.ngi = NGramIds(tc)
        assert torch.equal(ckb["layer_multipliers"], self.ngi.mult), (ckb["layer_multipliers"], self.ngi.mult)
        assert torch.equal(ckb["ngram_heads_vocab_sizes"], self.ngi.sizes)
        assert torch.equal(ckb["ngram_heads_offsets"], self.ngi.offs)
        self.ngt = None
        if ngram:
            # таблица только с /fast (NVMe): на HDD 16 случайных строк/слово = ~16 поисков дорожки (судья j0055)
            self.ngt = NGramTable(f"{FAST}/model-fp8-mtp-ple.safetensors", tc["split_ngram_parts"],
                                  tc["ple_embed_dim"] // self.ngi.heads, self.ngi.total)
        self.store = ExpertCache(self.topk, dev, ram_gb=ram_gb, threads=threads, pf=pf, spec=spec)
        self.pf = pf
        self.no_experts = False
        self.ple_read_s = 0.0
        self.hbuf = torch.empty(self.topk, 640, device=dev)
        self.obuf = torch.empty(self.topk, 2560, device=dev)
        self.reset()

    def _is_fp8(self, k):
        if "all" in self.fp8:
            return True
        return any(t in k for t in self.fp8)

    def _hc(self, p, small, big, combine=True):
        return dict(norm=small(p + ".hc_norm.weight"), down=big(p + ".input_mix_weight_down.weight"),
                    up=big(p + ".input_mix_weight_up.weight"),
                    inj=small(p + ".block_inject_weight.weight") if combine else None)

    def vram_report(self):
        return torch.cuda.memory_allocated() / 2**30

    # ---------------------------------------------------------------- состояние
    def reset(self):
        self.pos = 0
        self.tokens = []
        self.st = []
        for Ld in self.layers:
            if Ld["type"] == "linear_attention":
                self.st.append(dict(conv=None, S=None))
            else:
                self.st.append(dict(k=None, v=None, ik=None, bk=None))
        self.ple_conv = None                      # последние 9 gated_value_normed [9, 4H]

    def rope(self, positions):
        f = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([f, f], -1)
        return emb.cos().to(self.dev, torch.bfloat16), emb.sin().to(self.dev, torch.bfloat16)

    # ---------------------------------------------------------------- блоки
    def gated_residual(self, P, hyp):
        """hyp [T, 4H] -> mixed [T,H] (+ inj [T,4])"""
        hn = rmsnorm(hyp, P["norm"], self.eps, group=self.H)
        m = F.silu(P["down"](hn) / self.hc)
        m = torch.sigmoid(P["up"](m)).unflatten(-1, (self.hc, self.H))
        mixed = (m * hn.unflatten(-1, (self.hc, self.H))).mean(dim=-2)
        if P["inj"] is None:
            return mixed, None
        return mixed, 2 * torch.sigmoid(F.linear(hn, P["inj"]) / self.hc)

    DN_CHUNK = 512                                               # кратно 64 (блок алгоритма) ⇒ та же арифметика

    def deltanet(self, Ld, st, x):
        T = x.shape[0]
        if T > self.DN_CHUNK:                                    # длинный промт: память fp32 ~[48, T, 128] x10 — кусками
            return torch.cat([self.deltanet(Ld, st, x[a:a + self.DN_CHUNK]) for a in range(0, T, self.DN_CHUNK)], 0)
        nk, nv, dk, dv = self.tc["linear_num_key_heads"], self.tc["linear_num_value_heads"], \
            self.tc["linear_key_head_dim"], self.tc["linear_value_head_dim"]
        kd, vd = nk * dk, nv * dv
        qkv = Ld["qkv"](x)                                       # [T, 2kd+vd] bf16
        z = Ld["z"](x).reshape(T, -1, dv)
        b = F.linear(x, Ld["pb"]); a = F.linear(x, Ld["pa"])
        # причинная свёртка: предыдущие 3 входа (нули в начале) + текущие
        prev = st["conv"] if st["conv"] is not None else qkv.new_zeros(3, qkv.shape[1])
        full = torch.cat([prev, qkv], 0)                         # [3+T, C]
        st["conv"] = full[-3:].clone()
        c = F.conv1d(full.T.unsqueeze(0), Ld["conv"].unsqueeze(1), None, padding=0, groups=full.shape[1])
        qkv = F.silu(c)[0].T.to(torch.bfloat16)                  # [T, C]
        q, k, v = torch.split(qkv, [kd, kd, vd], dim=-1)
        q = q.reshape(1, T, nk, dk); k = k.reshape(1, T, nk, dk); v = v.reshape(1, T, nv, dv)
        beta = b.sigmoid().unsqueeze(0)
        g = (-Ld["A_log"].float().exp() * F.softplus(a.float() + Ld["dt_bias"])).unsqueeze(0)
        r = nv // nk
        q = q.repeat_interleave(r, dim=2); k = k.repeat_interleave(r, dim=2)
        if st["S"] is not None and T == 1:
            o, S = recurrent_gated_delta_rule(q, k, v, g, beta, st["S"])
        else:
            o, S = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=st["S"])
        st["S"] = S
        o = o.reshape(-1, dv); z = z.reshape(-1, dv)
        of = o.float()
        o = (Ld["gnorm"] * (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16))
        o = (o * torch.sigmoid(z.float())).to(torch.bfloat16).reshape(T, -1)
        return Ld["out"](o)

    def indexer_mask(self, Ld, st, x, cs, sn):
        """= Qwen4ExpTextQSAIndexer эталона для одной последовательности без паддинга -> bool [T, S] или None.
        Блок = 4 подряд идущих токена; ключ блока = среднее сырых ключей (fp32 -> bf16), k_layernorm, RoPE по позиции
        начала блока. Запрос p видит (p+1)//4 полных блоков: если их <= 512 — берутся все (= полное причинное
        внимание, возвращаю None), иначе top-512 по relu(q·k) суммой по 4 головам / sqrt(128) + хвост неполного блока."""
        T = x.shape[0]
        ihd, inh = self.tc["indexer_head_dim"], self.tc["indexer_n_heads"]
        cr = self.tc["indexer_compress_ratio"]
        btop = self.tc["indexer_budget"] // cr
        qk = Ld["iqk"](x)
        q, rk = torch.split(qk, [inh * ihd, ihd], dim=-1)
        st["ik"] = rk if st.get("ik") is None else torch.cat([st["ik"], rk], 0)     # сырые ключи [S, 128]
        S = st["ik"].shape[0]
        nb = S // cr
        if nb <= btop:
            return None
        q = apply_rope(rmsnorm(q.view(T, inh, ihd), Ld["iqn"], self.eps), cs, sn)    # [T, 4, 128]
        have = 0 if st.get("bk") is None else st["bk"].shape[0]
        if nb > have:                                                                # дописать ключи новых полных блоков
            raw = st["ik"][have * cr: nb * cr].view(nb - have, cr, ihd)
            pooled = rmsnorm(raw.float().mean(dim=1).to(raw.dtype), Ld["ikn"], self.eps)
            c, s_ = self.rope(torch.arange(have, nb) * cr)
            bk = apply_rope(pooled.unsqueeze(1), c[:, None, :], s_[:, None, :]).squeeze(1)
            st["bk"] = bk if st.get("bk") is None else torch.cat([st["bk"], bk], 0)
        bk = st["bk"]                                                                # [nb, 128]
        sc = torch.relu(torch.einsum("thd,bd->tbh", q.float(), bk.float())).sum(-1) / math.sqrt(ihd)   # [T, nb]
        p = torch.arange(S - T, S, device=x.device)
        ncb = (p + 1) // cr                                                          # полных блоков у запроса
        bidx = torch.arange(nb, device=x.device)
        sc = sc.masked_fill(bidx[None, :] >= ncb[:, None], float("-inf"))
        sel_b = torch.zeros(T, nb, dtype=torch.bool, device=x.device)
        many = ncb > btop
        if many.any():
            top = sc[many].topk(btop, dim=-1).indices
            sel_b[many] = torch.zeros_like(sel_b[many]).scatter(1, top, True)
        sel_b[~many] = bidx[None, :] < ncb[~many, None]
        j = torch.arange(S, device=x.device)
        jb = (j // cr).clamp_max(nb - 1)
        in_block = sel_b.gather(1, jb[None, :].expand(T, -1)) & (j[None, :] < (ncb * cr)[:, None])
        tail = (j[None, :] >= (ncb * cr)[:, None]) & (j[None, :] <= p[:, None])
        return in_block | tail

    def attention(self, Ld, st, x, cos, sin):
        T = x.shape[0]
        hd, nh, nkv = self.tc["head_dim"], self.tc["num_attention_heads"], self.tc["num_key_value_heads"]
        assert self.pos + T <= self.max_ctx, f"контекст {self.pos + T} > max_ctx {self.max_ctx}"
        cs, sn = cos[:, None, :], sin[:, None, :]
        sel = self.indexer_mask(Ld, st, x, cs, sn)
        qg = Ld["q"](x).view(T, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = rmsnorm(q, Ld["qn"], self.eps)                       # [T, nh, hd]
        k = rmsnorm(Ld["k"](x).view(T, -1, hd), Ld["kn"], self.eps)
        v = Ld["v"](x).view(T, -1, hd)
        q = apply_rope(q, cs, sn); k = apply_rope(k, cs, sn)
        st["k"] = k if st["k"] is None else torch.cat([st["k"], k], 0)
        st["v"] = v if st["v"] is None else torch.cat([st["v"], v], 0)
        K, V = st["k"], st["v"]                                  # [S, nkv, hd]
        Sn = K.shape[0]
        qh = q.transpose(0, 1).unsqueeze(0)                      # [1, nh, T, hd]
        kh = K.transpose(0, 1).unsqueeze(0); vh = V.transpose(0, 1).unsqueeze(0)
        mask = sel if sel is not None else torch.ones(T, Sn, dtype=torch.bool, device=x.device).tril(Sn - T)
        # блоками запросов: матрица весов [24, QB, S], не [24, T, S]. #324: QB падает с длиной контекста — при S 6144
        # и QB 256 временные веса не влезали в ~1.1 ГиБ свободной памяти (промт 7182 = отказ). Строки независимы.
        QB = int(os.environ.get("QW_QB") or 0) or (256 if Sn <= 2048 else 128 if Sn <= 4096 else 64)
        o = torch.cat([F.scaled_dot_product_attention(qh[:, :, q0:q0 + QB], kh, vh, attn_mask=mask[q0:q0 + QB],
                                                      scale=hd ** -0.5, enable_gqa=True)
                       for q0 in range(0, T, QB)], dim=2)
        o = o[0].transpose(0, 1).reshape(T, -1)
        o = o * torch.sigmoid(gate)
        return Ld["o"](o)

    def moe(self, Ld, L, x):
        T = x.shape[0]
        logits = F.linear(x, Ld["router"])
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        tv, ti = torch.topk(probs, self.topk, dim=-1)
        tv = (tv / tv.sum(-1, keepdim=True)).to(logits.dtype)
        if T == 1 and not self.no_experts:
            nxt = L + 1 < self.NL and (self.pf > 0 or self.store.spec > 0)
            if nxt:                                              # прогноз следующего слоя: его роутер по входу MoE этого
                nc = max(self.topk + 2, self.store.spec)
                pc = torch.topk(F.linear(x, self.layers[L + 1]["router"]), nc, dim=-1).indices
                both = torch.cat([ti[0], pc[0]]).tolist()        # одна синхронизация на слой
                idx, cands = both[:self.topk], both[self.topk:]
            else:
                idx = ti[0].tolist()
            h = self.store.start(L, idx)
            sh = Ld["sh_down"](F.silu(Ld["sh_gate"](x)) * Ld["sh_up"](x))     # счёт общего эксперта — пока летит диск
            sh = torch.sigmoid(F.linear(x, Ld["sh_g"])) * sh
            sidx = self.store.finish(h)
            if nxt and self.store.spec:                          # #324: на карту заранее (до prefetch: тот пропускает RAM)
                self.store.speculate(L + 1, cands)
            if nxt and self.pf > 0:                              # в «пузырь»: нужные чтения слоя уже закончены
                self.store.prefetch(L + 1, cands, protect={(L, e) for e in idx})
            s2 = self.store.s2[L, ti[0]]
            y = qmoe.moe_decode(self.store.slots, sidx, s2, tv[0].float(), x[0], self.hbuf, self.obuf)
            self.store.done(L)
            if L > 0 and self.pf > 0:
                self.store.drop_pf(L)
            return y.to(torch.bfloat16).unsqueeze(0) + sh
        sh = Ld["sh_down"](F.silu(Ld["sh_gate"](x)) * Ld["sh_up"](x))
        sh = torch.sigmoid(F.linear(x, Ld["sh_g"])) * sh
        if self.no_experts:
            return sh
        else:
            out = torch.zeros(T, self.H, dtype=torch.float32, device=self.dev)
            ti_c, tv_c = ti.cpu(), tv.float()
            s2c = self.store.s2[L].cpu()
            uniq = sorted(set(ti_c.flatten().tolist()))
            for b0 in range(0, len(uniq), self.store.PB):        # пачка экспертов: чтение параллельно, счёт по одному
                batch = uniq[b0:b0 + self.store.PB]
                rows = self.store.prefill_batch(L, batch)
                for e, host in zip(batch, rows):
                    tok, slot = torch.where(ti_c == e)
                    blk = host.to(self.dev, non_blocking=True)
                    gw, uw, dw = qmoe.block_deq(blk, s2c[e])
                    td = tok.to(self.dev)
                    xs = x[td].float()
                    h = F.silu(xs @ gw.T) * (xs @ uw.T)
                    out.index_add_(0, td, (h @ dw.T) * tv_c[td, slot.to(self.dev), None])
                torch.cuda.current_stream().synchronize()          # строки пачки можно перезаписывать
                self.store.nread += len(batch)
            out = out.to(torch.bfloat16)
        return out + sh

    def ple(self, P, hyp, ids):
        T = len(ids)
        prev = self.tokens[-self.ngi.ctx:] if self.tokens else []
        prev = [self.ngi.eos] * (self.ngi.ctx - len(prev)) + prev
        t0 = time.time()
        rows = self.ngi(prev, ids)                               # [T, 16]
        emb = self.ngt.rows(rows)
        self.ple_read_s += time.time() - t0                      # хеш + чтение строк, без счёта (судья j0055)
        emb = emb.to(self.dev, torch.bfloat16).flatten(-2)       # [T, 2560]
        kn = rmsnorm(P["key"](emb), P["nk"], self.eps, group=self.H).unflatten(-1, (self.hc, self.H))
        val = P["val"](emb)
        qn = rmsnorm(hyp, P["nq"], self.eps, group=self.H).unflatten(-1, (self.hc, self.H))
        gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val.unsqueeze(-2)
        gvn = rmsnorm(gv.flatten(-2), P["nc"], self.eps, group=self.H)
        gv = gv.flatten(-2)
        prevc = self.ple_conv if self.ple_conv is not None else gvn.new_zeros(9, gvn.shape[1])
        full = torch.cat([prevc, gvn], 0)                        # [9+T, 4H]
        self.ple_conv = full[-9:].clone()
        c = F.conv1d(full.T.unsqueeze(0), P["conv"].unsqueeze(1), None, groups=full.shape[1], dilation=3)
        return gv + F.silu(c)[0].T

    # ---------------------------------------------------------------- проход
    @torch.no_grad()
    def forward(self, ids, all_logits=False, logits_from=0):
        """ids: list[int] новых токенов -> logits последнего [V] fp32 (all_logits: [T - logits_from, V])"""
        T = len(ids)
        x = self.embed[torch.tensor(ids)].to(self.dev)           # [T, H] bf16
        cos, sin = self.rope(torch.arange(self.pos, self.pos + T))
        hyp = x.repeat(1, self.hc)                               # [T, 4H]
        for L, Ld in enumerate(self.layers):
            if "ple" in Ld and self.ngt is not None:       # без таблицы (ngram=False) — PLE выключен (только для сверки)
                hyp = hyp + self.ple(Ld["ple"], hyp, ids)
            h, inj = self.gated_residual(Ld["ahc"], hyp)
            if Ld["type"] == "linear_attention":
                h = self.deltanet(Ld, self.st[L], h)
            else:
                h = self.attention(Ld, self.st[L], h, cos, sin)
            hyp = hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
            h, inj = self.gated_residual(Ld["mhc"], hyp)
            h = self.moe(Ld, L, h)
            hyp = hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        mixed, _ = self.gated_residual(self.mixer, hyp)
        logits = self.lm_head(mixed[logits_from:] if all_logits else mixed[-1:])
        self.tokens += list(ids)
        self.pos += T
        return logits.float() if all_logits else logits[0].float()

    def generate(self, ids, new, stop=(248046, 248044)):
        out = []
        lg = self.forward(ids)
        for _ in range(new):
            t = int(lg.argmax())
            out.append(t)
            if t in stop:
                break
            lg = self.forward([t])
        return out
