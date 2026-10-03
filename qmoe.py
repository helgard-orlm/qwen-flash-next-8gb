"""Эксперты Qwen3.8-Flash-Next (NVFP4, SwiGLU 2560->640->2560) на шаге одного токена: K экспертов в слотах на карте.
Слот = блок перепаковки (2 764 800 Б): gate_w | up_w | down_w | gate_s | up_s | down_s (см. repack_experts.py).
NVFP4 (modelopt): байт = два E2M1, младший полубайт = чётный элемент; масштаб E4M3 на 16 элементов; × weight_scale_2.
Активации не квантуются (W4A16): x bf16 -> fp32, накопление fp32. Аппаратная распаковка E2M1 как в nvfp4_moe3 (sm_120a)."""
import torch
import triton
import triton.language as tl

HID, INTER = 2560, 640
BLOCK = 2764800
O_GW, O_UW, O_DW = 0, 819200, 1638400
O_GS, O_US, O_DS = 2457600, 2560000, 2662400
KB_UP, G_UP = HID // 2, HID // 16          # байт и групп в строке gate/up (1280, 160)
KB_DN, G_DN = INTER // 2, INTER // 16      # в строке down (320, 40)


@triton.jit
def _e2m1_hw(c):
    r = tl.inline_asm_elementwise("{ .reg .b8 t, z; mov.b16 {t, z}, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
                                  "=r,h", [c.to(tl.uint16)], dtype=tl.uint32, is_pure=True, pack=1)
    return (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gate_up(slots_ptr, sidx_ptr, s2_ptr, x_ptr, h_ptr,
             BLOCKB: tl.constexpr, BN: tl.constexpr, GB: tl.constexpr):
    """h[j, n] = silu(gate_n · x) * (up_n · x) для эксперта в слоте sidx[j]"""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(sidx_ptr + j).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    rows = pid * BN + tl.arange(0, BN)
    accg = tl.zeros([BN], dtype=tl.float32)
    accu = tl.zeros([BN], dtype=tl.float32)
    for g0 in range(0, 160, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        xe = tl.load(x_ptr + 2 * bi).to(tl.float32)
        xo = tl.load(x_ptr + 2 * bi + 1).to(tl.float32)
        wg = tl.load(base + 0 + rows[:, None] * 1280 + bi[None, :]).to(tl.int32)
        wu = tl.load(base + 819200 + rows[:, None] * 1280 + bi[None, :]).to(tl.int32)
        sg = tl.load(base + 2457600 + rows[:, None] * 160 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        su = tl.load(base + 2560000 + rows[:, None] * 160 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        pg = _e2m1_hw(wg & 15) * xe[None, :] + _e2m1_hw(wg >> 4) * xo[None, :]
        pu = _e2m1_hw(wu & 15) * xe[None, :] + _e2m1_hw(wu >> 4) * xo[None, :]
        accg += tl.sum(tl.sum(tl.reshape(pg, [BN, GB, 8]), axis=2) * sg, axis=1)
        accu += tl.sum(tl.sum(tl.reshape(pu, [BN, GB, 8]), axis=2) * su, axis=1)
    g = accg * tl.load(s2_ptr + j * 3 + 0)
    u = accu * tl.load(s2_ptr + j * 3 + 1)
    h = g / (1.0 + tl.exp(-g)) * u
    tl.store(h_ptr + j * 640 + rows, h)


@triton.jit
def _down(slots_ptr, sidx_ptr, s2_ptr, coef_ptr, h_ptr, out_ptr,
          BLOCKB: tl.constexpr, BR: tl.constexpr, GB: tl.constexpr):
    """out[j, r] = coef_j * ws2d_j * Σ_n Wd[r, n] h[j, n]"""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(sidx_ptr + j).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    rows = pid * BR + tl.arange(0, BR)
    acc = tl.zeros([BR], dtype=tl.float32)
    for g0 in range(0, 40, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        he = tl.load(h_ptr + j * 640 + 2 * bi)
        ho = tl.load(h_ptr + j * 640 + 2 * bi + 1)
        w = tl.load(base + 1638400 + rows[:, None] * 320 + bi[None, :]).to(tl.int32)
        s = tl.load(base + 2662400 + rows[:, None] * 40 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        p = _e2m1_hw(w & 15) * he[None, :] + _e2m1_hw(w >> 4) * ho[None, :]
        acc += tl.sum(tl.sum(tl.reshape(p, [BR, GB, 8]), axis=2) * s, axis=1)
    c = tl.load(coef_ptr + j) * tl.load(s2_ptr + j * 3 + 2)
    tl.store(out_ptr + j * 2560 + rows, acc * c)


def moe_decode(slots_u8, sidx, s2, coef, x, h_buf, out_buf):
    """slots_u8 [S, BLOCK] u8 · sidx [K] int32 (слот j-го эксперта) · s2 [K,3] f32 (ws2 gate/up/down) ·
    coef [K] f32 (вес роутера) · x [2560] bf16/fp32 -> fp32 [2560] (сумма по K)"""
    K = sidx.shape[0]
    _gate_up[(INTER // 32, K)](slots_u8, sidx, s2, x, h_buf, BLOCKB=BLOCK, BN=32, GB=8, num_warps=4)
    _down[(HID // 64, K)](slots_u8, sidx, s2, coef, h_buf, out_buf, BLOCKB=BLOCK, BR=64, GB=8, num_warps=4)
    return out_buf[:K].sum(0)


# ---------- эталонная распаковка (torch) ----------
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


SWAP = False   # контрольная подмена: старший полубайт = чётный элемент (заведомо неверно, для проверки перплексией)


def deq(w_u8, s_e4m3_u8, ws2):
    """w_u8 [N, K/2] · s [N, K/16] (байты E4M3) · ws2 -> fp32 [N, K]"""
    t = E2M1.to(w_u8.device)
    lo = t[(w_u8 & 15).long()]
    hi = t[(w_u8 >> 4).long()]
    if SWAP:
        lo, hi = hi, lo
    w = torch.stack([lo, hi], -1).reshape(w_u8.shape[0], -1)
    s = s_e4m3_u8.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return w * s * ws2


def block_deq(blk_u8, s2):
    """блок перепаковки (u8 [BLOCK]) -> gate [640,2560], up [640,2560], down [2560,640] fp32"""
    b = blk_u8
    g = deq(b[O_GW:O_UW].view(640, 1280), b[O_GS:O_US].view(640, 160), float(s2[0]))
    u = deq(b[O_UW:O_DW].view(640, 1280), b[O_US:O_DS].view(640, 160), float(s2[1]))
    d = deq(b[O_DW:O_GS].view(2560, 320), b[O_DS:BLOCK].view(2560, 40), float(s2[2]))
    return g, u, d
