"""y[n] = s[n] · Σ_k q[n,k]·x[k] — FP8 E4M3 со строковым масштабом × вектор (шаг одного токена), накопление fp32."""
import torch, triton, triton.language as tl


@triton.jit
def _gemv(q_ptr, s_ptr, x_ptr, y_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    rm = rows < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(q_ptr + rows[:, None].to(tl.int64) * K + ks[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + ks).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rm, other=0.0)
    tl.store(y_ptr + rows, (acc * s).to(tl.bfloat16), mask=rm)


def gemv(q, s, x):
    """q [N,K] float8_e4m3fn · s [N] fp32 · x [K] bf16 -> [N] bf16"""
    N, K = q.shape
    y = torch.empty(N, dtype=torch.bfloat16, device=q.device)
    BN = 16
    _gemv[(triton.cdiv(N, BN),)](q, s, x, y, N, K=K, BN=BN, BK=256, num_warps=4)
    return y
