import torch, time
from fp8gemv import gemv
from qwen_engine import to_fp8_rows
torch.manual_seed(0)
for N, K in ((10240, 2560), (6144, 2560), (2560, 6144)):
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    q, s = to_fp8_rows(w); q, s = q.cuda(), s.cuda().view(-1)
    x = torch.randn(K).to(torch.bfloat16).cuda()
    ref = (q.float() * s[:, None]) @ x.float()
    y = gemv(q, s, x); torch.cuda.synchronize()
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    old = lambda: torch.nn.functional.linear(x[None], (q.to(torch.bfloat16) * s[:, None]).to(torch.bfloat16))
    for f, nm in ((lambda: gemv(q, s, x), "ядро"), (old, "было")):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50): f()
        torch.cuda.synchronize(); dt = (time.time() - t) / 50
        print(f"{N}x{K} {nm}: {dt*1e6:.0f} мкс ({N*K/dt/1e9:.0f} ГБ/с fp8)" + (f", отн.ошибка {err:.1e}" if nm == "ядро" else ""))
