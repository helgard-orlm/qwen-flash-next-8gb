#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (свой движок qwen_engine.py, всё с /fast) — служба с API как у OpenAI (/v1/models,
/v1/chat/completions со стримом) для Open WebUI (:9800). Заказ helgard 02.10.2026 «берёмся за новый движок».
Образец — ds41_server.py (DeepSeek). Отличия:
  - состояние модели = eng.st (рекуррентные матрицы DeltaNet + KV + ключи индексатора) + свёртка PLE + история токенов;
    снимок = их копия в RAM ⇒ продолжение диалога дочитывает только новый хвост (пачкой — движок это умеет);
  - промт читается пачкой целиком (путь промта движка), ответ — по одному слову;
  - думание выключено (enable_thinking=False), жадный выбор слова — как чаты Nemotron/DeepSeek;
  - только текст (зрение и MTP не подключены).
Карта занята почти целиком (~6.2 ГиБ) ⇒ запуск из панели gpusvcs; DeepSeek и Qwen — по очереди."""
import collections, copy, json, os, sys, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("QW_PORT", "9805"))
RAM_GB = float(os.environ.get("QW_RAM_GB", "16"))
PF = int(os.environ.get("QW_PF", "2"))
SPEC = int(os.environ.get("QW_SPEC", "0"))          # #324: эксперты след. слоя на карту заранее
PIN = os.environ.get("QW_PIN", "")                  # #324: ядро генерирующего потока (производительное)
MAX_SEQ = int(os.environ.get("QW_MAX_SEQ", "8192"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QW_MAX_TOKENS", "2048"))
PREFILL_CHUNK = int(os.environ.get("QW_PREFILL_CHUNK", "3072"))       # промт длиннее — кусками (память карты; 2451 целиком: пик 6.83 ГиБ)


def prefill_chunk(pos):
    """#324: память промта растёт с (кусок × контекст): 3072×3072 проходил, 2048×6144 и 3072×6144 — отказ (промт 7182).
    Самый крупный кусок, укладывающийся в бюджет; не меньше 1024 — тогда qwen_delta_graph (порог ≥1024) всегда
    освобождает записи графов перед промтом."""
    for c in (PREFILL_CHUNK, 2048, 1536, 1024):
        if c <= PREFILL_CHUNK and c * (pos + c) <= PREFILL_CHUNK * PREFILL_CHUNK:
            return c
    return 1024
SNAP_N = int(os.environ.get("QW_SNAP_N", "8"))          # 2 снимка на ход (заголовок ответа + конец), ~120 МБ каждый
MODEL_ID = "qwen3.8-flash-next"
THINK_ID = MODEL_ID + "-think"               # думающий режим (<think>…</think>, Open WebUI сворачивает)
THINK_EFFORT = os.environ.get("QW_THINK_EFFORT", "medium")         # шаблон: low | medium | xhigh (его умолчание xhigh)
THINK_MAX_TOKENS = int(os.environ.get("QW_THINK_MAX_TOKENS", "6144"))

STATE = {"status": "loading", "error": None, "since": time.time(), "served": 0}
ENGINE = {}
GEN_LOCK = threading.Lock()
SNAPS = collections.OrderedDict()            # fed (tuple токенов) -> состояние на CPU
LIVE = {"fed": None}                         # какие токены сейчас в состоянии движка


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_ollama():
    """модели ollama делят ту же карту 8 ГБ — выгружаем перед загрузкой (как nem_server/ds41_server)"""
    try:
        ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3).read())
        for m in ps.get("models", []):
            rq = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                        data=json.dumps({"model": m["name"], "keep_alive": 0}).encode(),
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(rq, timeout=30).read()
            log("ollama выгружена:", m["name"])
    except Exception as e:
        log("ollama: пропуск выгрузки:", e)


def load_engine():
    try:
        free_ollama()
        sys.path.insert(0, HERE)
        import torch
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.set_num_threads(8)
        import qwen_engine as QE
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QE.SRC)
        eng = QE.QwenEngine("cuda", max_ctx=MAX_SEQ, ram_gb=RAM_GB, pf=PF, spec=SPEC)
        if os.environ.get("QW_DELTA_GRAPH", "0") == "1":
            from qwen_delta_graph import DeltaGraphs
            eng.delta_graphs = DeltaGraphs(eng)
            eng.delta_graphs.enabled = True
            log("DeltaNet CUDA graphs enabled (shared scratch pool)")
        ENGINE.update(eng=eng, tok=tok, torch=torch, im_start=tok.convert_tokens_to_ids("<|im_start|>"))
        STATE["status"] = "ready"
        log(f"готово: VRAM {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ, RAM-кэш {RAM_GB} ГБ, P={PF}, max_seq {MAX_SEQ}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE.update(status="error", error=repr(e))


def conv_msg(m):
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    role = m.get("role", "user")
    if role not in ("system", "user", "assistant"):
        role = "user"
    c = c or ""
    if role == "assistant" and "</think>" in c:                       # прошлое рассуждение в контекст не кладём
        c = c.split("</think>", 1)[1].lstrip("\n")
    if role == "assistant" and "<details" in c and "</details>" in c:  # так Open WebUI может вернуть свёрнутое рассуждение
        import re
        c = re.sub(r"<details[^>]*>.*?</details>\s*", "", c, flags=re.S)
    return {"role": role, "content": c}


def snap_take(eng):
    st = {"st": [{k: (v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v) for k, v in d.items()}
                 for d in eng.st],
          "ple_conv": None if eng.ple_conv is None else eng.ple_conv.detach().cpu(),
          "tokens": list(eng.tokens), "pos": eng.pos}
    fed = tuple(eng.tokens)
    SNAPS[fed] = st; SNAPS.move_to_end(fed)
    while len(SNAPS) > SNAP_N:
        SNAPS.popitem(last=False)


def snap_restore(eng, st):
    eng.st = [{k: (v.to(eng.dev) if hasattr(v, "to") else v) for k, v in d.items()} for d in st["st"]]
    eng.ple_conv = None if st["ple_conv"] is None else st["ple_conv"].to(eng.dev)
    eng.tokens = list(st["tokens"]); eng.pos = st["pos"]


def pick(lg, samp, gen):
    """samp None — жадно; иначе (T, top_k, top_p) как generation_config авторов"""
    torch = ENGINE["torch"]
    if samp is None:
        return int(lg.argmax())
    T, k, p = samp
    v, i = torch.topk(lg.float() / max(T, 1e-5), k)
    pr = torch.softmax(v, -1)
    cum = pr.cumsum(-1)
    pr = torch.where(cum - pr > p, torch.zeros_like(pr), pr)          # ядро top_p (первое слово остаётся всегда)
    return int(i[torch.multinomial(pr / pr.sum(), 1, generator=gen)])


def generate(messages, max_tokens, on_text, cancelled, think=False, effort=None, samp=None):
    torch, eng, tok = ENGINE["torch"], ENGINE["eng"], ENGINE["tok"]
    kw = {"reasoning_effort": effort or THINK_EFFORT} if think else {}
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think, tokenize=False, **kw)
    gen = torch.Generator(device="cuda"); gen.manual_seed(int(time.time() * 1000) % 2**31)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) + max_tokens > MAX_SEQ:
        max_tokens = MAX_SEQ - len(ids)
        if max_tokens < 16:
            on_text(f"[диалог слишком длинный: {len(ids)} токенов при пределе {MAX_SEQ} — начните новый чат]")
            return 0, len(ids), 0.0, 0.0
    stops = {248046, 248044}
    g, shown = [], ""
    with torch.inference_mode():
        torch.cuda.synchronize(); t0 = time.time()
        start, src = 0, "с нуля"
        if not STATE.get("nocache"):
            best = None
            for fed in [LIVE["fed"]] + list(reversed(list(SNAPS))):
                if fed and len(fed) < len(ids) and tuple(ids[:len(fed)]) == tuple(fed) and \
                        (best is None or len(fed) > len(best)):
                    best = fed
            if best is not None:
                if best != LIVE["fed"]:
                    snap_restore(eng, SNAPS[best])
                start, src = len(best), f"продолжение {len(best)} ток + хвост {len(ids) - len(best)}"
        if start == 0:
            eng.reset()
        LIVE["fed"] = None
        # снимок на начале последнего «<|im_start|>» (заголовок ответа): этот префикс совпадает с тем, как СЛЕДУЮЩИЙ
        # запрос отрисует историю в любом режиме (думающий вырезает рассуждение — конечный снимок тогда не подходит)
        im = ENGINE["im_start"]
        cut = max((i for i, t in enumerate(ids) if t == im), default=0)
        try:
            lg = None
            spans = ([(start, cut), (cut, len(ids))] if start < cut else [(start, len(ids))])
            for a0, b0 in spans:
                a = a0
                while a < b0:
                    c = prefill_chunk(eng.pos)
                    lg = eng.forward(ids[a:min(a + c, b0)]); a += c
                if b0 == cut and cut > 0:
                    snap_take(eng)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); eng.reset()
            log(f"промт {len(ids)} ток: нехватка видеопамяти ({src})")
            on_text("[Qwen: не хватило видеопамяти на чтение промта — сократите сообщение или начните новый чат]")
            return 0, len(ids), 0.0, 0.0
        torch.cuda.synchronize(); t_pre = time.time() - t0
        log(f"промт {len(ids)} ток: {src}, {t_pre:.1f} с" + (f", думает ({kw['reasoning_effort']})" if think else ""))
        if PIN:
            os.sched_setaffinity(0, {int(PIN)})  # single-thread decode; prefill keeps full affinity
        t1 = time.time()
        if think:
            on_text("<think>\n")                          # шаблон открыл <think> внутри промта
        while True:
            t = pick(lg, samp, gen)
            if t in stops:
                break
            g.append(t)
            s = tok.decode(g, skip_special_tokens=True)
            if not s.endswith("�") and len(s) > len(shown):          # не резать многобайтный символ пополам
                on_text(s[len(shown):]); shown = s
            if len(g) >= max_tokens or cancelled():
                break
            lg = eng.forward([t])
        s = tok.decode(g, skip_special_tokens=True)
        if len(s) > len(shown):
            on_text(s[len(shown):])
        # в состоянии — ids + g без последнего (последний сгенерированный ещё не подан); для продолжения диалога
        # шаблон добавит <|im_end|> после ответа — его дочитает хвост
        LIVE["fed"] = tuple(eng.tokens)
        try:
            snap_take(eng)
        except Exception as e:
            log("снимок не сохранён:", e)
    return len(g), len(ids), t_pre, time.time() - t1


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            pass

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self._json(200 if STATE["status"] == "ready" else 503,
                              {**STATE, "uptime_s": round(time.time() - STATE["since"]), "busy": GEN_LOCK.locked()})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "gpu104", "created": 0} for m in (MODEL_ID, THINK_ID)]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        waited = 0
        while STATE["status"] == "loading" and waited < 600:
            time.sleep(1); waited += 1
        if STATE["status"] != "ready":
            return self._json(503, {"error": {"message": f"модель не готова: {STATE['status']} {STATE['error'] or ''}"}})
        msgs = [conv_msg(m) for m in req.get("messages", [])]
        STATE["nocache"] = bool(req.get("qw_nocache"))
        model_id = req.get("model") or MODEL_ID
        think = model_id.endswith("-think")
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or
                         (THINK_MAX_TOKENS if think else DEFAULT_MAX_TOKENS))
        effort = req.get("reasoning_effort")
        if effort not in (None, "low", "medium", "xhigh"):
            effort = {"minimal": "low", "high": "xhigh"}.get(effort, THINK_EFFORT)
        temp = req.get("temperature")
        if think and temp is None:
            samp = (1.0, 20, 0.95)                         # generation_config авторов модели
        elif temp is not None and float(temp) > 0:
            samp = (float(temp), int(req.get("top_k") or 20), float(req.get("top_p") or 0.95))
        else:
            samp = None                                    # жадно (как чаты Nemotron/DeepSeek)
        cid, created, state = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time()), {"broken": False}
        with GEN_LOCK:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n").encode()
                    try:
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        state["broken"] = True

                def emit(t):
                    send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                          "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})

                n, plen, t_pre, dt = generate(msgs, max_tokens, emit, lambda: state["broken"], think, effort, samp)
                send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if n >= max_tokens else "stop"}],
                      "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
                send("[DONE]")
                try:
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                except Exception:
                    pass
            else:
                parts = []
                n, plen, t_pre, dt = generate(msgs, max_tokens, parts.append, lambda: False, think, effort, samp)
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                              "finish_reason": "length" if n >= max_tokens else "stop"}],
                                 "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
        STATE["served"] += 1
        log(f"промт {plen} ток за {t_pre:.1f} с, ответ {n} ток за {dt:.1f} с = {max(n - 1, 0) / max(dt, 1e-6):.2f} ток/с"
            + (" (прерван)" if state["broken"] else ""))


if __name__ == "__main__":
    os.chdir(HERE)
    threading.Thread(target=load_engine, daemon=True).start()
    log(f"qwen-stream на :{PORT}, грузится модель…")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
