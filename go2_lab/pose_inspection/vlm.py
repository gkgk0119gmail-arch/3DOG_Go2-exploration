"""Stage 5: local VLM (Qwen-VL family through transformers) served asynchronously from a worker thread.

``VlmClient.submit(images, text) -> Request``; the planner polls ``request.done`` and keeps acting on its previous
verdict meanwhile. The wall-clock latency of every call is recorded; ``Request.ready(t_sim)`` additionally waits
until the simulation clock has advanced by that latency since the request (so a fast simulation does not get answers
"for free"). Failures (exceptions, unparsable output) surface as ``request.error``.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Request:
    images: list
    text: str
    t_sim: float
    max_new_tokens: int
    done: bool = False
    reply: str = ""
    parsed: dict | None = None
    error: str | None = None
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict = field(default_factory=dict)

    def ready(self, t_sim: float) -> bool:
        return self.done and t_sim >= self.t_sim + self.latency_s


def parse_json_reply(text: str) -> dict | None:
    """First JSON object in the reply (models sometimes wrap it in prose or code fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s.replace("'", '"')):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


class VlmClient:
    def __init__(self, cfg, device: str = "cuda:0"):
        self.cfg, self.dev = cfg, device
        self.q: queue.Queue = queue.Queue()
        self.log: list[dict] = []
        self._model = None
        self._proc = None
        self._lock = threading.Lock()
        self._load()  # load up front: model loading must not count as reply latency
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._proc = AutoProcessor.from_pretrained(self.cfg.model)
        self._model = AutoModelForImageTextToText.from_pretrained(self.cfg.model, dtype=torch.bfloat16).to(
            self.dev).eval()

    def submit(self, images, text: str, t_sim: float, meta: dict | None = None) -> Request:
        r = Request(images=images, text=text, t_sim=t_sim, max_new_tokens=self.cfg.max_new_tokens, meta=meta or {})
        self.q.put(r)
        return r

    def _worker(self):
        import torch

        while True:
            r = self.q.get()
            t0 = time.time()
            try:
                if self._model is None:
                    self._load()
                content = [{"type": "image", "image": im} for im in r.images] + [{"type": "text", "text": r.text}]
                msgs = [{"role": "user", "content": content}]
                inp = self._proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self.dev)
                with torch.inference_mode():
                    out = self._model.generate(**inp, max_new_tokens=r.max_new_tokens, do_sample=False)
                gen = out[0, inp["input_ids"].shape[1]:]
                r.reply = self._proc.decode(gen, skip_special_tokens=True)
                r.tokens_in, r.tokens_out = int(inp["input_ids"].shape[1]), int(len(gen))
                r.parsed = parse_json_reply(r.reply)
                if r.parsed is None:
                    r.error = "unparsable"
            except Exception as e:  # noqa: BLE001
                r.error = f"{type(e).__name__}: {e}"
            r.latency_s = time.time() - t0
            r.done = True
            with self._lock:
                self.log.append({"t_sim": round(r.t_sim, 2), "latency_s": round(r.latency_s, 2), "tokens_in": r.tokens_in,
                                 "tokens_out": r.tokens_out, "error": r.error, "reply": r.reply[:600], **r.meta})

    def summary(self) -> dict:
        with self._lock:
            lat = [x["latency_s"] for x in self.log]
            return {"calls": len(self.log), "errors": sum(x["error"] is not None for x in self.log),
                    "latency_mean_s": sum(lat) / max(1, len(lat)), "latency_max_s": max(lat) if lat else 0.0,
                    "tokens_in": sum(x["tokens_in"] for x in self.log),
                    "tokens_out": sum(x["tokens_out"] for x in self.log)}
