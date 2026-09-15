from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class RecordingModelClient:
    """Transparent model proxy that durably records every request and response."""

    def __init__(self, inner: Any, log_path: Path):
        self.inner = inner
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.provider = getattr(inner, "provider", "unknown")
        self.model_id = getattr(inner, "model_id", "unknown")

    def generate_messages(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float, **kwargs: Any) -> Any:
        call_id = str(uuid.uuid4())
        started = time.time()
        request = {
            "event": "request", "call_id": call_id, "timestamp": started,
            "pid": os.getpid(), "provider": self.provider, "model": self.model_id,
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "kwargs": kwargs,
        }
        self._append(request)
        try:
            result = self.inner.generate_messages(messages, max_tokens, temperature, top_p, **kwargs)
            self._append({
                "event": "response", "call_id": call_id, "timestamp": time.time(),
                "elapsed_seconds": round(time.time() - started, 6),
                "text": result.text, "provider": getattr(result, "provider", self.provider),
                "model": getattr(result, "model", self.model_id),
                "latency_seconds": getattr(result, "latency_seconds", None),
                "raw": getattr(result, "raw", None),
            })
            return result
        except Exception as exc:
            self._append({
                "event": "error", "call_id": call_id, "timestamp": time.time(),
                "elapsed_seconds": round(time.time() - started, 6),
                "error_type": type(exc).__name__, "error": str(exc),
            })
            raise

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None, top_p: float = 1.0, **kwargs: Any) -> str:
        return self.generate_messages(
            [{"role": "user", "content": prompt}], max_tokens or 1024,
            temperature, top_p, **kwargs,
        ).text

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
