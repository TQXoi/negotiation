"""Dependency-free OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ChatCompletionsHTTPClient:
    def __init__(self, *, base_url: str | None, api_key: str | None, timeout: float = 1800):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from chat endpoint: {body[:2000]}") from exc
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"Chat endpoint returned no choices: {result}")
        return str(choices[0].get("message", {}).get("content") or "")
