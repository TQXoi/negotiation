"""Unified text-generation clients for local and API negotiation experiments."""

from __future__ import annotations

import os
import re
import json
import hashlib
import math
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from agenticpay.models.base_llm import BaseLLM
except Exception:  # pragma: no cover - lets this module be imported before sys.path setup.
    class BaseLLM:  # type: ignore[no-redef]
        pass


Message = Dict[str, str]
_OPENAI_LEGACY_LOCK = threading.Lock()
_CONTEXT_AUDIT_LOCK = threading.Lock()
_RESPONSE_CACHE_LOCK = threading.Lock()


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    latency_seconds: float
    raw: Optional[Any] = None


class ModelClient(BaseLLM):
    """Minimal provider-neutral generation interface."""

    provider: str
    model_id: str

    def generate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        result = self.generate_messages(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens or 1024,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return result.text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model_id={self.model_id})"


class LocalHFClient(ModelClient):
    """Transformers causal LM client compatible with AgenticPay BaseLLM.generate."""

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        enable_thinking: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.provider = "hf"
        self.model_id = model_path
        self.enable_thinking = enable_thinking
        dtype = self._resolve_dtype(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    def generate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        start = time.time()
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not self.enable_thinking:
            generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL | re.IGNORECASE).strip()
        return GenerationResult(
            text=generated,
            provider=self.provider,
            model=self.model_id,
            latency_seconds=round(time.time() - start, 3),
        )


class OpenAIClient(ModelClient):
    def __init__(self, model_id: str, base_url: Optional[str] = None):
        self.provider = "openai"
        self.model_id = model_id
        self.sdk_style = "v1"
        self.use_chat_completions = False
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        timeout_raw = os.environ.get("OPENAI_REQUEST_TIMEOUT")
        self.request_timeout = float(timeout_raw) if timeout_raw else None
        # Opt-in context protection for local OpenAI-compatible servers.  The
        # benchmark runner sets these explicitly, so hosted APIs and unrelated
        # experiments retain their native behavior.
        self.max_context_tokens = int(os.environ.get("NEGOTIATION_MAX_CONTEXT_TOKENS", "0") or 0)
        self.max_output_tokens = int(os.environ.get("NEGOTIATION_MAX_OUTPUT_TOKENS", "0") or 0)
        self.context_chars_per_token = float(os.environ.get("NEGOTIATION_CONTEXT_CHARS_PER_TOKEN", "3.0"))
        self.context_safety_margin = int(os.environ.get("NEGOTIATION_CONTEXT_SAFETY_MARGIN", "256"))
        request_seed_raw = os.environ.get("NEGOTIATION_REQUEST_SEED_BASE")
        self.request_seed_base = (
            int(request_seed_raw) if request_seed_raw not in (None, "") else None
        )
        response_cache_raw = os.environ.get("NEGOTIATION_RESPONSE_CACHE_JSONL")
        self.response_cache_path = (
            Path(response_cache_raw) if response_cache_raw else None
        )
        self._response_cache: Dict[str, str] = {}
        if self.response_cache_path and self.response_cache_path.exists():
            for line in self.response_cache_path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("cache_key") and isinstance(row.get("text"), str):
                    self._response_cache[str(row["cache_key"])] = row["text"]
        audit_path = os.environ.get("NEGOTIATION_CONTEXT_AUDIT_JSONL")
        self.context_audit_path = Path(audit_path) if audit_path else None
        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
                self.use_chat_completions = True
            if self.request_timeout is not None:
                kwargs["timeout"] = self.request_timeout
            self.client = OpenAI(**kwargs)
        except ImportError:
            import openai

            self.sdk_style = "legacy"
            self.client = openai
            self.client.api_key = self.api_key
            if self.base_url:
                self.client.api_base = self.base_url

    def generate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> GenerationResult:
        start = time.time()
        messages, max_tokens = self._context_safe_request(messages, max_tokens)
        # Optional common-random-number control for paired evaluations.  The
        # same public prompt under two framework variants receives the same
        # vLLM sample, while a different evaluation seed produces a new seller
        # rollout.  Hosted/default clients are unchanged unless explicitly
        # enabled by the experiment runner.
        if self.request_seed_base is not None and "seed" not in kwargs:
            material = json.dumps(
                {
                    "base": self.request_seed_base,
                    "model": self.model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            kwargs["seed"] = int(
                hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16
            )
        response_cache_key = self._response_cache_key(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            kwargs=kwargs,
        )
        cached_text = self._response_cache.get(response_cache_key)
        if cached_text is not None:
            return GenerationResult(
                text=cached_text,
                provider=self.provider,
                model=self.model_id,
                latency_seconds=round(time.time() - start, 3),
                raw={"response_cache_hit": True, "cache_key": response_cache_key},
            )
        if self.sdk_style == "legacy":
            if self.request_timeout is not None and "request_timeout" not in kwargs:
                kwargs["request_timeout"] = self.request_timeout
            # openai<1.0 stores api_base/api_key as module globals. Protect
            # per-client endpoints when buyer and seller use different vLLM ports.
            with _OPENAI_LEGACY_LOCK:
                previous_api_base = getattr(self.client, "api_base", None)
                previous_api_key = getattr(self.client, "api_key", None)
                if self.base_url:
                    self.client.api_base = self.base_url
                if self.api_key:
                    self.client.api_key = self.api_key
                try:
                    response = self.client.ChatCompletion.create(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **kwargs,
                    )
                finally:
                    self.client.api_base = previous_api_base
                    self.client.api_key = previous_api_key
            text = response["choices"][0]["message"]["content"].strip()
            raw = response.to_dict_recursive() if hasattr(response, "to_dict_recursive") else None
            self._store_cached_response(response_cache_key, text)
            return GenerationResult(
                text=text,
                provider=self.provider,
                model=self.model_id,
                latency_seconds=round(time.time() - start, 3),
                raw=raw,
            )

        if self.use_chat_completions:
            if self.request_timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.request_timeout
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            self._store_cached_response(response_cache_key, text)
            return GenerationResult(
                text=text,
                provider=self.provider,
                model=self.model_id,
                latency_seconds=round(time.time() - start, 3),
                raw=response.model_dump(mode="json"),
            )

        response = self.client.responses.create(
            model=self.model_id,
            input=messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        text = response.output_text.strip()
        self._store_cached_response(response_cache_key, text)
        return GenerationResult(
            text=text,
            provider=self.provider,
            model=self.model_id,
            latency_seconds=round(time.time() - start, 3),
            raw=response.model_dump(mode="json"),
        )

    def _response_cache_key(
        self,
        *,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        kwargs: Dict[str, Any],
    ) -> str:
        material = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "kwargs": kwargs,
        }
        return hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _store_cached_response(self, cache_key: str, text: str) -> None:
        if self.response_cache_path is None or cache_key in self._response_cache:
            return
        row = {
            "cache_key": cache_key,
            "provider": self.provider,
            "model": self.model_id,
            "text": text,
        }
        with _RESPONSE_CACHE_LOCK:
            if cache_key in self._response_cache:
                return
            self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.response_cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._response_cache[cache_key] = text

    def _context_safe_request(
        self,
        messages: List[Message],
        max_tokens: int,
    ) -> tuple[List[Message], int]:
        """Reserve completion space and compact only overlong prompt history.

        AgenticPay builds one large user prompt containing static context,
        conversation history, and a repeated rules suffix.  We retain the role
        header, the full rules suffix, and the newest whole round records while
        dropping oldest history.  A generic head/tail fallback handles other
        single-message prompts.  The audit stores sizes/hashes, never private
        prompt text.
        """

        requested_output = int(max_tokens)
        effective_output = (
            min(requested_output, self.max_output_tokens)
            if self.max_output_tokens > 0
            else requested_output
        )
        if self.max_context_tokens <= 0:
            return messages, effective_output
        input_tokens = max(
            256,
            self.max_context_tokens - effective_output - self.context_safety_margin,
        )
        char_limit = max(1024, int(input_tokens * self.context_chars_per_token))
        before_chars = sum(len(str(message.get("content", ""))) for message in messages)
        if before_chars <= char_limit:
            return messages, effective_output

        safe = [dict(message) for message in messages]
        if len(safe) == 1:
            safe[0]["content"] = self._compact_single_prompt(
                str(safe[0].get("content", "")), char_limit
            )
        else:
            # Preserve system rules and newest complete role messages.  If the
            # system message itself is too large, compact it by head/tail.
            system = safe[0]
            system_content = str(system.get("content", ""))
            system_budget = min(len(system_content), max(512, char_limit // 2))
            system["content"] = self._head_tail(system_content, system_budget)
            remaining = max(0, char_limit - len(system["content"]))
            kept: List[Message] = []
            used = 0
            for message in reversed(safe[1:]):
                content = str(message.get("content", ""))
                if used + len(content) <= remaining:
                    kept.append(message)
                    used += len(content)
                    continue
                if not kept and remaining > 512:
                    partial = dict(message)
                    partial["content"] = self._head_tail(content, remaining)
                    kept.append(partial)
                break
            safe = [system, *reversed(kept)]

        after_chars = sum(len(str(message.get("content", ""))) for message in safe)
        self._audit_context_compaction(
            before_chars=before_chars,
            after_chars=after_chars,
            requested_output=requested_output,
            effective_output=effective_output,
            messages=messages,
        )
        return safe, effective_output

    @classmethod
    def _compact_single_prompt(cls, content: str, limit: int) -> str:
        history_marker = "Conversation History:\n"
        suffix_marker = "\n\nPlease respond naturally"
        history_start = content.find(history_marker)
        suffix_start = content.find(suffix_marker, history_start + len(history_marker))
        if history_start < 0 or suffix_start < 0:
            return cls._head_tail(content, limit)

        role_header = content[: content.find("\n", 0) + 1]
        suffix = content[suffix_start:]
        # BaseAgent context is repeated in the role-specific guidance suffix;
        # replacing it prevents static JSON from crowding out recent actions.
        compact_prefix = role_header + "\nContext Information:\n[See authoritative task guidance below.]\n\n" + history_marker
        minimum_history_budget = 600
        if len(compact_prefix) + len(suffix) + minimum_history_budget > limit:
            suffix_budget = max(1024, limit - len(compact_prefix) - minimum_history_budget)
            suffix = cls._head_tail(suffix, suffix_budget)
        available = max(0, limit - len(compact_prefix) - len(suffix) - 32)
        history = content[history_start + len(history_marker) : suffix_start]
        rounds = re.split(r"(?=\[Round\s+\d+\])", history)
        kept: List[str] = []
        used = 0
        for block in reversed([block for block in rounds if block.strip()]):
            if used + len(block) > available:
                break
            kept.append(block)
            used += len(block)
        if not kept and available > 0:
            kept = [cls._head_tail(history, available)]
        compact_history = "[Earlier rounds omitted for context safety.]\n" + "".join(reversed(kept))
        result = compact_prefix + compact_history + suffix
        return cls._head_tail(result, limit) if len(result) > limit else result

    @staticmethod
    def _head_tail(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        marker = "\n[... context-safe middle truncation ...]\n"
        usable = max(0, limit - len(marker))
        head = int(usable * 0.42)
        return content[:head] + marker + content[-(usable - head) :]

    def _audit_context_compaction(
        self,
        *,
        before_chars: int,
        after_chars: int,
        requested_output: int,
        effective_output: int,
        messages: List[Message],
    ) -> None:
        if self.context_audit_path is None:
            return
        digest_source = "\n".join(str(message.get("content", "")) for message in messages)
        record = {
            "timestamp": time.time(),
            "model": self.model_id,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "estimated_before_tokens": math.ceil(before_chars / max(self.context_chars_per_token, 1e-6)),
            "estimated_after_tokens": math.ceil(after_chars / max(self.context_chars_per_token, 1e-6)),
            "requested_output_tokens": requested_output,
            "effective_output_tokens": effective_output,
            "max_context_tokens": self.max_context_tokens,
            "prompt_sha256": hashlib.sha256(digest_source.encode("utf-8", errors="replace")).hexdigest(),
        }
        self.context_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with _CONTEXT_AUDIT_LOCK:
            with self.context_audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class AnthropicClient(ModelClient):
    def __init__(self, model_id: str):
        import anthropic

        self.provider = "anthropic"
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> GenerationResult:
        start = time.time()
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        response = self.client.messages.create(
            model=self.model_id,
            system="\n\n".join(system_parts) if system_parts else None,
            messages=non_system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        text = "".join(getattr(block, "text", "") for block in response.content).strip()
        return GenerationResult(
            text=text,
            provider=self.provider,
            model=self.model_id,
            latency_seconds=round(time.time() - start, 3),
            raw=response.model_dump(mode="json"),
        )


class GeminiClient(ModelClient):
    def __init__(self, model_id: str):
        from google import genai

        self.provider = "google"
        self.model_id = model_id
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def generate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> GenerationResult:
        from google.genai import types

        start = time.time()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system")
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            ),
        )
        return GenerationResult(
            text=(response.text or "").strip(),
            provider=self.provider,
            model=self.model_id,
            latency_seconds=round(time.time() - start, 3),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def make_model_client(
    spec: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    enable_thinking: bool = False,
) -> ModelClient:
    """Create a client from specs such as hf:/path, openai:gpt-5.2, or /path."""

    if spec.startswith("openai@"):
        endpoint_and_model = spec.split("@", 1)[1]
        try:
            base_url, model_id = endpoint_and_model.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                "OpenAI endpoint specs must look like "
                "openai@http://127.0.0.1:8000/v1:MODEL_NAME"
            ) from exc
        return OpenAIClient(model_id, base_url=base_url)
    if spec.startswith("openai:"):
        return OpenAIClient(spec.split(":", 1)[1])
    if spec.startswith("anthropic:"):
        return AnthropicClient(spec.split(":", 1)[1])
    if spec.startswith("google:"):
        return GeminiClient(spec.split(":", 1)[1])
    if spec.startswith("hf:"):
        spec = spec.split(":", 1)[1]
    return LocalHFClient(
        model_path=spec,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
        enable_thinking=enable_thinking,
    )
