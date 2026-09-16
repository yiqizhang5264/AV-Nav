#!/usr/bin/env python3
"""Runtime-only VLM routing for the pinned STRIVE checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import time
from typing import Any


@dataclass(frozen=True)
class VLMRuntime:
    backend: str
    model: str
    base_url: str
    api_key: str
    disable_thinking: bool = False

    @classmethod
    def from_env(cls) -> "VLMRuntime":
        backend = os.environ.get("STRIVE_VLM_BACKEND", "gemini").strip().lower()
        if backend == "gemini":
            model = (
                os.environ.get("STRIVE_VLM_MODEL")
                or os.environ.get("STRIVE_GEMINI_MODEL")
                or "gemini-2.5-flash"
            ).strip()
            base_url = os.environ.get(
                "STRIVE_VLM_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ).strip()
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        elif backend == "openai_compatible":
            model = os.environ.get(
                "STRIVE_VLM_MODEL", "Qwen/Qwen3.5-9B"
            ).strip()
            base_url = os.environ.get(
                "STRIVE_VLM_BASE_URL", "http://127.0.0.1:8000/v1"
            ).strip()
            api_key = os.environ.get("STRIVE_VLM_API_KEY", "local").strip()
        else:
            raise ValueError(
                "STRIVE_VLM_BACKEND must be 'gemini' or 'openai_compatible'"
            )

        if not model or not base_url or not api_key:
            raise ValueError(f"Incomplete VLM configuration for backend {backend!r}")
        disable_thinking = os.environ.get(
            "STRIVE_VLM_DISABLE_THINKING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        return cls(backend, model, base_url, api_key, disable_thinking)

    def public_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "disable_thinking": self.disable_thinking,
        }


def _request_summary(kwargs: dict[str, Any]) -> dict[str, object]:
    messages = kwargs.get("messages", [])
    rendered = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "request_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "message_chars": len(rendered),
        "has_image": "image_url" in rendered,
    }


def _append_event(path: str, event: dict[str, object]) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


class _CompletionsProxy:
    def __init__(self, wrapped: Any, runtime: VLMRuntime, log_path: str):
        self._wrapped = wrapped
        self._runtime = runtime
        self._log_path = log_path

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["model"] = self._runtime.model
        if self._runtime.disable_thinking:
            extra_body = dict(kwargs.get("extra_body") or {})
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = template_kwargs
            kwargs["extra_body"] = extra_body
        event = {
            "backend": self._runtime.backend,
            "model": self._runtime.model,
            **_request_summary(kwargs),
        }
        started = time.perf_counter()
        try:
            result = self._wrapped.parse(*args, **kwargs)
        except Exception as error:
            event.update({
                "ok": False,
                "latency_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
            })
            _append_event(self._log_path, event)
            raise

        usage = getattr(result, "usage", None)
        event.update({
            "ok": True,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })
        _append_event(self._log_path, event)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _ChatProxy:
    def __init__(self, wrapped: Any, runtime: VLMRuntime, log_path: str):
        self._wrapped = wrapped
        self.completions = _CompletionsProxy(wrapped.completions, runtime, log_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _BetaProxy:
    def __init__(self, wrapped: Any, runtime: VLMRuntime, log_path: str):
        self._wrapped = wrapped
        self.chat = _ChatProxy(wrapped.chat, runtime, log_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def make_client_class(original_client: type, runtime: VLMRuntime) -> type:
    """Create an OpenAI client proxy that enforces one endpoint and model."""

    class RuntimeClient:
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["api_key"] = runtime.api_key
            kwargs["base_url"] = runtime.base_url
            self._client = original_client(*args, **kwargs)
            self.beta = _BetaProxy(
                self._client.beta,
                runtime,
                os.environ.get("STRIVE_VLM_LOG", ""),
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._client, name)

        def __enter__(self) -> "RuntimeClient":
            self._client.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._client.__exit__(*args)

    return RuntimeClient


def install_openai_runtime(runtime: VLMRuntime) -> None:
    import openai

    openai.OpenAI = make_client_class(openai.OpenAI, runtime)
