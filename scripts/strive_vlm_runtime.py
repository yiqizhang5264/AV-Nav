#!/usr/bin/env python3
"""Runtime-only VLM routing for the pinned STRIVE checkout."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class VLMRuntime:
    backend: str
    model: str
    base_url: str
    api_key: str

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
        return cls(backend, model, base_url, api_key)

    def public_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
        }


class _CompletionsProxy:
    def __init__(self, wrapped: Any, model: str):
        self._wrapped = wrapped
        self._model = model

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["model"] = self._model
        return self._wrapped.parse(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _ChatProxy:
    def __init__(self, wrapped: Any, model: str):
        self._wrapped = wrapped
        self.completions = _CompletionsProxy(wrapped.completions, model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _BetaProxy:
    def __init__(self, wrapped: Any, model: str):
        self._wrapped = wrapped
        self.chat = _ChatProxy(wrapped.chat, model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def make_client_class(original_client: type, runtime: VLMRuntime) -> type:
    """Create an OpenAI client proxy that enforces one endpoint and model."""

    class RuntimeClient:
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["api_key"] = runtime.api_key
            kwargs["base_url"] = runtime.base_url
            self._client = original_client(*args, **kwargs)
            self.beta = _BetaProxy(self._client.beta, runtime.model)

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
