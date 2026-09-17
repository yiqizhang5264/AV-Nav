#!/usr/bin/env python3
"""Runtime-only VLM routing for the pinned STRIVE checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import time
from typing import Annotated, Any, get_args, get_origin


@dataclass(frozen=True)
class VLMRuntime:
    backend: str
    model: str
    base_url: str
    api_key: str
    disable_thinking: bool = False
    max_completion_tokens: int | None = None
    max_reasoning_steps: int | None = None
    max_string_chars: int | None = None
    max_list_items: int | None = None

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
        raw_max_tokens = os.environ.get("STRIVE_VLM_MAX_COMPLETION_TOKENS", "").strip()
        if raw_max_tokens:
            max_completion_tokens = int(raw_max_tokens)
            if max_completion_tokens <= 0:
                raise ValueError("STRIVE_VLM_MAX_COMPLETION_TOKENS must be positive")
        elif backend == "openai_compatible":
            max_completion_tokens = 1024
        else:
            max_completion_tokens = None
        raw_max_steps = os.environ.get("STRIVE_VLM_MAX_REASONING_STEPS", "").strip()
        if raw_max_steps:
            max_reasoning_steps = int(raw_max_steps)
            if max_reasoning_steps <= 0:
                raise ValueError("STRIVE_VLM_MAX_REASONING_STEPS must be positive")
        elif backend == "openai_compatible":
            max_reasoning_steps = 3
        else:
            max_reasoning_steps = None
        max_string_chars = 256 if backend == "openai_compatible" else None
        max_list_items = 32 if backend == "openai_compatible" else None
        return cls(
            backend,
            model,
            base_url,
            api_key,
            disable_thinking,
            max_completion_tokens,
            max_reasoning_steps,
            max_string_chars,
            max_list_items,
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "disable_thinking": self.disable_thinking,
            "max_completion_tokens": self.max_completion_tokens,
            "max_reasoning_steps": self.max_reasoning_steps,
            "max_string_chars": self.max_string_chars,
            "max_list_items": self.max_list_items,
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


def _concise_retry_messages(messages: list[Any]) -> list[Any]:
    instruction = (
        "Return a concise valid response matching the requested schema. "
        "Return only the final answer fields, with no reasoning or repeated text."
    )
    retried = [dict(message) for message in messages]
    if retried and retried[0].get("role") == "system":
        original = retried[0].get("content", "")
        retried[0]["content"] = f"{original}\n\n{instruction}"
        return retried
    return [{"role": "system", "content": instruction}, *retried]


def _final_only_response_format(response_format: Any) -> Any:
    model_fields = getattr(response_format, "model_fields", None)
    if not model_fields or "steps" not in model_fields:
        return response_format
    from pydantic import create_model

    fields = {
        name: (info.annotation, ...)
        for name, info in model_fields.items()
        if name != "steps"
    }
    return create_model(f"{response_format.__name__}FinalOnly", **fields)


def _bounded_response_format(
    response_format: Any,
    max_steps: int | None,
    max_string_chars: int | None,
    max_list_items: int | None,
) -> Any:
    if max_steps is None and max_string_chars is None and max_list_items is None:
        return response_format
    model_fields = getattr(response_format, "model_fields", None)
    if not model_fields:
        return response_format
    from pydantic import Field, create_model

    def bound_annotation(annotation: Any, field_name: str) -> Any:
        if annotation is str and max_string_chars is not None:
            return Annotated[str, Field(max_length=max_string_chars)]
        origin = get_origin(annotation)
        if origin is list:
            item_annotation = bound_annotation(get_args(annotation)[0], "")
            bounded_list = list[item_annotation]
            limit = max_steps if field_name == "steps" else max_list_items
            if limit is not None:
                return Annotated[bounded_list, Field(max_length=limit)]
            return bounded_list
        if getattr(annotation, "model_fields", None):
            return build_model(annotation)
        return annotation

    def build_model(model: Any) -> Any:
        fields = {
            name: (bound_annotation(info.annotation, name), ...)
            for name, info in model.model_fields.items()
        }
        return create_model(
            f"{model.__name__}Bounded",
            __base__=model,
            **fields,
        )

    return build_model(response_format)


class _CompletionsProxy:
    def __init__(self, wrapped: Any, runtime: VLMRuntime, log_path: str):
        self._wrapped = wrapped
        self._runtime = runtime
        self._log_path = log_path

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["model"] = self._runtime.model
        if "response_format" in kwargs:
            kwargs["response_format"] = _bounded_response_format(
                kwargs["response_format"],
                self._runtime.max_reasoning_steps,
                self._runtime.max_string_chars,
                self._runtime.max_list_items,
            )
        if self._runtime.max_completion_tokens is not None:
            kwargs.setdefault(
                "max_completion_tokens", self._runtime.max_completion_tokens
            )
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
            if type(error).__name__ != "LengthFinishReasonError":
                event.update({
                    "ok": False,
                    "latency_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                })
                _append_event(self._log_path, event)
                raise

            retry_kwargs = dict(kwargs)
            retry_kwargs["messages"] = _concise_retry_messages(
                list(kwargs.get("messages", []))
            )
            retry_kwargs["response_format"] = _final_only_response_format(
                kwargs.get("response_format")
            )
            retry_kwargs["temperature"] = 0.0
            if self._runtime.max_completion_tokens is not None:
                retry_kwargs["max_completion_tokens"] = min(
                    self._runtime.max_completion_tokens,
                    512,
                )
            event.update({
                "ok": False,
                "latency_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "retry": True,
            })
            _append_event(self._log_path, event)
            started = time.perf_counter()
            try:
                result = self._wrapped.parse(*args, **retry_kwargs)
            except Exception as retry_error:
                retry_event = {
                    "backend": self._runtime.backend,
                    "model": self._runtime.model,
                    **_request_summary(retry_kwargs),
                    "ok": False,
                    "latency_seconds": time.perf_counter() - started,
                    "error_type": type(retry_error).__name__,
                    "retry_attempt": 1,
                }
                _append_event(self._log_path, retry_event)
                raise
            event = {
                "backend": self._runtime.backend,
                "model": self._runtime.model,
                **_request_summary(retry_kwargs),
                "retry_attempt": 1,
            }

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
