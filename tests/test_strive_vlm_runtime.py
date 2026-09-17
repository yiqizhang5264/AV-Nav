import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from strive_vlm_runtime import VLMRuntime, make_client_class


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def parse(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return kwargs


class LengthFinishReasonError(Exception):
    pass


class _LengthThenSuccessCompletions(_FakeCompletions):
    def __init__(self):
        super().__init__()
        self.calls = []

    def parse(self, *args, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise LengthFinishReasonError("too long")
        return kwargs


class _FakeClient:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = kwargs
        self.beta = type("Beta", (), {})()
        self.beta.chat = type("Chat", (), {})()
        self.beta.chat.completions = _FakeCompletions()


class StriveVLMRuntimeTests(unittest.TestCase):
    def test_local_defaults_do_not_require_gemini_key(self):
        with patch.dict(os.environ, {"STRIVE_VLM_BACKEND": "openai_compatible"}, clear=True):
            runtime = VLMRuntime.from_env()
        self.assertEqual(runtime.model, "Qwen/Qwen3.5-9B")
        self.assertEqual(runtime.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(runtime.api_key, "local")
        self.assertEqual(runtime.max_completion_tokens, 1024)
        self.assertEqual(runtime.max_reasoning_steps, 3)
        self.assertEqual(runtime.max_string_chars, 256)
        self.assertEqual(runtime.max_list_items, 32)

    def test_client_overrides_endpoint_key_and_every_model_argument(self):
        runtime = VLMRuntime("openai_compatible", "local-model", "http://vlm/v1", "key")
        client = make_client_class(_FakeClient, runtime)(
            api_key="upstream-key", base_url="https://upstream.invalid"
        )
        result = client.beta.chat.completions.parse(model="hard-coded-model", messages=[])
        self.assertEqual(_FakeClient.last_init["api_key"], "key")
        self.assertEqual(_FakeClient.last_init["base_url"], "http://vlm/v1")
        self.assertEqual(result["model"], "local-model")

    def test_disable_thinking_is_injected_without_losing_extra_body(self):
        runtime = VLMRuntime(
            "openai_compatible", "local-model", "http://vlm/v1", "key", True
        )
        client = make_client_class(_FakeClient, runtime)()
        result = client.beta.chat.completions.parse(
            messages=[], extra_body={"top_k": 20}
        )
        self.assertEqual(result["extra_body"]["top_k"], 20)
        self.assertEqual(
            result["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            False,
        )

    def test_completion_limit_is_injected(self):
        runtime = VLMRuntime(
            "openai_compatible",
            "local-model",
            "http://vlm/v1",
            "key",
            max_completion_tokens=768,
        )
        client = make_client_class(_FakeClient, runtime)()
        result = client.beta.chat.completions.parse(messages=[])
        self.assertEqual(result["max_completion_tokens"], 768)

    def test_reasoning_steps_schema_is_bounded(self):
        class FieldInfo:
            def __init__(self, annotation):
                self.annotation = annotation

        class Step:
            model_fields = {
                "explanation": FieldInfo(str),
                "output": FieldInfo(str),
            }

        class Result:
            model_fields = {
                "steps": FieldInfo(list[Step]),
                "answer": FieldInfo(str),
            }

        captured = {}

        def field(**kwargs):
            return kwargs

        def create_model(name, __base__, **fields):
            captured[name] = {"base": __base__, "fields": fields}
            generated_fields = {
                field_name: FieldInfo(definition[0])
                for field_name, definition in fields.items()
            }
            return type(name, (__base__,), {"model_fields": generated_fields})

        runtime = VLMRuntime(
            "openai_compatible",
            "local-model",
            "http://vlm/v1",
            "key",
            max_reasoning_steps=3,
            max_string_chars=256,
            max_list_items=32,
        )
        client = make_client_class(_FakeClient, runtime)()
        fake_pydantic = types.SimpleNamespace(Field=field, create_model=create_model)
        with patch.dict(sys.modules, {"pydantic": fake_pydantic}):
            result = client.beta.chat.completions.parse(
                messages=[], response_format=Result
            )
        result_model = result["response_format"]
        outer = captured["ResultBounded"]
        steps_annotation = outer["fields"]["steps"][0]
        self.assertEqual(steps_annotation.__metadata__[0]["max_length"], 3)
        bounded_step = steps_annotation.__origin__.__args__[0]
        explanation = captured[bounded_step.__name__]["fields"]["explanation"][0]
        self.assertEqual(explanation.__metadata__[0]["max_length"], 256)
        answer = outer["fields"]["answer"][0]
        self.assertEqual(answer.__metadata__[0]["max_length"], 256)
        self.assertEqual(result_model.__name__, "ResultBounded")

    def test_non_reasoning_list_uses_general_bound(self):
        class FieldInfo:
            def __init__(self, annotation):
                self.annotation = annotation

        class Result:
            model_fields = {"res": FieldInfo(list[str])}

        def field(**kwargs):
            return kwargs

        def create_model(name, __base__, **fields):
            generated_fields = dict(getattr(__base__, "model_fields", {}))
            generated_fields.update({
                field_name: FieldInfo(definition[0])
                for field_name, definition in fields.items()
            })
            return type(name, (__base__,), {"model_fields": generated_fields})

        runtime = VLMRuntime(
            "openai_compatible",
            "local-model",
            "http://vlm/v1",
            "key",
            max_reasoning_steps=3,
            max_list_items=32,
        )
        client = make_client_class(_FakeClient, runtime)()
        fake_pydantic = types.SimpleNamespace(Field=field, create_model=create_model)
        with patch.dict(sys.modules, {"pydantic": fake_pydantic}):
            result = client.beta.chat.completions.parse(
                messages=[], response_format=Result
            )
        annotation = result["response_format"].model_fields["res"].annotation
        self.assertEqual(annotation.__metadata__[0]["max_length"], 32)

    def test_length_failure_retries_with_concise_instruction(self):
        class LengthThenSuccessClient(_FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.beta.chat.completions = _LengthThenSuccessCompletions()

        runtime = VLMRuntime(
            "openai_compatible",
            "local-model",
            "http://vlm/v1",
            "key",
            max_completion_tokens=512,
        )
        client = make_client_class(LengthThenSuccessClient, runtime)()
        result = client.beta.chat.completions.parse(
            messages=[{"role": "user", "content": "classify"}]
        )
        self.assertEqual(result["temperature"], 0.0)
        self.assertEqual(result["max_completion_tokens"], 1024)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("at most three", result["messages"][0]["content"])

    def test_length_retry_merges_existing_system_message(self):
        class LengthThenSuccessClient(_FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.beta.chat.completions = _LengthThenSuccessCompletions()

        runtime = VLMRuntime(
            "openai_compatible",
            "local-model",
            "http://vlm/v1",
            "key",
            max_completion_tokens=512,
        )
        client = make_client_class(LengthThenSuccessClient, runtime)()
        result = client.beta.chat.completions.parse(
            messages=[
                {"role": "system", "content": "choose a room"},
                {"role": "user", "content": "options"},
            ]
        )
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("choose a room", result["messages"][0]["content"])
        self.assertIn("at most three", result["messages"][0]["content"])

    def test_public_config_never_contains_key_value(self):
        runtime = VLMRuntime("gemini", "model", "https://example.test", "secret")
        self.assertEqual(runtime.public_dict()["api_key_configured"], True)
        self.assertNotIn("secret", repr(runtime.public_dict()))

    def test_call_telemetry_records_hash_not_prompt_or_key(self):
        runtime = VLMRuntime("gemini", "model", "https://example.test", "secret")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calls.jsonl")
            with patch.dict(os.environ, {"STRIVE_VLM_LOG": path}):
                client = make_client_class(_FakeClient, runtime)()
                client.beta.chat.completions.parse(
                    model="ignored", messages=[{"role": "user", "content": "private prompt"}]
                )
            with open(path, encoding="utf-8") as stream:
                content = stream.read()
        self.assertIn('"ok": true', content)
        self.assertIn('"request_sha256"', content)
        self.assertNotIn("private prompt", content)
        self.assertNotIn("secret", content)


if __name__ == "__main__":
    unittest.main()
