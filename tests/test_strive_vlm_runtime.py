import os
import sys
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

    def test_client_overrides_endpoint_key_and_every_model_argument(self):
        runtime = VLMRuntime("openai_compatible", "local-model", "http://vlm/v1", "key")
        client = make_client_class(_FakeClient, runtime)(
            api_key="upstream-key", base_url="https://upstream.invalid"
        )
        result = client.beta.chat.completions.parse(model="hard-coded-model", messages=[])
        self.assertEqual(_FakeClient.last_init["api_key"], "key")
        self.assertEqual(_FakeClient.last_init["base_url"], "http://vlm/v1")
        self.assertEqual(result["model"], "local-model")

    def test_public_config_never_contains_key_value(self):
        runtime = VLMRuntime("gemini", "model", "https://example.test", "secret")
        self.assertEqual(runtime.public_dict()["api_key_configured"], True)
        self.assertNotIn("secret", repr(runtime.public_dict()))


if __name__ == "__main__":
    unittest.main()
