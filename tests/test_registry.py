import tempfile
import unittest
from pathlib import Path

from goethe import registry
from tests.fixtures import REPO

GOOD = """
[models.m]
title = "M"
engine = "ninfer"
served_name = "m-1"
native_context = 1000
recipe = "nvfp4-kernels"
min_vram_gib = 8
tested = []
[models.m.base]
kv_dtype = "fp8"
spec = "mtp"
draft_tokens = 3
prefill_chunk = 1024
[models.m.intents.agent]
lanes = 1
context = 4096
why = "w"
[models.m.intents.chat]
lanes = 1
context = 4096
why = "w"
[models.m.intents.long]
lanes = 1
context = 4096
why = "w"
"""


def load_text(text):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "models.toml"
        path.write_text(text)
        return registry.load(path)


class Load(unittest.TestCase):
    def test_registry_in_the_repo_loads(self):
        models = registry.load()
        m = models["qwen3-8-27b"]
        self.assertEqual((m.engine, m.served_name, m.native_context), ("ninfer", "qwen3.8-27b", 262144))
        self.assertEqual(set(m.intents), {"agent", "chat", "long"})

    def test_minimal_valid_file(self):
        self.assertEqual(load_text(GOOD)["m"].served_name, "m-1")

    def test_missing_and_unknown_keys_are_errors(self):
        with self.assertRaisesRegex(registry.RegistryError, "missing.*title"):
            load_text(GOOD.replace('title = "M"\n', ""))
        with self.assertRaisesRegex(registry.RegistryError, "unknown.*extra"):
            load_text(GOOD.replace('tested = []', 'tested = []\nextra = 1'))
        with self.assertRaisesRegex(registry.RegistryError, "missing.*long"):
            load_text(GOOD.split("[models.m.intents.long]")[0])

    def test_unknown_engine(self):
        with self.assertRaisesRegex(registry.RegistryError, "engine"):
            load_text(GOOD.replace('engine = "ninfer"', 'engine = "vllm"'))

    def test_find_by_id_or_served_name(self):
        models = registry.load()
        self.assertIs(registry.find(models, "qwen3-8-27b"), registry.find(models, "qwen3.8-27b"))
        with self.assertRaisesRegex(registry.RegistryError, "unknown model"):
            registry.find(models, "nope")


class Pin(unittest.TestCase):
    def test_recipe_pin_has_everything_goethe_reads(self):
        m = registry.load()["qwen3-8-27b"]
        pin = registry.read_pin(REPO / m.recipe / "NINFER_PIN")
        for key in ("MODEL_REPO", "MODEL_FILE", "MODEL_BYTES", "MODEL_SHA256", "NINFER_REV"):
            self.assertIn(key, pin)
        self.assertTrue(pin["MODEL_BYTES"].isdigit())
        self.assertEqual(len(pin["MODEL_SHA256"]), 64)


if __name__ == "__main__":
    unittest.main()
