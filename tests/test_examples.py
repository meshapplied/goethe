"""Serve examples parse, and the Flash-Next NVFP4 weight-only example carries the measured settings."""
import tomllib
import unittest
from pathlib import Path

from tests.fixtures import REPO

EXAMPLES = REPO / "examples"


def _args(path: Path) -> list[str]:
    with open(path, "rb") as f:
        return tomllib.load(f)["serve"]["extra_args"]


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


class Examples(unittest.TestCase):
    def test_every_example_parses(self):
        for path in sorted(EXAMPLES.glob("*.toml")):
            with self.subTest(example=path.name):
                with open(path, "rb") as f:
                    raw = tomllib.load(f)
                self.assertIn("model", raw)
                self.assertIn("serve", raw)
                self.assertEqual(raw["serve"]["engine"], "sglang")

    def test_weight_only_example_matches_the_measured_c1_serve(self):
        path = EXAMPLES / "nvfp4-weight-only.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        args = _args(path)
        self.assertEqual(raw["model"]["name"], "qwen38-flash-next-nvfp4-wo")
        self.assertEqual(_value(args, "--quantization"), "compressed-tensors")
        self.assertEqual(_value(args, "--speculative-algorithm"), "NEXTN")
        self.assertEqual(_value(args, "--speculative-num-steps"), "3")
        self.assertEqual(_value(args, "--speculative-num-draft-tokens"), "4")
        self.assertIn("spec/hot_self_64k.pt", _value(args, "--speculative-token-map"))
        self.assertEqual(_value(args, "--context-length"), "65536")
        self.assertEqual(_value(args, "--max-running-requests"), "1")
        self.assertEqual(_value(args, "--served-model-name"), raw["model"]["name"])
        self.assertIn("--ple-offload-embedding", args)


if __name__ == "__main__":
    unittest.main()
