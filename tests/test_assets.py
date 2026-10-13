"""The patch assets in nvfp4-kernels/patches must agree with each other and with NINFER_PIN."""
import json
import unittest

from goethe import registry
from tests.fixtures import REPO

KERNELS = REPO / "nvfp4-kernels"
PATCHES = KERNELS / "patches"
ROWS, ROW_BYTES = 131072, 2560 + 160  # q4_g64 row: 2560 code bytes plus 160 scale bytes


class HeadPatch(unittest.TestCase):
    def test_swap_lists_match_the_row_file(self):
        swap = json.loads((PATCHES / "head_swap.json").read_text())
        drop, add = swap["drop"], swap["add"]
        self.assertEqual(len(drop), len(add))
        self.assertEqual(len(set(drop)), len(drop))
        self.assertEqual(len(set(add)), len(add))
        self.assertFalse(set(drop) & set(add))
        self.assertEqual((PATCHES / "head_rows.dat").stat().st_size, len(add) * ROW_BYTES)
        self.assertLess(max(drop + add), 248320)

    def test_pin_has_what_apply_head_reads(self):
        pin = registry.read_pin(KERNELS / "NINFER_PIN")
        for key in ("HEAD_OFFSET", "HEAD_BYTES", "IDS_OFFSET", "IDS_BYTES"):
            self.assertTrue(pin[key].isdigit(), key)
        self.assertEqual(int(pin["HEAD_BYTES"]), ROWS * 2560 + ROWS * 160)
        self.assertEqual(int(pin["IDS_BYTES"]), ROWS * 4)
        for key in ("HEAD_ORIG_SHA256", "IDS_ORIG_SHA256", "HEAD_PATCHED_SHA256", "IDS_PATCHED_SHA256",
                    "MODEL_SHA256", "MODEL_PATCHED_SHA256"):
            self.assertRegex(pin[key], r"^[0-9a-f]{64}$", key)

    def test_ninfer_patch_touches_only_the_two_schedule_files(self):
        files = [line.split(" b/")[1] for line in (PATCHES / "ninfer-dn-sched.patch").read_text().splitlines()
                 if line.startswith("diff --git")]
        self.assertEqual(files, ["src/ops/linear/nvfp4/shapes/n5120_k17408.cu",
                                 "src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu"])


if __name__ == "__main__":
    unittest.main()
