import json
import unittest

from goethe.bench import score, tool_ok


class Score(unittest.TestCase):
    def test_a_broken_tool_call_is_invalid_and_warmup_is_dropped(self):
        self.assertEqual(tool_ok({"tool_calls": [
            {"function": {"arguments": json.dumps({"command": "ls"})}},
            {"function": {"arguments": "not json"}},
        ]}), (1, 1))
        rows = [
            {"completion_tokens": 1, "wall_s": 1, "prompt_tokens": 10, "tool_valid": 0,
             "tool_invalid": 9, "decode_tok_s": 1},
            {"completion_tokens": 1, "wall_s": 1, "prompt_tokens": 10, "tool_valid": 0,
             "tool_invalid": 9, "decode_tok_s": 1},
            {"completion_tokens": 100, "wall_s": 2, "prompt_tokens": 1000, "tool_valid": 2, "tool_invalid": 0,
             "decode_tok_s": 250, "acc_n": 7, "acc_d": 10},
        ]
        out = score(rows, [400.0])
        self.assertEqual(out["scored"], 1)
        self.assertEqual(out["tool_calls_invalid"], 0)
        self.assertEqual(out["decode_tok_s"], 250)
        self.assertEqual(out["tok_s_per_watt"], 0.625)
