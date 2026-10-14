import json
import unittest

from goethe.bench import parse_done, score, tool_ok


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
             "decode_tok_s": 250, "acc_n": 7, "acc_d": 10, "cache_tokens": 800, "prompt_logged": 1000, "ttft_s": 1.0},
            {"completion_tokens": 100, "wall_s": 2, "prompt_tokens": 1000, "tool_valid": 0, "tool_invalid": 0,
             "decode_tok_s": 100, "acc_n": 1, "acc_d": 10, "cache_tokens": 0, "prompt_logged": 1000, "ttft_s": 3.0},
        ]
        out = score(rows, [400.0])
        self.assertEqual(out["scored"], 2)
        self.assertEqual(out["tool_calls_invalid"], 0)
        self.assertEqual(out["decode_tok_s"], 142.9)
        self.assertEqual(out["cache_hit"], "40.0%")
        self.assertEqual(out["ttft_s"], 2.0)
        self.assertEqual(out["tok_s_per_watt"], 0.357)

    def test_a_done_line_keeps_counts_that_contain_commas(self):
        line = ("req#1 done | openai-chat | output limit | prompt 4,863 | output 1,500 | "
                "cache 1,024 (21.1%) | TTFT 1.5s | total 7.4s | prefill 7.17k tok/s | "
                "decode 252.3 tok/s | dflash2 accepted 1,128/2,587 (43.6%)")
        self.assertEqual(parse_done(line), [{
            "prompt_logged": 4863, "cache_tokens": 1024, "ttft_s": 1.5,
            "decode_tok_s": 252.3, "acc_n": 1128, "acc_d": 2587,
        }])
        real = ("2026-10-14 18:11:04.122  INFO  req#1 done | openai-chat | output limit | "
                "prompt 4,863 | output 1,500 | cache 0 (0.0%) | TTFT 1.5s | total 7.4s | "
                "prefill 7.17k tok/s | decode 252.3 tok/s | dflash2 accepted 1,128/2,587 (43.6%)")
        self.assertEqual(parse_done(real)[0]["acc_n"], 1128)
        self.assertEqual(parse_done(real)[0]["cache_tokens"], 0)
