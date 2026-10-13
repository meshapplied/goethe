import unittest

from goethe import hw

LINE = "0, NVIDIA GeForce RTX 5090, 32607, 32100, 580.178.04, 12.0, 575.00, 16, 16"


class Probe(unittest.TestCase):
    def test_parse(self):
        (g,) = hw.parse(LINE + "\n")
        self.assertEqual((g.index, g.name, g.total_mib, g.free_mib), (0, "NVIDIA GeForce RTX 5090", 32607, 32100))
        self.assertEqual((g.driver, g.compute_cap, g.power_limit_w, g.pcie_width), ("580.178.04", "12.0", 575.0, 16))
        self.assertAlmostEqual(g.total_gib, 31.84, places=2)

    def test_not_available_fields_become_none(self):
        (g,) = hw.parse("1, GPU X, 100, 50, 580.1, 12.0, [N/A], [N/A], [N/A]")
        self.assertEqual((g.index, g.power_limit_w, g.pcie_width, g.pcie_width_max), (1, None, None, None))

    def test_two_gpus(self):
        self.assertEqual([g.index for g in hw.parse(LINE + "\n" + LINE.replace("0,", "1,", 1))], [0, 1])

    def test_malformed_line_is_an_error(self):
        with self.assertRaises(ValueError):
            hw.parse("0, only, three")

    def test_probe_results(self):
        self.assertEqual(hw.probe(lambda cmd: (127, "")), ([], "nvidia-smi not found"))
        self.assertEqual(hw.probe(lambda cmd: (9, "")), ([], "nvidia-smi failed (exit 9)"))
        self.assertEqual(hw.probe(lambda cmd: (0, "")), ([], "nvidia-smi reports no GPU"))
        gpus, err = hw.probe(lambda cmd: (0, LINE))
        self.assertEqual((len(gpus), err), (1, None))
        self.assertEqual(hw.probe(lambda cmd: (0, "garbage"))[0], [])

    def test_probe_asks_for_the_fields_parse_expects(self):
        seen = []
        hw.probe(lambda cmd: (seen.append(cmd), (0, LINE))[1])
        self.assertIn(f"--query-gpu={hw.QUERY}", seen[0])
        self.assertEqual(hw.QUERY.count(",") + 1, 9)

    def test_compute_apps(self):
        out = "4242, /usr/bin/python3, 2048\n77, weird, name, [N/A]\n"
        self.assertEqual(hw.compute_apps(lambda cmd: (0, out)),
                         [(4242, "/usr/bin/python3", 2048), (77, "weird, name", 0)])
        self.assertEqual(hw.compute_apps(lambda cmd: (127, "")), [])


if __name__ == "__main__":
    unittest.main()
