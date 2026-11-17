"""The installer tells the first-run costs and ends with the exact PATH command."""

import unittest

from tests.fixtures import REPO

INSTALLER = (REPO / "engine" / "cdn" / "install.sh").read_text()


class InstallerText(unittest.TestCase):
    def test_it_says_the_engine_environment_is_about_5_gb_once(self):
        self.assertIn("about 30 MB", INSTALLER)
        self.assertRegex(INSTALLER, r"engine environment: about 5 GB, once")

    def test_it_ends_with_the_exact_path_command(self):
        self.assertIn('export PATH=\\"${LINK_DIR}:\\$PATH\\"', INSTALLER)


if __name__ == "__main__":
    unittest.main()
