import tempfile
import unittest
from pathlib import Path

from goethe import store


class Paths(unittest.TestCase):
    def test_xdg_variables_win(self):
        env = {"HOME": "/h", "XDG_DATA_HOME": "/d", "XDG_STATE_HOME": "/s", "XDG_CONFIG_HOME": "/c"}
        self.assertEqual((store.data_dir(env), store.state_dir(env), store.config_path(env)),
                         (Path("/d/goethe"), Path("/s/goethe"), Path("/c/goethe/config.toml")))

    def test_home_fallbacks(self):
        env = {"HOME": "/h"}
        self.assertEqual((store.data_dir(env), store.state_dir(env), store.config_path(env)),
                         (Path("/h/.local/share/goethe"), Path("/h/.local/state/goethe"),
                          Path("/h/.config/goethe/config.toml")))

    def test_recipe_home_order(self):
        with tempfile.TemporaryDirectory() as home:
            env = {"HOME": home}
            self.assertEqual(store.recipe_home(env), Path(home) / ".local/share/goethe/nvfp4")
            (Path(home) / ".cache/goethe/nvfp4").mkdir(parents=True)
            self.assertEqual(store.recipe_home(env), Path(home) / ".cache/goethe/nvfp4")
            self.assertEqual(store.recipe_home({**env, "NVFP4_HOME": "/x"}), Path("/x"))


if __name__ == "__main__":
    unittest.main()
