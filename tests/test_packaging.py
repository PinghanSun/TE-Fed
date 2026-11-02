"""Renderer entry points must travel with the installed package."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from tefed.rendering import BlenderRenderer


class RendererPackagingTests(unittest.TestCase):
    def test_default_renderers_are_package_local_and_syntax_valid(self):
        import tefed
        package = Path(tefed.__file__).resolve().parent
        renderer = BlenderRenderer(sys.executable)
        for path in (renderer.script_path, renderer.generic_script_path):
            with self.subTest(script=path.name):
                self.assertTrue(path.is_relative_to(package))
                self.assertTrue(path.is_file())
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
        self.assertTrue(renderer.generic_script_path.with_name("blender_render.py").is_file())

    def test_user_script_overrides_remain_available(self):
        with TemporaryDirectory() as directory:
            common = Path(directory) / "render.py"
            generic = Path(directory) / "generic.py"
            common.write_text("# user renderer\n", encoding="utf-8")
            generic.write_text("# user generic renderer\n", encoding="utf-8")
            renderer = BlenderRenderer(sys.executable, script_path=common,
                                       generic_script_path=generic)
            self.assertEqual(renderer.script_path, common.resolve())
            self.assertEqual(renderer.generic_script_path, generic.resolve())


if __name__ == "__main__":
    unittest.main()
