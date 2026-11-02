"""Check a built wheel outside the checkout without training dependencies."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def check_wheel(wheel: Path) -> str:
    wheel = wheel.resolve(strict=True)
    with TemporaryDirectory(prefix="tefed-wheel-") as directory:
        root = Path(directory)
        installed = root / "installed"
        with ZipFile(wheel) as archive:
            archive.extractall(installed)
        # -I -S excludes the source tree, editable installs and user/site packages.
        code = '''
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import tefed
from tefed.rendering import BlenderRenderer
installed = Path(sys.argv[1]).resolve()
assert Path(tefed.__file__).resolve().is_relative_to(installed)
renderer = BlenderRenderer(sys.executable)
for script in (renderer.script_path, renderer.generic_script_path):
    assert script.is_file() and script.is_relative_to(installed), script
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
common = renderer.generic_script_path.with_name("blender_render.py")
assert common.is_file(), common
from tefed.cli import main
try:
    main(["--help"])
except SystemExit as error:
    assert error.code == 0, error.code
print("Wheel imports, CLI help and both Blender entry points: passed")
'''
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code, str(installed)],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    wheels = sorted(args.dist.glob("*.whl"))
    if len(wheels) != 1:
        parser.error("--dist must contain exactly one built wheel")
    print(check_wheel(wheels[0]), end="")


if __name__ == "__main__":
    main()
