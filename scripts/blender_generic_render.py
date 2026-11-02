"""Compatibility launcher for the packaged Blender entry point."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    runpy.run_module("tefed.blender_scripts.blender_generic_render", run_name="__main__")
