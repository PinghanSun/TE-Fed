"""Launch a material-based IR preview with a local Blender installation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv=None):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=folder / 'vehicle_environment.blend')
    parser.add_argument('--preset', choices=['nir', 'lwir'], default='lwir')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--camera', default='Camera')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int)
    parser.add_argument('--rays', type=int, choices=[1, 4], default=4)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        scene = args.scene.expanduser().resolve(strict=True)
        if scene.suffix.lower() != '.blend' or not scene.is_file():
            raise ValueError('--scene must identify a blend file')
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError('Choose a fresh output directory')
        candidates = [folder / 'render_scene_ir.py', folder / 'source/render_scene_ir.py']
        stage = next((path for path in candidates if path.is_file()), None)
        if stage is None:
            raise FileNotFoundError('render_scene_ir.py is missing')
        executable = shutil.which(args.blender)
        if executable is None:
            executable = str(Path(args.blender).expanduser().resolve(strict=True))
            if not Path(executable).is_file():
                raise ValueError('--blender must identify its executable')
        if not 32 <= args.width <= 4096 or args.height is not None and not 32 <= args.height <= 4096:
            raise ValueError('Image dimensions must be within 32-4096')
        stage_args = ['--preset', args.preset, '--camera', args.camera, '--width', str(args.width),
                      '--samples', str(args.rays), '--with-rgb', '--output', str(output)]
        if args.height is not None:
            stage_args.extend(['--height', str(args.height)])
        configuration = args.config
        if configuration is None:
            bundled = folder / 'parameters' / (args.preset + '_example.json')
            configuration = bundled if bundled.is_file() else None
        if configuration is not None:
            stage_args.extend(['--config', str(configuration.expanduser().resolve(strict=True))])
        if args.check:
            stage_args.append('--check')
        command = [executable, '--background', str(scene), '--python-exit-code', '1', '--python', str(stage), '--', *stage_args]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1200)
        if not args.check and output.is_dir():
            (output / 'execution.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Blender exited with {result.returncode}:\n{result.stderr[-2000:]}')
        if args.check:
            prefix = 'IR_CHECK_RESULT '
            checks = [json.loads(line[len(prefix):]) for line in result.stdout.splitlines() if line.startswith(prefix)]
            if len(checks) != 1:
                raise RuntimeError('The IR stage did not return a valid preflight result')
            checked = checks[0]
            if (checked.get('configuration_valid') is not True or checked.get('mode') != args.preset
                    or checked.get('scene_checked') is not True or checked.get('camera') != args.camera
                    or Path(checked.get('source_blend', '')).resolve() != scene):
                raise RuntimeError('IR preflight did not validate the requested scene, camera and mode')
            print(json.dumps({'status': 'ready', 'preset': args.preset, 'scene': str(scene)}, indent=2))
            return 0
        data = json.loads((output / 'ir.json').read_text(encoding='utf-8'))
        if data['mode'] != args.preset or not data['source_unchanged'] or not (output / 'rgb.png').is_file():
            raise RuntimeError('IR output verification failed')
        print(json.dumps({'status': 'ok', 'mode': data['mode'], 'output': str(output),
                          'image_size': data['image_size'], 'parameter_basis': data['parameter_basis']}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())
