"""Render a scene example with paths resolved relative to its request file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def renderer_path():
    folder = Path(__file__).resolve().parent
    candidates = [folder / 'renderer' / 'blender_generic_render.py',
                  folder.parent / 'src' / 'tefed' / 'blender_scripts' / 'blender_generic_render.py']
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('The generic Blender renderer is missing')


def prepare(args):
    request_file = args.request.expanduser().resolve(strict=True)
    request = json.loads(request_file.read_text(encoding='utf-8-sig'))
    scene = Path(request['scene_path']).expanduser()
    if not scene.is_absolute():
        scene = request_file.parent / scene
    scene = scene.resolve(strict=True)
    if not scene.is_file() or scene.suffix.lower() != '.blend':
        raise ValueError('scene_path must identify a blend file')
    script = renderer_path()
    common = script.with_name('blender_render.py')
    if not common.is_file():
        raise FileNotFoundError('The shared Blender renderer is missing')
    executable = shutil.which(args.blender)
    if executable is None:
        candidate = Path(args.blender).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError('Blender was not found; pass --blender with its executable path')
        executable = str(candidate)
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Choose a fresh output directory')
    if scene == output or output in scene.parents or output in request_file.parents:
        raise ValueError('The output directory must be separate from the input files')
    request['scene_path'] = str(scene)
    return request_file, request, script, executable, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path(__file__).resolve().with_name('render_request.json'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--check', action='store_true', help='Check paths without rendering or creating files')
    args = parser.parse_args(argv)
    try:
        source, payload, script, executable, output = prepare(args)
        if args.check:
            print(json.dumps({'status': 'ready', 'request': str(source), 'scene': payload['scene_path'],
                              'renderer': str(script), 'blender': executable, 'output': str(output)}, indent=2))
            return 0
        output.mkdir(parents=True, exist_ok=False)
        bound_request = output / 'request.json'
        bound_request.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        response = output / 'response.json'
        command = [executable, '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(script),
                   '--', '--request', str(bound_request), '--response', str(response)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=600)
        (output / 'execution.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(f'Blender exited with {result.returncode}; see {output / "execution.log"}')
        data = json.loads(response.read_text(encoding='utf-8'))
        if data.get('status') != 'ok':
            raise RuntimeError(f'Renderer rejected the request: {data.get("error", "unknown error")}')
        for key in ['rgb', 'depth', 'nir', 'instance_mask']:
            path = Path(data[key]).resolve(strict=True)
            if not path.is_relative_to(output) or not path.is_file():
                raise RuntimeError(f'Invalid renderer output: {key}')
        print(json.dumps({'status': 'ok', 'response': str(response),
                          'visible_targets': len(data['targets']), 'output': str(output)}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())
