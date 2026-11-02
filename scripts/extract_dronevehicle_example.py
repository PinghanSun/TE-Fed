"""Prepare one local RGB/IR sample and its horizontal-box annotations."""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tefed.data import validate_manifest

if __package__:
    from .prepare_manifest import read_class_aliases, read_class_map, read_dronevehicle_labels, read_ir
else:
    from prepare_manifest import read_class_aliases, read_class_map, read_dronevehicle_labels, read_ir


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def extract_example(rgb, ir, annotation, class_map, sample_id, client_id, output,
                    *, split='test', difficult_policy='error', class_aliases=None):
    """Write a fresh example directory after validating the selected pair."""
    paths = {name: Path(value).expanduser().resolve(strict=True) for name, value in (
        ('rgb', rgb), ('ir', ir), ('annotation', annotation), ('class_map', class_map))}
    if class_aliases is not None:
        paths['class_aliases'] = Path(class_aliases).expanduser().resolve(strict=True)
    if any(not path.is_file() for path in paths.values()):
        raise ValueError('RGB, IR, annotation and class-map must be files')
    if len({paths[name].stem.casefold() for name in ('rgb', 'ir', 'annotation')}) != 1:
        raise ValueError('RGB, IR and annotation filenames must share a sample ID')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new example directory: {output}')
    sample_id, client_id = str(sample_id).strip(), str(client_id).strip()
    if not sample_id or not client_id:
        raise ValueError('sample-id and client-id must be nonempty')
    source_bytes = {name: path.read_bytes() for name, path in paths.items()}
    mapping = read_class_map(paths['class_map'])
    aliases = read_class_aliases(paths.get('class_aliases'), mapping)
    classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    with Image.open(paths['rgb']) as image:
        rgb_array = np.array(image)
    if rgb_array.dtype != np.uint8 or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError('RGB must decode to uint8 with three channels')
    height, width = rgb_array.shape[:2]
    ir_array = read_ir(paths['ir'], (width, height))
    operation = 'preserve uint8 samples'
    if np.issubdtype(ir_array.dtype, np.floating):
        ir_array = np.rint(ir_array * 255).astype(np.uint8)
        operation = 'round(value * 255) from [0,1] floating-point values'
    labels = read_dronevehicle_labels(paths['annotation'], (width, height), mapping, difficult_policy, aliases)
    record = {'id': sample_id, 'client_id': client_id, 'split': split,
              'rgb': 'inputs/rgb.png', 'nir': 'inputs/ir.png'}
    if split != 'unlabeled':
        record['labels'] = labels
    data = {'task': 'dronevehicle', 'class_names': classes,
            'modality_names': {'nir': 'IR'}, 'source_class_aliases': aliases, 'records': [record]}
    validate_manifest([record], metadata={key: value for key, value in data.items() if key != 'records'},
                      base_dir=output)
    sources = {name: {'filename': paths[name].name, 'sha256': sha256(payload).hexdigest()}
               for name, payload in source_bytes.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.tefed-example-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'example'
        (stage / 'inputs').mkdir(parents=True)
        arrays = {'rgb': rgb_array, 'ir': ir_array}
        files = {}
        for name, array in arrays.items():
            path = stage / 'inputs' / f'{name}.png'
            Image.fromarray(array).save(path)
            files[name] = {'path': f'inputs/{name}.png', 'sha256': sha256(path.read_bytes()).hexdigest(),
                           'dtype': str(array.dtype), 'shape': list(array.shape)}
        manifest = {'task': 'dronevehicle', 'sample_id': sample_id, 'client_id': client_id,
                    'source_split': split, 'image_size': [width, height],
                    'files': files, 'class_names': classes, 'source_files': sources,
                    'preprocessing': {'rgb': 'decode RGB pixels and encode lossless PNG',
                                      'ir': operation, 'crop': 'none', 'registration': 'unchanged',
                                      'class_aliases': aliases},
                    'inference_executed': False}
        _json(stage / 'manifest.json', manifest)
        _json(stage / 'data_manifest.json', data)
        _json(stage / 'labels.json', {'sample_id': sample_id, 'image_size': [width, height],
                                    'coordinate_system': 'normalized RGB xyxy',
                                    'class_names': classes, 'labels': labels,
                                    'source_class_aliases': aliases,
                                    'oriented_conversion': 'minimum enclosing horizontal box',
                                    'difficult_policy': difficult_policy})
        _json(stage / 'class_map.json', mapping)
        if aliases:
            _json(stage / 'class_aliases.json', aliases)
        (stage / ('source_annotation' + paths['annotation'].suffix.lower())).write_bytes(source_bytes['annotation'])
        if any(path.read_bytes() != source_bytes[name] for name, path in paths.items()):
            raise RuntimeError('A source file changed during extraction')
        if output.exists():
            raise FileExistsError(f'Example directory appeared during extraction: {output}')
        stage.rename(output)
    return {'output': str(output), 'sample_id': sample_id, 'class_names': classes,
            'labels': len(labels), 'image_size': [width, height], 'inference_executed': False}


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--rgb', required=True, help='Local RGB image')
    command.add_argument('--ir', required=True, help='Same-ID single-channel IR image or [0,1] float NPY')
    command.add_argument('--annotation', required=True, help='Same-ID RGB-side oriented annotation XML or TXT')
    command.add_argument('--class-map', required=True, help='Training category-name to class-ID JSON')
    command.add_argument('--class-aliases', help='Source category aliases mapped to canonical class-map names')
    command.add_argument('--sample-id', required=True)
    command.add_argument('--client-id', required=True)
    command.add_argument('--split', choices=('labeled', 'unlabeled', 'validation', 'test', 'hard'), default='test')
    command.add_argument('--difficult-policy', choices=('error', 'include', 'drop'), default='error')
    command.add_argument('--output', required=True, help='New example directory')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    result = extract_example(args.rgb, args.ir, args.annotation, args.class_map,
                             args.sample_id, args.client_id, args.output,
                             split=args.split, difficult_policy=args.difficult_policy,
                             class_aliases=args.class_aliases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f'Example extraction failed: {error}', file=sys.stderr)
        raise SystemExit(2)
