"""RGB/IR preview behavior using small software fixtures."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from tefed.preview import preview_example


class DualPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.example = self.root / 'example'
        self.example.mkdir()
        self.output = self.root / 'output'
        self.rgb = np.zeros((24, 64, 3), dtype=np.uint8)
        self.rgb[..., 0] = 37
        self.ir = np.tile(np.arange(64, dtype=np.uint8) * 4, (24, 1))
        self.manifest = {'task': 'dronevehicle', 'data_origin': 'software_fixture',
                         'image_size': [64, 24], 'files': {}}
        self.image('rgb', self.rgb)
        self.image('ir', self.ir)

    def image(self, name, values):
        path = self.example / f'{name}.png'
        Image.fromarray(values).save(path)
        self.manifest['files'][name] = {'path': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'dtype': str(values.dtype), 'shape': list(values.shape)}
        self.save()

    def save(self):
        (self.example / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_ir_values_and_rgb_bytes_survive_preview(self):
        before = {p.name: p.read_bytes() for p in self.example.iterdir()}
        result = preview_example(self.example, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.example.iterdir()})
        self.assertEqual((self.output / 'rgb.png').read_bytes(), before['rgb.png'])
        with Image.open(self.output / 'ir_display.png') as image:
            np.testing.assert_array_equal(np.array(image), np.repeat(self.ir[..., None], 3, axis=2))
        with Image.open(self.output / 'rgb_ir.png') as image:
            self.assertEqual(image.size, (144, 120))
        self.assertEqual(result['panel_labels'], ['(a) RGB', '(b) IR'])
        saved = json.loads((self.output / 'preview.json').read_text(encoding='utf-8'))
        self.assertFalse(saved['inference_executed'])
        self.assertEqual(saved['outputs']['combined'], 'rgb_ir.png')

    def test_missing_ir_rejects_without_creating_outputs(self):
        (self.example / 'ir.png').unlink()
        with self.assertRaises(FileNotFoundError):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_ir_dimension_mismatch_rejects(self):
        self.image('ir', self.ir[:, :-1])
        with self.assertRaisesRegex(ValueError, 'dimensions must match'):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_rgb_ir_is_not_interpreted_as_a_scalar_sensor(self):
        self.image('ir', np.repeat(self.ir[..., None], 3, axis=2))
        with self.assertRaisesRegex(ValueError, 'single-channel'):
            preview_example(self.example, self.output)

    def test_16bit_ir_requires_training_preprocessing(self):
        self.image('ir', self.ir.astype(np.uint16) * 100)
        with self.assertRaisesRegex(ValueError, '8-bit'):
            preview_example(self.example, self.output)

    def test_task_and_hash_are_checked_before_writing(self):
        for mutation in ('task', 'hash'):
            with self.subTest(mutation=mutation):
                self.manifest['task'] = 'unknown' if mutation == 'task' else 'dronevehicle'
                if mutation == 'hash':
                    self.manifest['files']['ir']['sha256'] = '0' * 64
                self.save()
                with self.assertRaises(ValueError):
                    preview_example(self.example, self.output)
                self.assertFalse(self.output.exists())

    def test_ir_display_scaling_cannot_be_added_silently(self):
        self.manifest['display_mappings'] = {'ir': {'vmin': 10, 'vmax': 20}}
        self.save()
        with self.assertRaisesRegex(ValueError, 'original uint8 IR'):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
