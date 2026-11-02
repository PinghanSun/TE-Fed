"""RGB/IR pairing and local example extraction using small image fixtures."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from scripts.extract_dronevehicle_example import extract_example, main as extract_main
from scripts.prepare_manifest import (dronevehicle_records, main as prepare_main,
                                     read_class_aliases, read_class_map)
from tefed.data import load_manifest, load_sample


class DroneVehiclePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.annotations = self.root / 'annotations'
        self.rgb_dir = self.root / 'rgb'
        self.ir_dir = self.root / 'ir'
        for directory in (self.annotations, self.rgb_dir, self.ir_dir):
            directory.mkdir()
        self.rgb = self.rgb_dir / '0001.png'
        self.ir = self.ir_dir / '0001.png'
        self.annotation = self.annotations / '0001.txt'
        self.mapping = self.root / 'classes.json'
        self.class_map = {'van': 0, 'truck': 1, 'car': 2, 'bus': 3, 'freight_car': 4}
        self.mapping.write_text(json.dumps(self.class_map), encoding='utf-8')
        self.rgb_array = np.full((24, 40, 3), [70, 90, 110], dtype=np.uint8)
        self.ir_array = np.arange(24 * 40, dtype=np.uint16).reshape(24, 40).astype(np.uint8)
        Image.fromarray(self.rgb_array).save(self.rgb)
        Image.fromarray(self.ir_array).save(self.ir)
        self.annotation.write_text('imagesource: fixture\ngsd: 1\n4 3 20 3 20 15 4 15 car 0\n', encoding='utf-8')
        self.output = self.root / 'example'

    def records(self, **kwargs):
        return dronevehicle_records(self.annotations, self.rgb_dir, 'c1', 'test',
                                    self.mapping, '.png', 'error', **kwargs)

    def extract(self, **kwargs):
        return extract_example(self.rgb, self.ir, self.annotation, self.mapping,
                               'dv:0001', 'c1', self.output, **kwargs)

    def input_bytes(self):
        return {path: path.read_bytes() for directory in (self.annotations, self.rgb_dir, self.ir_dir)
                for path in directory.iterdir() if path.is_file()} | {self.mapping: self.mapping.read_bytes()}

    def xml(self, name='car', difficulty=0, size=(40, 24)):
        tree = ElementTree.Element('annotation')
        dimensions = ElementTree.SubElement(tree, 'size')
        for key, value in zip(('width', 'height'), size):
            ElementTree.SubElement(dimensions, key).text = str(value)
        item = ElementTree.SubElement(tree, 'object')
        ElementTree.SubElement(item, 'name').text = name
        ElementTree.SubElement(item, 'difficult').text = str(difficulty)
        polygon = ElementTree.SubElement(item, 'polygon')
        for corner, xy in enumerate(((4, 3), (20, 3), (20, 15), (4, 15)), 1):
            for axis, value in zip(('x', 'y'), xy):
                ElementTree.SubElement(polygon, f'{axis}{corner}').text = str(value)
        path = self.annotations / '0001.xml'
        ElementTree.ElementTree(tree).write(path, encoding='utf-8', xml_declaration=True)
        return path

    def aliases(self, mapping=None):
        path = self.root / 'aliases.json'
        path.write_text(json.dumps(mapping if mapping is not None else {'feright car': 'freight_car',
                                                                       'feright_car': 'freight_car'}), encoding='utf-8')
        return path

    def test_paired_records_keep_class_order_and_ir_paths(self):
        records, metadata = self.records(ir_root=self.ir_dir)
        self.assertEqual(records[0]['nir'], str(self.ir.resolve()))
        self.assertEqual(records[0]['labels'][0]['class_id'], 2)
        self.assertEqual(metadata['class_names'], list(self.class_map))
        self.assertEqual(metadata['modalities'], ['RGB', 'IR'])
        np.testing.assert_allclose(records[0]['labels'][0]['box'], [.1, .125, .5, .625])

    def test_rgb_only_conversion_is_still_supported(self):
        records, metadata = self.records()
        self.assertNotIn('nir', records[0])
        self.assertNotIn('modalities', metadata)

    def test_missing_rgb_or_ir_pair_is_rejected(self):
        self.ir.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing IR pair'):
            self.records(ir_root=self.ir_dir)
        self.rgb.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing RGB pair'):
            self.records()

    def test_rgb_ir_resolution_must_match(self):
        Image.new('L', (40, 20)).save(self.ir)
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            self.records(ir_root=self.ir_dir)

    def test_duplicate_casefolded_ids_are_rejected(self):
        for name in ('ss.png', '\u00df.png'):
            Image.fromarray(self.rgb_array).save(self.rgb_dir / name)
        with self.assertRaisesRegex(ValueError, 'Duplicate RGB sample ID'):
            self.records(ir_root=self.ir_dir)

    def test_multichannel_and_high_bit_ir_are_rejected(self):
        for values, message in [(self.rgb_array, 'single-channel'),
                                (self.ir_array.astype(np.uint16), 'uint8')]:
            with self.subTest(dtype=str(values.dtype), shape=values.shape):
                Image.fromarray(values).save(self.ir)
                with self.assertRaisesRegex(ValueError, message):
                    self.records(ir_root=self.ir_dir)

    def test_explicit_ir_suffix_and_float_inputs(self):
        values = self.ir_array.astype(np.float32) / 255
        np.save(self.ir_dir / '0001.npy', values)
        records, _ = self.records(ir_root=self.ir_dir, ir_suffix='.npy')
        self.assertTrue(records[0]['nir'].endswith('0001.npy'))
        for bad in (float('nan'), -0.1, 1.1):
            values[0, 0] = bad
            np.save(self.ir_dir / '0001.npy', values)
            with self.assertRaisesRegex(ValueError, 'finite floating-point'):
                self.records(ir_root=self.ir_dir, ir_suffix='.npy')

    def test_invalid_category_maps_are_rejected(self):
        for mapping in ({'car': False}, {'car': 1}, {'car': 0, 'bus': 0}, {'': 0}):
            self.mapping.write_text(json.dumps(mapping), encoding='utf-8')
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                read_class_map(self.mapping)

    def test_unknown_classes_and_difficult_objects_are_explicit(self):
        self.annotation.write_text('4 3 20 3 20 15 4 15 unknown 0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown category'):
            self.records()
        self.annotation.write_text('4 3 20 3 20 15 4 15 car 1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'difficult-policy'):
            self.extract()
        self.extract(difficult_policy='drop')
        labels = json.loads((self.output / 'labels.json').read_text(encoding='utf-8'))
        self.assertEqual(labels['labels'], [])

    def test_prepare_cli_roundtrip_reads_paired_ir(self):
        target = self.root / 'manifest.json'
        with redirect_stdout(StringIO()):
            prepare_main(['--input', str(self.annotations), '--output', str(target),
                          '--input-format', 'dronevehicle', '--image-root', str(self.rgb_dir),
                          '--ir-root', str(self.ir_dir), '--image-suffix', '.png',
                          '--class-map', str(self.mapping), '--client-id', 'c1', '--split', 'test', '--check-files'])
        manifest = load_manifest(target, strict_files=True)
        sample = load_sample(manifest.records[0], size=(40, 24), require_modalities=('nir',))
        np.testing.assert_array_equal(sample['nir'], self.ir_array)

    def test_prepare_cannot_overwrite_a_source_file(self):
        before = self.annotation.read_bytes()
        with self.assertRaisesRegex(ValueError, 'must not replace'):
            prepare_main(['--input', str(self.annotations), '--output', str(self.annotation),
                          '--input-format', 'dronevehicle', '--image-root', str(self.rgb_dir),
                          '--image-suffix', '.png', '--class-map', str(self.mapping),
                          '--client-id', 'c1', '--split', 'test'])
        self.assertEqual(self.annotation.read_bytes(), before)

    def test_extraction_preserves_pixels_labels_and_sources(self):
        before = self.input_bytes()
        result = self.extract()
        self.assertFalse(result['inference_executed'])
        self.assertEqual(self.input_bytes(), before)
        with Image.open(self.output / 'inputs/rgb.png') as image:
            np.testing.assert_array_equal(np.array(image), self.rgb_array)
        with Image.open(self.output / 'inputs/ir.png') as image:
            np.testing.assert_array_equal(np.array(image), self.ir_array)
        preview = json.loads((self.output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(preview['task'], 'dronevehicle')
        self.assertEqual(set(preview['files']), {'rgb', 'ir'})
        self.assertEqual(preview['image_size'], [40, 24])
        self.assertEqual(preview['preprocessing']['crop'], 'none')
        self.assertEqual((self.output / 'source_annotation.txt').read_bytes(), self.annotation.read_bytes())
        manifest = load_manifest(self.output / 'data_manifest.json', strict_files=True)
        sample = load_sample(manifest.records[0], size=(40, 24), require_modalities=('nir',))
        self.assertEqual(manifest.metadata['class_names'], list(self.class_map))
        self.assertEqual(sample['targets'][0]['class_id'], 2)
        np.testing.assert_array_equal(sample['nir'], self.ir_array)

    def test_extraction_float_ir_records_training_quantization(self):
        values = np.linspace(0, 1, 24 * 40, dtype=np.float32).reshape(24, 40)
        ir = self.ir_dir / '0001.npy'
        np.save(ir, values)
        extract_example(self.rgb, ir, self.annotation, self.mapping, '0001', 'c1', self.output)
        with Image.open(self.output / 'inputs/ir.png') as image:
            np.testing.assert_array_equal(np.array(image), np.rint(values * 255).astype(np.uint8))
        metadata = json.loads((self.output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertIn('round(value * 255)', metadata['preprocessing']['ir'])

    def test_extraction_rejects_existing_output_without_touching_it(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_invalid_inputs_leave_no_output(self):
        before = self.input_bytes()
        with self.assertRaisesRegex(ValueError, 'unknown category'):
            self.annotation.write_text('4 3 20 3 20 15 4 15 invalid 0\n', encoding='utf-8')
            self.extract()
        self.assertFalse(self.output.exists())
        self.annotation.write_bytes(before[self.annotation])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            extract_example(self.rgb, self.ir, self.annotation, self.mapping, ' ', 'c1', self.output)
        self.assertFalse(self.output.exists())

    def test_example_write_failure_cleans_staging_directory(self):
        before = self.input_bytes()
        original = Image.Image.save
        calls = 0

        def save(image, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('write failed')
            return original(image, *args, **kwargs)

        with patch('scripts.extract_dronevehicle_example.Image.Image.save', new=save):
            with self.assertRaisesRegex(OSError, 'write failed'):
                self.extract()
        self.assertEqual(calls, 2)
        self.assertEqual(self.input_bytes(), before)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.tefed-example-*')), [])

    def test_a_concurrently_created_output_is_preserved(self):
        from scripts.extract_dronevehicle_example import _json
        marker = self.output / 'keep.txt'

        def write(path, value):
            _json(path, value)
            if not self.output.exists():
                self.output.mkdir()
                marker.write_text('other writer', encoding='utf-8')

        with patch('scripts.extract_dronevehicle_example._json', new=write):
            with self.assertRaisesRegex(FileExistsError, 'appeared during extraction'):
                self.extract()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'other writer')
        self.assertEqual(list(self.root.glob('.tefed-example-*')), [])

    def test_explicit_pair_ids_must_match_and_unlabeled_manifest_hides_gt(self):
        other = self.ir_dir / 'other.png'
        Image.fromarray(self.ir_array).save(other)
        with self.assertRaisesRegex(ValueError, 'share a sample ID'):
            extract_example(self.rgb, other, self.annotation, self.mapping, '0001', 'c1', self.output)
        self.extract(split='unlabeled')
        data = json.loads((self.output / 'data_manifest.json').read_text(encoding='utf-8'))
        self.assertNotIn('labels', data['records'][0])
        self.assertTrue(json.loads((self.output / 'labels.json').read_text(encoding='utf-8'))['labels'])

    def test_extract_cli_writes_a_preview_readable_example(self):
        with redirect_stdout(StringIO()):
            result = extract_main(['--rgb', str(self.rgb), '--ir', str(self.ir),
                                   '--annotation', str(self.annotation), '--class-map', str(self.mapping),
                                   '--sample-id', '0001', '--client-id', 'c1', '--output', str(self.output)])
        self.assertEqual(result, 0)
        from tefed.preview import preview_example
        result = preview_example(self.output, self.root / 'preview')
        self.assertFalse(result['inference_executed'])
        self.assertTrue((self.root / 'preview/rgb_ir.png').is_file())

    def test_xml_polygons_use_declared_size_and_keep_source_coordinates(self):
        self.annotation.unlink()
        xml = self.xml()
        before = xml.read_bytes()
        records, _ = self.records(ir_root=self.ir_dir)
        np.testing.assert_allclose(records[0]['labels'][0]['box'], [.1, .125, .5, .625])
        extract_example(self.rgb, self.ir, xml, self.mapping, '0001', 'c1', self.output)
        self.assertEqual((self.output / 'source_annotation.xml').read_bytes(), before)
        self.assertEqual(records[0]['labels'][0]['class_id'], 2)

    def test_xml_and_txt_for_one_id_are_ambiguous(self):
        self.xml()
        with self.assertRaisesRegex(ValueError, 'Ambiguous annotation ID'):
            self.records(ir_root=self.ir_dir)

    def test_xml_size_mismatch_or_missing_polygon_is_rejected(self):
        self.annotation.unlink()
        xml = self.xml(size=(39, 24))
        with self.assertRaisesRegex(ValueError, 'XML size'):
            self.records(ir_root=self.ir_dir)
        xml = self.xml()
        tree = ElementTree.parse(xml)
        polygon = tree.getroot().find('object/polygon')
        polygon.remove(polygon.find('x4'))
        tree.write(xml, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'eight polygon coordinates'):
            self.records(ir_root=self.ir_dir)

    def test_invalid_xml_and_missing_size_are_rejected(self):
        self.annotation.unlink()
        xml = self.xml()
        xml.write_text('<annotation>', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'invalid XML'):
            self.records(ir_root=self.ir_dir)
        xml = self.xml()
        tree = ElementTree.parse(xml)
        tree.getroot().remove(tree.getroot().find('size'))
        tree.write(xml, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'XML needs integer size'):
            self.records(ir_root=self.ir_dir)

    def test_native_spelling_requires_explicit_aliases(self):
        self.annotation.unlink()
        self.xml(name='feright car')
        with self.assertRaisesRegex(ValueError, 'unknown category'):
            self.records(ir_root=self.ir_dir)
        records, metadata = self.records(ir_root=self.ir_dir, class_aliases_path=self.aliases())
        self.assertEqual(records[0]['labels'][0]['class_id'], 4)
        self.assertEqual(metadata['source_class_aliases']['feright car'], 'freight_car')

    def test_aliases_cannot_invent_classes_or_remap_canonical_names(self):
        for mapping in ({'feright car': 'missing'}, {'car': 'truck'}, {'feright car': 4}, {'': 'car'}):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, 'existing canonical'):
                read_class_aliases(self.aliases(mapping), self.class_map)

    def test_txt_alias_with_space_and_difficulty_is_supported(self):
        self.annotation.write_text('4 3 20 3 20 15 4 15 feright car 1\n', encoding='utf-8')
        aliases = self.aliases()
        with self.assertRaisesRegex(ValueError, 'difficult-policy'):
            self.records(ir_root=self.ir_dir, class_aliases_path=aliases)
        rows, metadata = dronevehicle_records(self.annotations, self.rgb_dir, 'c1', 'test',
                                             self.mapping, '.png', 'include', ir_root=self.ir_dir,
                                             class_aliases_path=aliases)
        self.assertEqual(rows[0]['labels'][0]['class_id'], 4)
        self.assertEqual(metadata['difficult_policy'], 'include')

    def test_xml_difficulty_uses_the_same_include_and_drop_rules(self):
        self.annotation.unlink()
        self.xml(difficulty=1)
        with self.assertRaisesRegex(ValueError, 'difficult-policy'):
            self.records(ir_root=self.ir_dir)
        rows, _ = dronevehicle_records(self.annotations, self.rgb_dir, 'c1', 'test',
                                       self.mapping, '.png', 'drop', ir_root=self.ir_dir)
        self.assertEqual(rows[0]['labels'], [])

    def test_both_clis_record_explicit_xml_aliases(self):
        self.annotation.unlink()
        xml = self.xml(name='feright_car')
        aliases = self.aliases()
        target = self.root / 'prepared.json'
        with redirect_stdout(StringIO()):
            prepare_main(['--input', str(self.annotations), '--output', str(target),
                          '--input-format', 'dronevehicle', '--image-root', str(self.rgb_dir),
                          '--ir-root', str(self.ir_dir), '--image-suffix', '.png',
                          '--class-map', str(self.mapping), '--class-aliases', str(aliases),
                          '--client-id', 'c1', '--split', 'test', '--check-files'])
            extract_main(['--rgb', str(self.rgb), '--ir', str(self.ir), '--annotation', str(xml),
                          '--class-map', str(self.mapping), '--class-aliases', str(aliases),
                          '--sample-id', '0001', '--client-id', 'c1', '--output', str(self.output)])
        self.assertEqual(load_manifest(target).records[0].labels[0].class_id, 4)
        metadata = json.loads((self.output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['preprocessing']['class_aliases']['feright_car'], 'freight_car')
        self.assertEqual(json.loads((self.output / 'class_aliases.json').read_text(encoding='utf-8')),
                         json.loads(aliases.read_text(encoding='utf-8')))
        self.assertEqual((self.output / 'source_annotation.xml').read_bytes(), xml.read_bytes())


if __name__ == '__main__':
    unittest.main()
