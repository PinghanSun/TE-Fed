"""Scene metadata validation and non-overwriting publication."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/create_example_scene.py'
SPEC = importlib.util.spec_from_file_location('scene_starter', SCRIPT)
starter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(starter)


class SceneStarterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'example.blend'

    def test_task_maps_cover_instances_and_class_order(self):
        tomato = starter.prepare('tomato', self.output)
        self.assertEqual(tomato['scene_class_map'], {'non_ripe': 0, 'ripe': 1})
        self.assertEqual(tomato['class_names'], ['non-ripe', 'ripe'])
        drone = starter.prepare('dronevehicle', self.output)
        self.assertEqual(set(drone['scene_class_map']), {'car', 'truck', 'bus', 'van', 'freight_car'})
        self.assertEqual(set(drone['scene_class_map'].values()), set(range(5)))
        self.assertFalse(self.output.exists())

    def test_custom_ids_keep_semantic_root_names(self):
        classes = self.root / 'classes.json'
        classes.write_text(json.dumps({'ripe': 0, 'non-ripe': 1}), encoding='utf-8-sig')
        plan = starter.prepare('tomato', self.output, classes)
        self.assertEqual(plan['class_names'], ['ripe', 'non-ripe'])
        self.assertEqual(plan['scene_class_map'], {'ripe': 0, 'non_ripe': 1})

    def test_invalid_or_incomplete_maps_are_rejected(self):
        cases = [{'ripe': 0}, {'ripe': 0, 'non-ripe': 0},
                 {'ripe': True, 'non-ripe': 0}, {'ripe': 3, 'non-ripe': 0},
                 {'ripe': 0, 'unknown': 1}, ['ripe', 'non-ripe']]
        for mapping in cases:
            with self.subTest(mapping=mapping):
                path = self.root / 'classes.json'
                path.write_text(json.dumps(mapping), encoding='utf-8')
                with self.assertRaises(ValueError):
                    starter.prepare('tomato', self.output, path)
        self.assertFalse(self.output.exists())

    def test_existing_scene_or_sidecar_is_preserved(self):
        for suffix in ('.blend', '.scene.json', '.request.json'):
            with self.subTest(suffix=suffix):
                path = self.output.with_suffix(suffix)
                path.write_bytes(b'existing input')
                with self.assertRaises(FileExistsError):
                    starter.prepare('tomato', self.output)
                self.assertEqual(path.read_bytes(), b'existing input')
                path.unlink()

    def test_output_suffix_and_resolution_validation(self):
        with self.assertRaises(ValueError):
            starter.prepare('tomato', self.output.with_suffix('.json'))
        for size in ((0, 640), (640, 4097), (640,), (True, 640)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                starter.prepare('tomato', self.output, resolution=size)

    def fake_blender(self):
        def save(filepath, check_existing):
            Path(filepath).write_bytes(b'software scene fixture')
        return SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(save_as_mainfile=save)))

    def test_publication_writes_matching_request_and_metadata(self):
        plan = starter.prepare('dronevehicle', self.output)
        with patch.dict(sys.modules, {'bpy': self.fake_blender()}), patch.object(starter, 'build_scene'):
            result = starter.create(plan)
        metadata = json.loads(self.output.with_suffix('.scene.json').read_text(encoding='utf-8'))
        request = json.loads(self.output.with_suffix('.request.json').read_text(encoding='utf-8'))
        self.assertEqual(result, metadata)
        self.assertEqual(metadata['data_origin'], 'procedural_example')
        self.assertEqual(request['scene_path'], str(self.output))
        self.assertEqual(request['scene_class_map'], metadata['scene_class_map'])
        self.assertIn(request['descriptor']['source_class'], metadata['class_map'].values())
        self.assertEqual(self.output.read_bytes(), b'software scene fixture')

    def test_failed_publication_removes_only_its_new_files(self):
        plan = starter.prepare('tomato', self.output)
        metadata = self.output.with_suffix('.scene.json')
        metadata.write_bytes(b'created by another caller')
        with patch.dict(sys.modules, {'bpy': self.fake_blender()}), patch.object(starter, 'build_scene'):
            with self.assertRaises(FileExistsError):
                starter.create(plan)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix('.request.json').exists())
        self.assertEqual(metadata.read_bytes(), b'created by another caller')

    def test_check_command_needs_no_blender_and_creates_no_files(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--task', 'tomato', '--output',
                                 str(self.output), '--check'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['class_names'], ['non-ripe', 'ripe'])
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
