"""Class-ID order across training, checkpoints and Tomato rendering."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tefed import engine, inference, rendering


class ClassMappingTests(unittest.TestCase):
    def manifest(self, names=None):
        return SimpleNamespace(metadata={} if names is None else {'class_names': names}, records=[])

    def test_manifest_names_bind_in_class_id_order_and_survive_save(self):
        for names in (['ripe', 'non-ripe'], ['non-ripe', 'ripe']):
            with self.subTest(names=names), TemporaryDirectory() as directory:
                config = {'task': 'tomato', 'model': {'classes': 2}}
                manifest = self.manifest(names)
                engine.validate_manifest_config(config, manifest)
                self.assertEqual(config['model']['class_names'], names)
                self.assertIsNot(config['model']['class_names'], manifest.metadata['class_names'])
                bound = deepcopy(config)
                engine.validate_manifest_config(config, manifest)
                self.assertEqual(config, bound)
                model = torch.nn.Linear(2, 2)
                model.schema_version = 2
                checkpoint = Path(directory) / 'fixture.pt'
                engine.save_checkpoint(checkpoint, model, torch.nn.Linear(2, 2), config, {}, [], -1)
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved['config']['model']['class_names'], names)
                self.assertEqual(inference.prediction_names(None, saved['config']['model']), dict(enumerate(names)))

    def test_manifest_and_config_names_must_agree(self):
        config = {'model': {'classes': 2, 'class_names': ['ripe', 'non-ripe']}}
        with self.assertRaisesRegex(ValueError, 'class-ID order'):
            engine.validate_manifest_config(config, self.manifest(['non-ripe', 'ripe']))
        for names in (['ripe'], ['ripe', 'ripe'], ['ripe', '']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                engine.validate_class_names(names, 2)

    def test_missing_historical_names_are_not_inferred(self):
        config = {'model': {'classes': 2}}
        before = deepcopy(config)
        engine.validate_manifest_config(config, self.manifest())
        self.assertEqual(config, before)
        self.assertEqual(inference.prediction_names(None, config['model']), {})
        self.assertEqual(inference.prediction_names({'1': 'custom_name'}, config['model']), {1: 'custom_name'})

    def test_display_names_cannot_reinterpret_recorded_classes(self):
        model = {'classes': 2, 'class_names': ['non-ripe', 'ripe']}
        self.assertEqual(inference.prediction_names({'1': 'ripe'}, model), {0: 'non-ripe', 1: 'ripe'})
        with self.assertRaisesRegex(ValueError, 'checkpoint class-ID order'):
            inference.prediction_names({'0': 'ripe', '1': 'non-ripe'}, model)

    def test_evaluation_rejects_reversed_manifest_before_output(self):
        model = SimpleNamespace(config={'class_names': ['non-ripe', 'ripe']})
        config = {'model': {'classes': 2}}
        with self.assertRaisesRegex(ValueError, 'Evaluation class_names'):
            engine._check_evaluation_classes(model, self.manifest(['ripe', 'non-ripe']), config)

    def test_initial_weights_reject_known_label_order_conflict(self):
        model = SimpleNamespace(schema_version=2, config={'class_names': ['non-ripe', 'ripe']})
        checkpoint = {'model_schema_version': 2,
                      'config': {'model': {'classes': 2, 'class_names': ['ripe', 'non-ripe']}}}
        with patch.object(engine.torch, 'load', return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, 'Initial checkpoint class_names'):
                engine.inherit_initial(model, 'fixture.pt')

    def test_tomato_semantics_support_both_orders_and_reject_conflicts(self):
        self.assertEqual(rendering.tomato_class_map(), {'ripe': 0, 'non-ripe': 1})
        for names in (['ripe', 'non-ripe'], ['non-ripe', 'ripe']):
            expected = {'ripe': names.index('ripe'), 'non-ripe': names.index('non-ripe')}
            self.assertEqual(rendering.tomato_class_map(names), expected)
            self.assertEqual(rendering.tomato_class_map(names, expected), expected)
            reversed_map = {'ripe': expected['non-ripe'], 'non-ripe': expected['ripe']}
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                rendering.tomato_class_map(names, reversed_map)
        with self.assertRaisesRegex(ValueError, 'custom Tomato'):
            rendering.tomato_class_map(['class0', 'class1'])
        self.assertEqual(rendering.tomato_class_map(['class0', 'class1'], {'ripe': 1, 'non-ripe': 0}),
                         {'ripe': 1, 'non-ripe': 0})
        for mapping in ({'ripe': 0, 'non-ripe': 0}, {'ripe': True, 'non-ripe': 0}, {'ripe': 0}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                rendering.tomato_class_map(explicit=mapping)

    def test_server_preserves_source_id_and_passes_manifest_semantics(self):
        from tefed.server import construct_cases
        from tests.test_server import (FakeBank, FakeRenderer, FakeRouter, STATE, calibration,
                                       config_for, services, source_descriptor)
        descriptor = source_descriptor()
        backend = FakeRenderer()
        config = config_for(services(backend))
        config['model']['class_names'] = ['non-ripe', 'ripe']
        before = deepcopy(config['model'])
        with TemporaryDirectory() as folder:
            construct_cases([descriptor], FakeBank(), FakeRouter(),
                            {descriptor['source_state_key']: STATE}, ['client1'], calibration(),
                            config, 0, 0, folder)
        self.assertEqual(backend.requests[0]['tomato_class_map'], {'ripe': 1, 'non-ripe': 0})
        self.assertEqual(backend.requests[0]['descriptor']['source_class'], descriptor['source_class'])
        self.assertEqual(config['model'], before)


class RendererClassContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(rendering.__file__).with_name('blender_scripts') / 'blender_render.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        selected = [node for node in tree.body if
                    isinstance(node, ast.FunctionDef) and node.name in
                    {'fruit_class_id', 'fruit_body_color', 'body_objects', 'viewpoint_cue'}]
        namespace = {'re': re, 'math': math, 'np': np,
                     'FRUIT_PATTERN': re.compile(r'^(project_asset_tomato_(red|green|yellow)_\d+_\d+_\d+)_')}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        cls.functions = namespace

    def test_blender_groups_and_color_remain_physical_under_id_permutation(self):
        def body(color, index):
            return SimpleNamespace(name=f'project_asset_tomato_{color}_0_0_{index}_body', type='MESH',
                                   hide_render=False,
                                   material_slots=[SimpleNamespace(material=SimpleNamespace(name='fruit_color'))])
        scene = SimpleNamespace(objects=[body('red', 0), body('green', 1), body('yellow', 2)])
        for mapping in ({'ripe': 0, 'non-ripe': 1}, {'ripe': 1, 'non-ripe': 0}):
            groups = self.functions['body_objects'](scene, mapping)
            self.assertEqual(len(groups), 3)
            for group in groups.values():
                expected = mapping['ripe' if group['color'] == 'red' else 'non-ripe']
                self.assertEqual(group['class_id'], expected)
                color = self.functions['fruit_body_color'](group['color'], 0)
                self.assertEqual(color[0] > color[1], group['color'] == 'red')

    def test_class_aspect_reference_tracks_ripe_id(self):
        results = []
        for ripe in (0, 1):
            value = self.functions['viewpoint_cue'](np.zeros((5, 5)), [.1, .1, .6, .6], ripe, 100, 100,
                                                     ripe_class_id=ripe)
            results.append(value)
        self.assertEqual(results[0], results[1])

    def test_renderer_serializes_explicit_mapping_without_relabeling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'fixture.blend'
            scene.write_bytes(b'software-renderer-contract-fixture')
            backend = rendering.BlenderRenderer(sys.executable, scene)
            request = {'descriptor': {'attrs': [0.] * 8, 'source_class': 1},
                       'seed': 0, 'case_id': 'fixture',
                       'tomato_class_map': {'ripe': 1, 'non-ripe': 0}}

            def render(command, **kwargs):
                request_path = Path(command[command.index('--request') + 1])
                payload = json.loads(request_path.read_text(encoding='utf-8'))
                self.assertEqual(payload['tomato_class_map'], request['tomato_class_map'])
                self.assertEqual(payload['descriptor']['source_class'], 1)
                response = {'status': 'ok', 'targets': [{'class_id': 1, 'box': [0., 0., 1., 1.]}],
                            'focal_target_index': 0, 'attrs_measured': [None] * 8}
                for name in ('rgb', 'depth', 'nir', 'instance_mask'):
                    path = request_path.parent / name
                    path.write_bytes(b'fixture')
                    response[name] = str(path)
                response_path = Path(command[command.index('--response') + 1])
                response_path.write_text(json.dumps(response), encoding='utf-8')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with patch.object(rendering.subprocess, 'run', side_effect=render):
                response = backend.render(request, root / 'render')
            self.assertEqual(response['targets'][0]['class_id'], 1)


class ReplayClassMetadataTests(unittest.TestCase):
    def test_derived_names_match_reference_without_names(self):
        from tefed.data import load_manifest
        from tefed.replay import ReplayController
        from tests.test_experiment_protocols import setup_replay
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            engine.validate_manifest_config(config, load_manifest(config['experiment']['manifest']))
            ReplayController(config, ['client1'], 2, 0)
            changed = deepcopy(config)
            changed['model']['class_names'].reverse()
            with self.assertRaisesRegex(ValueError, 'training configuration differs'):
                ReplayController(changed, ['client1'], 2, 0)
            changed = deepcopy(config)
            changed['optimizer']['learning_rate'] *= 2
            with self.assertRaisesRegex(ValueError, 'training configuration differs'):
                ReplayController(changed, ['client1'], 2, 0)

    def test_explicit_reference_order_is_not_ignored(self):
        from tefed.data import load_manifest
        from tefed.experiments import canonical_hash
        from tefed.replay import ReplayController, _training_binding
        from tests.test_experiment_protocols import setup_replay
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            engine.validate_manifest_config(config, load_manifest(config['experiment']['manifest']))
            reversed_config = deepcopy(config)
            reversed_config['model']['class_names'].reverse()
            path = Path(config['replay']['exposure_protocol'])
            protocol = json.loads(path.read_text(encoding='utf-8'))
            protocol.pop('protocol_sha256')
            protocol['training_configuration_sha256'] = _training_binding(reversed_config)
            protocol['protocol_sha256'] = canonical_hash(protocol)
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'training configuration differs'):
                ReplayController(config, ['client1'], 2, 0)


class DistributedClassMetadataTests(unittest.TestCase):
    def test_unbound_and_bound_configs_share_digest_without_image_access(self):
        from tefed.data import load_manifest
        from tefed.software_fixture import create_fixture
        from tefed.transport import shared_config_digest
        with TemporaryDirectory() as directory:
            config = create_fixture(Path(directory))
            derived = deepcopy(config)
            engine.validate_manifest_config(derived, load_manifest(config['experiment']['manifest']))
            self.assertNotIn('class_names', config['model'])
            expected = shared_config_digest(derived)
            with patch('PIL.Image.open', side_effect=AssertionError('Image access during shared digest')):
                with patch('numpy.load', side_effect=AssertionError('Array access during shared digest')):
                    self.assertEqual(shared_config_digest(config), expected)
                    self.assertEqual(shared_config_digest(derived), expected)
            self.assertNotIn('class_names', config['model'])

    def test_coordinator_manifest_needs_no_local_images(self):
        from tefed.software_fixture import create_fixture
        from tefed.transport import shared_config_digest
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'worker')
            expected = shared_config_digest(config)
            path = Path(config['experiment']['manifest'])
            body = json.loads(path.read_text(encoding='utf-8'))
            for record in body['records']:
                for key in ('rgb', 'depth', 'nir'):
                    if key in record:
                        record[key] = str(root / 'not-on-coordinator' / Path(record[key]).name)
            coordinator_manifest = root / 'coordinator.json'
            coordinator_manifest.write_text(json.dumps(body), encoding='utf-8')
            coordinator = deepcopy(config)
            coordinator['experiment']['manifest'] = str(coordinator_manifest)
            self.assertEqual(shared_config_digest(coordinator), expected)

    def test_explicit_label_order_stays_in_shared_digest(self):
        from tefed.software_fixture import create_fixture
        from tefed.transport import shared_config_digest
        with TemporaryDirectory() as directory:
            config = create_fixture(Path(directory))
            expected = shared_config_digest(config)
            declared = deepcopy(config)
            declared['model']['class_names'] = ['non-ripe', 'ripe']
            self.assertNotEqual(shared_config_digest(declared), expected)
            with self.assertRaisesRegex(ValueError, 'Shared class_names'):
                shared_config_digest({**config, 'model': {**config['model'], 'class_names': ['ripe']}})

    def test_worker_rejects_name_conflicts_at_startup_and_task_boundary(self):
        from tefed.software_fixture import create_fixture
        from tefed.transport import PROTOCOL_VERSION, shared_config_digest
        from tefed.worker import WorkerService
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'data')
            conflicting = deepcopy(config)
            conflicting['model']['class_names'] = ['non-ripe', 'ripe']
            with self.assertRaisesRegex(ValueError, 'class-ID order'):
                WorkerService(conflicting, 'client0', 'node0', b'x' * 32, root / 'bad-cache')
            self.assertFalse((root / 'bad-cache').exists())
            service = WorkerService(config, 'client0', 'node0', b'x' * 32, root / 'cache')
            self.assertEqual(service.info['config_sha256_by_method']['tefed'], shared_config_digest(config))
            task = {'protocol_version': PROTOCOL_VERSION, 'model_schema_version': 2,
                    'run_id': 'class-conflict', 'task_id': 'initialize', 'worker_id': 'node0',
                    'client_id': 'client0', 'action': 'initialize', 'round_index': 0, 'method': 'tefed',
                    'seed': 0, 'source_key_sha256': service.info['source_key_sha256'],
                    'package_version': service.info['package_version'],
                    'execution_sha256': service.info['source_sha256'],
                    'partition_sha256': service.info['partition_sha256'],
                    'calibration_sha256': service.info['calibration_sha256'],
                    'config_sha256': shared_config_digest(conflicting)}
            with patch('tefed.client.execute_client_task', side_effect=AssertionError('Unexpected client execution')):
                with self.assertRaisesRegex(ValueError, 'Task configuration differs'):
                    service.execute(task)


if __name__ == '__main__':
    unittest.main()
