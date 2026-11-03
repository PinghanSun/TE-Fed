"""Primary manuscript settings and the independent short runtime fixture."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from tefed.baselines import optimizer_learning_rate, policy_dict, validate_baseline_config

NATIVE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('ultralytics') is not None


class PaperSettingsTests(unittest.TestCase):
    def configs(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        return [tomllib.loads((root / name).read_text(encoding='utf-8'))
                for name in ('tomato.toml', 'dronevehicle.toml')]

    def test_primary_configuration_matches_shared_manuscript_settings(self):
        for config in self.configs():
            validate_baseline_config(config)
            exp, model, opt = config['experiment'], config['model'], config['optimizer']
            self.assertEqual((exp['rounds'], exp['schedule_rounds'], exp['local_epochs'], exp['batch_size']), (300, 300, 2, 4))
            self.assertEqual(exp['seeds'], [0, 1, 2, 3, 4])
            self.assertEqual((model['architecture'], model['image_size'], model['specialists']), ('yolo26s.yaml', 640, 5))
            self.assertEqual((opt['learning_rate'], opt['final_learning_rate'], opt['momentum'], opt['weight_decay']), (.01, .001, .937, .0005))
            self.assertEqual((config['decision']['render_budget'], config['decision']['reuse_cap']), (1152, 6))
            for method in ('consistent_teacher_fedawa', 'fedsto', 'fsslod'):
                self.assertTrue(policy_dict(method, 0, 0, config)['reset_teacher_each_round'])
            self.assertEqual(policy_dict('fedavg', 0, 0, config)['alignment_scope'], 'labeled')
            self.assertEqual(policy_dict('tefed', 0, 0, config)['alignment_scope'], 'all')

    def test_continuation_preserves_primary_lr_and_ema_prefix_and_endpoints(self):
        short = self.configs()[0]
        long = deepcopy(short)
        long['experiment']['rounds'] = 1000
        for r in (0, 1, 2, 3, 49, 149, 299):
            self.assertEqual(optimizer_learning_rate(short, r), optimizer_learning_rate(long, r))
            for e in (0, 1):
                self.assertEqual(policy_dict('fsslod', r, e, short)['teacher_decay'],
                                 policy_dict('fsslod', r, e, long)['teacher_decay'])
        self.assertAlmostEqual(policy_dict('fsslod', 299, 1, long)['teacher_decay'], .99 + .009 * 599 / 599)
        for r in (300, 499, 999):
            self.assertEqual(optimizer_learning_rate(long, r), .001)
            self.assertAlmostEqual(policy_dict('fsslod', r, 0, long)['teacher_decay'], .999)

    def test_bad_schedule_and_teacher_reset_types_are_rejected(self):
        for value in (0, -1, True, '300'):
            with self.assertRaises(ValueError):
                optimizer_learning_rate({'experiment': {'schedule_rounds': value}}, 0)
        for value in (0, 1, None, 'true'):
            with self.assertRaises(ValueError):
                validate_baseline_config({'baselines': {'fsslod': {'reset_teacher_each_round': value}}})

    @unittest.skipUnless(NATIVE, 'Requires native training dependencies')
    def test_conventional_contrast_uses_only_labeled_regions_and_skips_rgb_only(self):
        import torch
        from tefed import engine
        from tefed.model import RoutedYOLO
        torch.set_num_threads(2)
        torch.manual_seed(9)
        config = {'model': {'architecture': 'yolo26n.yaml', 'classes': 2, 'specialists': 0},
                  'experiment': {'rounds': 1, 'local_epochs': 1, 'batch_size': 3},
                  'optimizer': {'amp': False, 'warmup_rounds': 0, 'learning_rate': .001},
                  'loss': {'rho': .1}}
        rng = np.random.default_rng(8)
        items = []
        for i in range(3):
            sample = {'id': str(i), 'rgb': rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                      'depth': np.full((64, 64), 2000., np.float32), 'depth_scale': 65535.,
                      'nir': rng.integers(0, 256, (64, 64), dtype=np.uint8),
                      'targets': [{'box': [.1, .1, .8, .8], 'class_id': i % 2}]}
            items.append({'kind': 'labeled' if i < 2 else 'pseudo', 'sample': sample,
                          'route_weights': {0: 1.}, 'candidates': [{'R': .8, 'route': [0], 'weights': [1.]}]})
        model = RoutedYOLO(config)
        with patch('tefed.engine._crop_embeddings', wraps=engine._crop_embeddings) as crop:
            ledger = engine.train_local(model, items, config, method='confidence')
        self.assertEqual(ledger['contrastive_batches'], 1)
        self.assertEqual({s['id'] for s in crop.call_args.args[2]}, {'0', '1'})
        config['model']['detector_rgb_only'] = True
        ledger = engine.train_local(model, items, config, method='confidence')
        self.assertEqual(ledger['contrastive_batches'], 0)

    @unittest.skipUnless(NATIVE, 'Requires native training dependencies')
    def test_reset_teachers_use_broadcast_for_assessment(self):
        import torch
        from tefed.client import execute_client_round, module_state
        from tefed.model import RoutedYOLO, Router
        from tefed.software_fixture import create_fixture
        from tefed.data import load_manifest
        torch.set_num_threads(2)
        for method in ('consistent_teacher_fedawa', 'fsslod'):
            with self.subTest(method=method), TemporaryDirectory() as directory:
                config = create_fixture(Path(directory))
                config['experiment'].update(method=method, local_epochs=1)
                config['model']['specialists'] = 0
                config['baselines'] = {method: {'reset_teacher_each_round': True}}
                model, router = RoutedYOLO(config), Router()
                old = module_state(model)
                old = {key: value + 10 if np.issubdtype(value.dtype, np.floating) else value for key, value in old.items()}
                manifest = load_manifest(config['experiment']['manifest'])
                calibration = json.loads(Path(config['experiment']['calibration']).read_text())
                with patch('tefed.engine.assess_frame', return_value={'candidates': []}) as assess:
                    result = execute_client_round(model, router, manifest, calibration['state_prior'], calibration, config,
                        client_id='client0', round_index=1, run_secret=b'fixture', persistent_state={'teacher': old})
                self.assertIs(assess.call_args.args[0], model)
                self.assertGreater(result['ledger']['teacher_updates'], 0)
                first = next(iter(dict(model.named_parameters())))
                np.testing.assert_allclose(result['persistent_state']['teacher'][first], module_state(model)[first], atol=.01)


if __name__ == '__main__':
    unittest.main()
