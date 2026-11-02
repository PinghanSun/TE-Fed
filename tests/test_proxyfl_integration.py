"""ProxyFL native detector execution, uncertain regions, and exact round recovery."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.check_baselines import tiny_architecture
from tefed import engine
from tefed.baselines import METHOD_REGISTRY, policy_dict, validate_baseline_config
from tefed.resume import load_snapshot
from tefed.software_fixture import create_fixture


def configuration(root, threshold):
    cfg = create_fixture(root / 'fixture')
    cfg['experiment'].update(method='proxyfl', save_rounds=[1, 2])
    cfg['model']['architecture'] = tiny_architecture()
    cfg['baselines'] = {'proxyfl': {'pseudo_threshold': threshold, 'proposal_floor': 0.,
        'proxy_weight': 1., 'server_steps': 2, 'server_learning_rate': .005, 'prior_momentum': .99}}
    return cfg


def equal_tree(test, left, right):
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            equal_tree(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_tree(test, a, b)
    elif isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    else:
        test.assertEqual(left, right)


class ProxyFLConfigurationTests(unittest.TestCase):
    def test_registry_and_invalid_method_parameters(self):
        self.assertIn('proxyfl', METHOD_REGISTRY)
        policy = policy_dict('proxyfl', 0, 0)
        self.assertEqual(policy['aggregation'], 'proxyfl')
        self.assertFalse(policy['use_experts'])
        self.assertFalse(policy['use_corrections'])
        self.assertEqual(policy['pseudo_threshold'], .95)
        for key, value in [('server_steps', 0), ('server_steps', 1.5), ('server_steps', True),
                           ('server_learning_rate', 0), ('prior_momentum', 1),
                           ('prior_momentum', -.1), ('proposal_floor', 1.1), ('proxy_weight', float('nan'))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_baseline_config({'baselines': {'proxyfl': {key: value}}})


class ProxyFLNativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_low_confidence_regions_enter_proxy_learning_without_box_targets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = configuration(root, .999999)
            result = engine.run_experiment(cfg, output_dir=root / 'low')
            low_count = 0
            for row in result['rounds']:
                self.assertEqual(row['aggregation']['proxy_tuning']['classifier_count'], 6)
                self.assertEqual(len(row['aggregation']['class_prior']), 2)
                for local in row['clients'].values():
                    stats = local['proxyfl']
                    self.assertGreater(stats['proposal_count'], 0)
                    self.assertGreater(stats['anchor_count'], 0)
                    self.assertEqual(local['presentations']['pseudo'], 0)
                    self.assertTrue(all(np.isfinite(value) for value in stats['losses']))
                    low_count += stats['low_count']
            self.assertGreater(low_count, 0)
            model, _, saved = engine.load_checkpoint(result['checkpoint'])
            self.assertEqual(len(model.heads), 1)
            self.assertEqual(saved['config']['experiment']['method'], 'proxyfl')
            self.assertEqual(set(result['metrics']), {'participating', 'unseen', 'new-client', 'hard'})

    def test_prior_and_model_state_match_after_interruption_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = configuration(root, 0.)
            full = engine.run_experiment(cfg, output_dir=root / 'full')
            partial = root / 'partial'
            original = engine.train_local

            def interrupt(*args, **kwargs):
                if kwargs['round_id'] == 1:
                    raise RuntimeError('interrupt second round')
                return original(*args, **kwargs)

            with patch('tefed.engine.train_local', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt second round'):
                    engine.run_experiment(cfg, output_dir=partial)
            resumed = engine.run_experiment(cfg, output_dir=partial, resume=partial / 'resume.pt')
            left = load_snapshot(Path(full['output_dir']) / 'resume.pt')
            right = load_snapshot(partial / 'resume.pt')
            for key in ('model', 'router', 'states', 'aggregation_state', 'client_runtime', 'rng'):
                equal_tree(self, left[key], right[key])
            self.assertEqual(full['metrics'], resumed['metrics'])
            prior = right['aggregation_state']['class_prior']
            self.assertAlmostEqual(sum(prior), 1.)
            self.assertNotEqual(prior, [.5, .5])
            self.assertEqual(prior, resumed['rounds'][-1]['aggregation']['class_prior'])
            checkpoint = partial / 'checkpoints/round_000002.pt'
            self.assertTrue(checkpoint.is_file())
            source = partial / 'rounds.jsonl'
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            self.assertEqual([row['round_index'] for row in rows], [0, 1])
