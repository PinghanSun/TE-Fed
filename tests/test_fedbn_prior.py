from copy import deepcopy
import io
import importlib.util
import random
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from tefed.fedbn import build_validation_bn_prior, validate_bn_prior, apply_bn_prior, merge_validation_bn_priors


@unittest.skipUnless(torch is not None, 'Requires native training dependencies')
class FedBNPriorTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.BatchNorm1d(2)).double()
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2, dtype=torch.float64))
            model[1].weight.copy_(torch.tensor([2., 3.]))
            model[1].bias.copy_(torch.tensor([-.5, .5]))
            model[1].running_mean.copy_(torch.tensor([.3, .7]))
            model[1].running_var.copy_(torch.tensor([1.5, 2.5]))
        return model

    def data(self):
        return torch.tensor([[1., 5.], [3., 8.], [5., 12.], [7., 18.], [11., 20.], [12., 21.]], dtype=torch.float64)

    def ids(self):
        return [f'validation-{i}' for i in range(6)]

    def test_known_pooled_moments_affine_and_safe_checkpoint(self):
        model, data = self.model(), self.data()
        prior = build_validation_bn_prior(model, [data[:2], data[2:]], validation_ids=self.ids(),
                                         binding={'initial_checkpoint_sha256': 'a' * 64})
        torch.testing.assert_close(prior['state']['1.running_mean'], data.mean(0))
        torch.testing.assert_close(prior['state']['1.running_var'], data.var(0, unbiased=True))
        self.assertEqual(prior['state']['1.num_batches_tracked'].item(), 1)
        self.assertEqual(prior['protocol']['activation_counts'], {'1': 6})
        for field in ['weight', 'bias']:
            torch.testing.assert_close(prior['state']['1.' + field], model[1].state_dict()[field])
        self.assertTrue(all(value.device.type == 'cpu' for value in prior['state'].values()))
        buffer = io.BytesIO()
        torch.save(prior, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        validate_bn_prior(model, loaded, validation_ids=list(reversed(self.ids())),
                          binding={'initial_checkpoint_sha256': 'a' * 64})

    def test_partition_and_order_invariance_with_multiple_bn_layers(self):
        model = torch.nn.Sequential(self.model(), torch.nn.ReLU(), torch.nn.BatchNorm1d(2)).double()
        data = self.data()
        whole = build_validation_bn_prior(model, [data], validation_ids=self.ids())
        parts = build_validation_bn_prior(model, [data[5:], data[2:5], data[:2]], validation_ids=list(reversed(self.ids())))
        for key in whole['state']:
            torch.testing.assert_close(whole['state'][key], parts['state'][key], rtol=1e-12, atol=1e-12)
        self.assertEqual(whole['protocol']['validation_ids_sha256'], parts['protocol']['validation_ids_sha256'])
        with torch.no_grad():
            upstream = model[0].eval()(data).relu()
        torch.testing.assert_close(whole['state']['2.running_mean'], upstream.mean(0))
        torch.testing.assert_close(whole['state']['2.running_var'], upstream.var(0, unbiased=True))

    def test_spatial_channel_population_has_unbiased_variance(self):
        model = torch.nn.BatchNorm2d(2, affine=False).double()
        data = torch.arange(48, dtype=torch.float64).reshape(3, 2, 2, 4)
        prior = build_validation_bn_prior(model, [data[:1], data[1:]], validation_ids=['v1', 'v2', 'v3'])
        torch.testing.assert_close(prior['state']['running_mean'], data.mean((0, 2, 3)))
        torch.testing.assert_close(prior['state']['running_var'], data.var((0, 2, 3), unbiased=True))
        self.assertEqual(prior['protocol']['activation_counts'], {'': 24})

    def test_preserves_weights_gradients_modes_requires_grad_and_all_rng(self):
        model, data = self.model().train(), self.data()
        model[1].eval()
        model[0].weight.requires_grad_(False)
        model[1].weight.grad = torch.tensor([7., 8.], dtype=torch.float64)
        gradient = model[1].weight.grad
        before = deepcopy(model.state_dict())
        modes = [layer.training for layer in model.modules()]
        flags = [p.requires_grad for p in model.parameters()]
        random.seed(45); np.random.seed(46); torch.manual_seed(47)
        rng = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        def forward(current, batch):
            self.assertFalse(current.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not p.requires_grad for p in current.parameters()))
            random.random(); np.random.random(); torch.rand(3)
            return current(batch)
        build_validation_bn_prior(model, [data], validation_ids=self.ids(), forward=forward)
        for key in before:
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
        self.assertEqual([layer.training for layer in model.modules()], modes)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertIs(model[1].weight.grad, gradient)
        torch.testing.assert_close(gradient, torch.tensor([7., 8.], dtype=torch.float64))
        self.assertEqual(random.getstate(), rng[0])
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], rng[1][0]); np.testing.assert_array_equal(current_numpy[1], rng[1][1])
        self.assertEqual(current_numpy[2:], rng[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))

    def test_restores_state_and_removes_hooks_after_failed_forward(self):
        model, data = self.model().train(), self.data()
        before = deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        def fail(current, batch):
            current(batch)
            torch.rand(4)
            with torch.no_grad():
                current[0].weight.add_(3)
            raise RuntimeError('deliberate forward failure')
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            build_validation_bn_prior(model, [data], validation_ids=self.ids(), forward=fail)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(model[1]._forward_pre_hooks)
        for key in before:
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)

    def test_unvisited_bn_is_explicit_and_never_filled_with_global_stats(self):
        model = torch.nn.ModuleDict({'used': self.model(), 'unused': torch.nn.BatchNorm1d(2).double()})
        forward = lambda current, batch: current['used'](batch)
        with self.assertRaisesRegex(ValueError, 'did not visit'):
            build_validation_bn_prior(model, [self.data()], validation_ids=self.ids(), forward=forward)
        prior = build_validation_bn_prior(model, [self.data()], validation_ids=self.ids(), forward=forward, require_all=False)
        self.assertEqual(prior['protocol']['unvisited_modules'], ['unused'])
        self.assertFalse(any(key.startswith('unused.') for key in prior['state']))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            apply_bn_prior(model, prior)
        validate_bn_prior(model, prior, require_all=False)

    def test_rejects_wrong_splits_identity_counts_and_nonfinite_inputs(self):
        model, data = self.model(), self.data()
        for split in ['test', 'hard', 'unlabeled', 'labeled']:
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'validation'):
                build_validation_bn_prior(model, [data], validation_ids=self.ids(), source_split=split)
        for ids, batches in [(['v', 'v'], [data[:2]]), (['v'], [data]), (self.ids(), [])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_validation_bn_prior(model, batches, validation_ids=ids)
        data[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_validation_bn_prior(model, [data], validation_ids=self.ids())
        with self.assertRaisesRegex(ValueError, 'fewer than two'):
            build_validation_bn_prior(model, [torch.ones(1, 2, dtype=torch.float64)], validation_ids=['one'])
        with self.assertRaisesRegex(ValueError, 'fixed running'):
            build_validation_bn_prior(torch.nn.BatchNorm1d(2, track_running_stats=False),
                                      [torch.ones(2, 2)], validation_ids=['v1', 'v2'])

    def test_strict_tensor_schema_binding_and_bn_only_install(self):
        model = self.model()
        prior = build_validation_bn_prior(model, [self.data()], validation_ids=self.ids())
        for change in ['missing', 'extra', 'shape', 'dtype', 'nan', 'negative_variance', 'wrong_ids']:
            invalid = deepcopy(prior)
            if change == 'missing': del invalid['state']['1.running_mean']
            elif change == 'extra': invalid['state']['0.weight'] = model[0].weight.detach().clone()
            elif change == 'shape': invalid['state']['1.running_mean'] = torch.zeros(3, dtype=torch.float64)
            elif change == 'dtype': invalid['state']['1.running_mean'] = torch.zeros(2, dtype=torch.float32)
            elif change == 'nan': invalid['state']['1.weight'][0] = float('nan')
            elif change == 'negative_variance': invalid['state']['1.running_var'][0] = -1
            else: invalid['protocol']['validation_ids'] = ['other']
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_bn_prior(model, invalid)
        with self.assertRaisesRegex(ValueError, 'different validation'):
            validate_bn_prior(model, prior, validation_ids=['other'])
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            validate_bn_prior(model, prior, binding={'initial_checkpoint_sha256': 'b' * 64})
        before = model[0].weight.detach().clone()
        model.train()
        apply_bn_prior(model, prior)
        torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)
        torch.testing.assert_close(model[1].running_mean, self.data().mean(0))
        self.assertTrue(model.training)

    def test_worker_moments_merge_matches_full_pool_with_unequal_counts(self):
        model, data = self.model(), self.data()
        binding = {'initial_checkpoint_sha256': 'a' * 64}
        left = build_validation_bn_prior(model, [data[:2]], validation_ids=self.ids()[:2], binding=binding)
        right = build_validation_bn_prior(model, [data[2:5], data[5:]], validation_ids=self.ids()[2:], binding=binding)
        whole = build_validation_bn_prior(model, [data], validation_ids=self.ids(), binding=binding)
        original = deepcopy(model.state_dict())
        merged = merge_validation_bn_priors(model, [right, left], binding=binding)
        reversed_merge = merge_validation_bn_priors(model, [left, right], binding=binding)
        self.assertEqual(merged['protocol']['input_count'], 6)
        self.assertEqual(merged['protocol']['activation_counts'], {'1': 6})
        self.assertEqual(merged['protocol']['validation_ids_sha256'], whole['protocol']['validation_ids_sha256'])
        for key in whole['state']:
            torch.testing.assert_close(merged['state'][key], whole['state'][key], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(merged['state'][key], reversed_merge['state'][key], rtol=0, atol=0)
        torch.testing.assert_close(merged['moments']['1']['m2'], data.var(0, unbiased=False) * 6)
        for key in original:
            torch.testing.assert_close(model.state_dict()[key], original[key], rtol=0, atol=0)

    def test_merge_rejects_overlapping_ids_bad_moments_and_different_initial_model(self):
        model, data = self.model(), self.data()
        left = build_validation_bn_prior(model, [data[:2]], validation_ids=self.ids()[:2],
                                        binding={'initial_checkpoint_sha256': 'a' * 64})
        right = build_validation_bn_prior(model, [data[2:]], validation_ids=self.ids()[2:],
                                         binding={'initial_checkpoint_sha256': 'a' * 64})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            merge_validation_bn_priors(model, [left, left])
        invalid = deepcopy(right)
        invalid['moments']['1']['m2'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            merge_validation_bn_priors(model, [left, invalid])
        invalid = deepcopy(right)
        invalid['moments']['1']['mean'] += 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            merge_validation_bn_priors(model, [left, invalid])
        invalid = deepcopy(right)
        invalid['state']['1.weight'] += 1
        with self.assertRaisesRegex(ValueError, 'affine differs'):
            merge_validation_bn_priors(model, [left, invalid])
        invalid = deepcopy(right)
        invalid['protocol']['binding']['initial_checkpoint_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'different initial'):
            merge_validation_bn_priors(model, [left, invalid])
        with self.assertRaisesRegex(ValueError, 'initial checkpoint binding differs'):
            merge_validation_bn_priors(model, [left, right], binding={'initial_checkpoint_sha256': 'b' * 64})

    def test_merged_source_metadata_requires_schema_hash_and_parent_bindings(self):
        model, data = self.model(), self.data()
        binding = {'initial_model_sha256': 'a' * 64, 'preprocessing_sha256': 'b' * 64}
        left = build_validation_bn_prior(model, [data[:2]], validation_ids=self.ids()[:2], binding=binding)
        right = build_validation_bn_prior(model, [data[2:]], validation_ids=self.ids()[2:], binding=binding)
        merged = merge_validation_bn_priors(model, [left, right], binding=binding)
        validate_bn_prior(model, merged)
        for change in ['empty', 'wrong_binding', 'missing_binding', 'invalid_hash', 'extra_source_key',
                       'missing_sources', 'missing_aggregation', 'duplicate_source', 'wrong_aggregation']:
            invalid = deepcopy(merged)
            protocol = invalid['protocol']
            if change == 'empty': protocol['source_priors'] = []
            elif change == 'wrong_binding': protocol['source_priors'][0]['binding']['initial_model_sha256'] = 'c' * 64
            elif change == 'missing_binding': del protocol['source_priors'][0]['binding']['preprocessing_sha256']
            elif change == 'invalid_hash': protocol['source_priors'][0]['validation_ids_sha256'] = 'not-a-sha256'
            elif change == 'extra_source_key': protocol['source_priors'][0]['extra'] = 1
            elif change == 'missing_sources': del protocol['source_priors']
            elif change == 'missing_aggregation': del protocol['aggregation']
            elif change == 'duplicate_source': protocol['source_priors'].append(deepcopy(protocol['source_priors'][0]))
            else: protocol['aggregation'] = 'unknown'
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '(?i)(merged|binding)'):
                validate_bn_prior(model, invalid)

    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Requires native detector dependencies')
    def test_native_general_detector_forward_visits_all_bn_branches(self):
        from tefed.model import RoutedYOLO
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            model = RoutedYOLO({'model': {'architecture': 'yolo26n.yaml', 'classes': 2, 'specialists': 0}})
            before = deepcopy(model.state_dict())
            prior = build_validation_bn_prior(model, [torch.zeros(2, 5, 64, 64)],
                validation_ids=['v1', 'v2'], forward=lambda current, batch: current(batch, range(len(current.heads))))
            self.assertTrue(prior['protocol']['visited_modules'])
            self.assertEqual(prior['protocol']['unvisited_modules'], [])
            self.assertEqual(len(prior['state']), 5 * len(prior['protocol']['visited_modules']))
            for key in before:
                torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
