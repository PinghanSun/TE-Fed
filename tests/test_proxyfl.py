"""ProxyFL native detector mechanisms and numerical loss contracts."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import tefed.proxyfl as proxyfl
    from tefed.model import RoutedYOLO
    from tefed.proxyfl import class_proxy_parameters, optimize_global_proxies, pool_contrastive_loss, proxy_objective
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires native training dependencies')
class PoolObjectiveTests(unittest.TestCase):
    def test_disjoint_negative_pools_and_unrenormalized_low_positive(self):
        features = torch.tensor([[1., 0], [.5, math.sqrt(.75)], [.6, .8]], requires_grad=True)
        proxy = torch.tensor([[2., 0.]], requires_grad=True)
        positive = .6 * F.normalize(proxy, dim=1)
        targets = torch.cat([positive, torch.zeros_like(positive), positive])
        pools = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
        mask = torch.tensor([True, False, True])
        loss, details = pool_contrastive_loss(features, targets, pools, mask, return_details=True)
        first = math.log1p(math.exp(.5 - .6))
        second = math.log1p(math.exp(.6 * .5 + .8 * math.sqrt(.75) - .6 * .6))
        self.assertAlmostEqual(float(loss.detach()), (first + second) / 2, places=6)
        self.assertEqual(details, {'anchor_count': 2, 'negative_pair_count': 2})
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertGreater(float(proxy.grad.abs().sum()), 0)
        # Annotated samples participate as negatives and never as loss anchors.
        self.assertGreater(float(features.grad[1].abs().sum()), 0)

    def test_negative_cosines_and_overlapping_pools_are_excluded(self):
        features = torch.tensor([[1., 0], [-.5, math.sqrt(.75)], [1., 0]], requires_grad=True)
        targets = torch.tensor([[1., 0], [0., 0], [0., 0]])
        pools = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
        loss, details = pool_contrastive_loss(features, targets, pools,
            torch.tensor([True, False, False]), return_details=True)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(details['negative_pair_count'], 0)
        loss.backward()
        self.assertEqual(float(features.grad.abs().sum()), 0.)

    def test_empty_possible_set_is_not_an_anchor(self):
        features = torch.tensor([[1., 0]], requires_grad=True)
        loss, details = pool_contrastive_loss(features, features * 0,
            torch.zeros(1, 2, dtype=torch.bool), torch.tensor([True]), return_details=True)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(details['anchor_count'], 0)
        self.assertEqual(details['negative_pair_count'], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_empty_possible_set_is_omitted_from_anchors_and_negatives(self):
        features = torch.tensor([[1., 0], [.5, math.sqrt(.75)], [1., 0]], requires_grad=True)
        targets = torch.tensor([[1., 0], [0., 0], [0., 0]], requires_grad=True)
        pools = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
        loss, details = pool_contrastive_loss(features, targets, pools,
            torch.tensor([True, False, True]), return_details=True)
        self.assertAlmostEqual(float(loss.detach()), math.log1p(math.exp(.5 - 1.)), places=6)
        self.assertEqual(details, {'anchor_count': 1, 'negative_pair_count': 1})
        loss.backward()
        # The labeled row still provides the negative; the empty row has no role.
        self.assertGreater(float(features.grad[1].abs().sum()), 0.)
        self.assertEqual(float(features.grad[2].abs().sum()), 0.)
        self.assertEqual(float(targets.grad[2].abs().sum()), 0.)


@unittest.skipUnless(AVAILABLE, 'Requires native training dependencies')
class NativeProxyFLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        cls.model = RoutedYOLO({'model': {'architecture': 'yolo26n.yaml', 'classes': 2,
                                        'specialists': 0, 'contrastive_dim': 8}})

    @staticmethod
    def sample(identity, targets):
        rgb = np.random.default_rng(34).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        return {'id': identity, 'rgb': rgb, 'depth': None, 'nir': None, 'targets': targets}

    @staticmethod
    def strong(sample, seed):
        return {**sample, 'rgb': (sample['rgb'].astype(np.float32) * .8).astype(np.uint8)}

    def test_server_matches_euclidean_direction_margin_and_updates_both_branches(self):
        model = deepcopy(self.model)
        parameters = class_proxy_parameters(model)
        self.assertEqual(len(parameters), 6)
        self.assertTrue(any('one2one_cv3' in name for name in parameters))
        self.assertTrue(any('.cv3.' in name for name in parameters))
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        clients = []
        for sign in (-1, 1):
            state = model.export_state()
            for name in parameters:
                suffix = name.split('.', 2)[2]
                value = state['heads']['0'][suffix]
                value[0] += sign * .03
                value[1] += sign * .01
            clients.append(state)
        expected = {}
        for name, parameter in parameters.items():
            trial = parameter.detach().clone().requires_grad_()
            anchors = torch.cat([torch.as_tensor(state['heads']['0'][name.split('.', 2)[2]]).flatten(1) for state in clients])
            targets = torch.tensor([0, 1, 0, 1])
            distances = torch.cdist(trial.detach().flatten(1), trial.detach().flatten(1))
            distances.fill_diagonal_(float('inf'))
            margin = distances.amin(1).amax().clamp(max=100)
            loss = F.cross_entropy(-torch.cdist(anchors, trial.flatten(1)) - margin * F.one_hot(targets, 2), targets)
            gradient, = torch.autograd.grad(loss, trial)
            expected[name] = trial.detach() - .005 * gradient
        stats = optimize_global_proxies(model, clients, steps=1, learning_rate=.005)
        self.assertEqual(stats['classifier_count'], 6)
        self.assertEqual(stats['client_count'], 2)
        self.assertTrue(math.isfinite(stats['initial_loss']))
        self.assertLess(stats['final_loss'], stats['initial_loss'])
        for name, value in model.state_dict().items():
            if name in parameters:
                torch.testing.assert_close(value, expected[name])
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(value.grad is None for value in parameters.values()))

    def test_local_labeled_high_low_pools_reach_native_trunk_and_class_proxies(self):
        model = deepcopy(self.model)
        model.train()
        normalizers = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        normalizers[0].eval()
        bn_before = [(module.training, deepcopy(module.state_dict())) for module in normalizers]
        hooks_before = {name: len(module._forward_pre_hooks) for name, module in model.named_modules()}
        labeled = self.sample('labeled', [{'class_id': 1, 'box': [.1, .1, .5, .5]}])
        unlabeled = self.sample('unlabeled', [])
        batch = [{'kind': 'labeled', 'sample': labeled}, {'kind': 'pseudo', 'sample': unlabeled,
                  'proxy_candidates': [{'index': 0, 'posterior': [.99, .01], 'score': .9},
                                       {'index': 1, 'posterior': [.6, .4], 'score': .8},
                                       {'index': 2, 'posterior': [.2, .8], 'score': .01}]}]
        loss, stats = proxy_objective(model, batch, {'model': {}}, [.5, .5],
            strong_view=self.strong, round_id=0, epoch=0)
        self.assertEqual(stats['labeled_count'], 1)
        self.assertEqual(stats['high_count'], 1)
        self.assertEqual(stats['low_count'], 1)
        self.assertEqual(stats['omitted_empty_count'], 0)
        self.assertEqual(stats['proposal_count'], 2)
        self.assertEqual(stats['anchor_count'], 4)
        self.assertEqual(stats['class_counts'], [1, 1])
        self.assertGreater(stats['negative_pair_count'], 0)
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        class_gradients = [parameter.grad for name, parameter in class_proxy_parameters(model).items() if '.cv3.' in name]
        self.assertTrue(any(value is not None and float(value.abs().sum()) > 0 for value in class_gradients))
        self.assertTrue(any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in model.trunk.parameters()))
        for module, (training, state) in zip(normalizers, bn_before):
            self.assertEqual(module.training, training)
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertEqual({name: len(module._forward_pre_hooks) for name, module in model.named_modules()}, hooks_before)

    def test_local_only_confidence_uses_global_class_and_counts_each_proposal_once(self):
        model = deepcopy(self.model)
        with torch.no_grad():
            for tower in model.heads[0].detect.cv3:
                tower[-1].weight.zero_()
                tower[-1].bias.copy_(torch.tensor([-10., 10.]))
        item = {'kind': 'pseudo', 'sample': self.sample('unlabeled', []),
                'proxy_candidates': [{'index': 0, 'posterior': [.6, .4], 'score': .7}]}
        _, stats = proxy_objective(model, [item], {'model': {}}, [.5, .5],
            strong_view=self.strong, round_id=0, epoch=0)
        self.assertEqual(stats['high_count'], 1)
        self.assertEqual(stats['class_counts'], [1, 0])
        self.assertEqual(stats['anchor_count'], 2)

    def test_adding_empty_low_pool_changes_neither_icpl_nor_labeled_negatives(self):
        labeled = self.sample('labeled', [{'class_id': 1, 'box': [.1, .1, .5, .5]}])
        candidates = [{'index': 0, 'posterior': [.99, .01], 'score': .9},
                      {'index': 1, 'posterior': [.6, .4], 'score': .8}]
        original_forward = proxyfl._class_feature_forward
        def uniform_local(*args):
            features, logits = original_forward(*args)
            return features, torch.zeros_like(logits)
        results = []
        with patch.object(proxyfl, '_class_feature_forward', side_effect=uniform_local):
            for extra in ([], [{'index': 2, 'posterior': [.5, .5], 'score': .7}]):
                model = deepcopy(self.model)
                batch = [{'kind': 'labeled', 'sample': labeled},
                         {'kind': 'pseudo', 'sample': self.sample('unlabeled', []),
                          'proxy_candidates': candidates + extra}]
                loss, stats = proxy_objective(model, batch, {'model': {}}, [.5, .5],
                    strong_view=self.strong, round_id=0, epoch=0)
                loss.backward()
                gradients = {name: None if p.grad is None else p.grad.clone()
                             for name, p in model.named_parameters()}
                results.append((loss.detach(), stats, gradients))
        baseline, augmented = results
        torch.testing.assert_close(baseline[0], augmented[0], rtol=0, atol=0)
        for field in ('labeled_count', 'high_count', 'low_count', 'anchor_count',
                      'negative_pair_count', 'class_counts'):
            self.assertEqual(baseline[1][field], augmented[1][field], field)
        self.assertGreater(augmented[1]['negative_pair_count'], 0)
        self.assertEqual(augmented[1]['anchor_count'], 4)
        self.assertEqual(augmented[1]['proposal_count'], 3)
        self.assertEqual(augmented[1]['omitted_empty_count'], 1)
        for name, gradient in baseline[2].items():
            if gradient is None:
                self.assertIsNone(augmented[2][name], name)
            else:
                torch.testing.assert_close(gradient, augmented[2][name], rtol=0, atol=0)

    def test_all_empty_low_pools_are_omitted_but_high_confidence_stays_singleton(self):
        model = deepcopy(self.model)
        original_forward = proxyfl._class_feature_forward
        def uniform_local(*args):
            features, logits = original_forward(*args)
            return features, torch.zeros_like(logits)
        item = {'kind': 'pseudo', 'sample': self.sample('unlabeled', []),
                'proxy_candidates': [{'index': 0, 'posterior': [.5, .5], 'score': .7}]}
        with patch.object(proxyfl, '_class_feature_forward', side_effect=uniform_local):
            loss, stats = proxy_objective(model, [item], {'model': {}}, [.5, .5],
                strong_view=self.strong, round_id=0, epoch=0)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(stats['omitted_empty_count'], 1)
        self.assertEqual(stats['low_count'], 0)
        self.assertEqual(stats['anchor_count'], 0)
        self.assertEqual(stats['negative_pair_count'], 0)
        self.assertEqual(stats['class_counts'], [0, 0])
        loss.backward()
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in model.parameters()))
        _, high = proxy_objective(deepcopy(self.model), [item],
            {'model': {}, 'baselines': {'proxyfl': {'pseudo_threshold': .4}}}, [.5, .5],
            strong_view=self.strong, round_id=0, epoch=0)
        self.assertEqual(high['high_count'], 1)
        self.assertEqual(high['low_count'], 0)
        self.assertEqual(high['omitted_empty_count'], 0)
        self.assertEqual(high['anchor_count'], 2)
        self.assertEqual(high['class_counts'], [1, 0])

    def test_labeled_only_prior_counts_do_not_require_unlabeled_anchors(self):
        item = {'kind': 'labeled', 'sample': self.sample('labeled', [{'class_id': 1, 'box': [.1, .1, .5, .5]}])}
        loss, stats = proxy_objective(deepcopy(self.model), [item], {'model': {}}, None,
            strong_view=self.strong, round_id=0, epoch=0)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(stats['class_counts'], [0, 1])
        self.assertEqual(stats['anchor_count'], 0)

    def test_proxy_forward_restores_batchnorm_and_hooks_on_failure(self):
        model = deepcopy(self.model)
        flags = {name: module.training for name, module in model.named_modules()}
        hooks = {name: len(module._forward_pre_hooks) for name, module in model.named_modules()}
        item = {'kind': 'pseudo', 'sample': self.sample('unlabeled', []),
                'proxy_candidates': [{'index': 0, 'posterior': [.6, .4], 'score': .7}]}
        def failure(sample, seed):
            raise RuntimeError('augmentation failure')
        with self.assertRaisesRegex(RuntimeError, 'augmentation failure'):
            proxy_objective(model, [item], {'model': {}}, [.5, .5], strong_view=failure, round_id=0, epoch=0)
        self.assertEqual({name: module.training for name, module in model.named_modules()}, flags)
        self.assertEqual({name: len(module._forward_pre_hooks) for name, module in model.named_modules()}, hooks)


if __name__ == '__main__':
    unittest.main()
