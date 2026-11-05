"""Executable controls for the paper's A=0 weights and E=B AND C protocol."""
from copy import deepcopy
import unittest

import torch

from scripts.run_suite import merge
from tefed.baselines import policy_dict
from tefed.engine import _head_weights
from tefed.experiments import (FACTORIAL_DEFINITIONS, FACTORIAL_VERSION,
                               apply_qualification_control, canonical_hash,
                               factorial_plans, validate_factorial_config)
from tefed.losses import WeightedNativeLoss, target_batch
from tefed.model import RoutedYOLO


def configuration(a, b=1, c=1):
    patch = next(patch for _, patch in factorial_plans()
                 if patch['factorial']['factors'] == {'A': a, 'B': b, 'C': c})
    return merge({'experiment': {'test_mode': True}}, patch)


def pseudo_item(a):
    candidates = [
        {'valid': True, 'R': .5, 'D_E': .5, 'posterior': [.8, .2],
         'cues': {'C': .7, 'G': .8}, 'disposition': 'qualified',
         'route': [0, 1, 2], 'weights': [.5, .3, .2]},
        {'valid': True, 'R': .8, 'D_E': .5, 'posterior': [.2, .8],
         'cues': {'C': .8, 'G': .8}, 'disposition': 'qualified',
         'route': [0, 3, 4], 'weights': [.4, .35, .25]},
    ]
    targets = [{'box': [.15, .2, .65, .8], 'class_id': 0},
               {'box': [.7, .1, .9, .3], 'class_id': 1}]
    return {'kind': 'pseudo', 'sample': {'targets': targets},
            'candidates': [apply_qualification_control(candidate, configuration(a))
                           for candidate in candidates]}


class QualificationWeightTests(unittest.TestCase):
    def test_unit_weights_cover_every_target_but_only_the_existing_head_union(self):
        off, on = pseudo_item(0), pseudo_item(1)
        for head in range(5):
            self.assertEqual(_head_weights(off, head), [1., 1.])
        self.assertEqual(_head_weights(off, 5), [0., 0.])
        self.assertEqual(_head_weights(on, 2), [.1, 0.])
        self.assertEqual(_head_weights(on, 3), [0., .8 * .35])
        for after, before in zip(off['candidates'], on['candidates']):
            for field in ('R', 'route', 'weights', 'cues'):
                self.assertEqual(after[field], before[field])
        self.assertNotIn('qualification_control', on['candidates'][0])

    def test_mixed_weight_policies_are_rejected(self):
        mixed = pseudo_item(0)
        mixed['candidates'][1] = pseudo_item(1)['candidates'][1]
        with self.assertRaisesRegex(ValueError, 'mix pseudo-target weight policies'):
            _head_weights(mixed, 0)

    def test_native_A0_matches_unweighted_loss_while_A1_weights_background_and_targets(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        model = RoutedYOLO({'model': {'architecture': 'yolo26n.yaml', 'classes': 2,
                                       'specialist_width': .5}}).train()
        predictions = model(torch.rand(1, 5, 64, 64), (2,))[2]
        off, on = pseudo_item(0), pseudo_item(1)
        targets = off['sample']['targets']
        expected = model.make_criterion(2)(predictions, target_batch(targets, 'cpu'))[0].sum()
        unit_criterion, weighted_criterion = WeightedNativeLoss(model, 2), WeightedNativeLoss(model, 2)
        unit, _ = unit_criterion(predictions, targets, _head_weights(off, 2), 'cpu')
        weighted, _ = weighted_criterion(predictions, targets, _head_weights(on, 2), 'cpu')
        torch.testing.assert_close(unit, expected, rtol=1e-5, atol=1e-5)
        self.assertGreater(float(unit.detach()), float(weighted.detach()))
        for context in unit_criterion.contexts:
            self.assertEqual(context.background_weight, 1.)
            torch.testing.assert_close(context.target_weights, torch.ones(1, 2))
            torch.testing.assert_close(context.anchor_weights, torch.ones_like(context.anchor_weights))
        for context in weighted_criterion.contexts:
            self.assertAlmostEqual(context.background_weight, .05)
            torch.testing.assert_close(context.target_weights, torch.tensor([[.1, 0.]]))
            self.assertTrue((context.anchor_weights == context.background_weight).any())


class EffectiveCorrectionProtocolTests(unittest.TestCase):
    def test_all_three_E0_arms_disable_correction_even_when_C_is_one(self):
        for a in (0, 1):
            for b, c in ((0, 0), (0, 1), (1, 0)):
                with self.subTest(a=a, b=b, c=c):
                    config = configuration(a, b, c)
                    config['experiment']['test_mode'] = False
                    config.update(renderer={'enabled': False}, loss={'beta': 0.})
                    self.assertEqual(validate_factorial_config(config), {'A': a, 'B': b, 'C': c})
                    policy = policy_dict(config['experiment']['method'], 0, 0, config)
                    self.assertTrue(policy['use_experts'])
                    self.assertFalse(policy['use_corrections'])
        self.assertTrue(policy_dict('generic_hc', 0, 0)['use_corrections'])
        self.assertIsNone(validate_factorial_config({'experiment': {'method': 'generic_hc'}}))

    def test_old_operation_version_and_hash_cannot_silently_use_new_semantics(self):
        old_version = configuration(0)
        old_version['factorial']['version'] = 'tefed-factorial-v1'
        stale_hash = configuration(0)
        old_definitions = deepcopy(FACTORIAL_DEFINITIONS)
        old_definitions['A']['off'] = 'Fixed posterior qualification at 0.75; retain routed reliability weights.'
        stale_hash['factorial']['definitions_sha256'] = canonical_hash(old_definitions)
        for config in (old_version, stale_hash):
            with self.assertRaisesRegex(ValueError, 'version or operation definitions'):
                validate_factorial_config(config)
        self.assertEqual(configuration(0)['factorial']['version'], FACTORIAL_VERSION)

    def test_B0C1_cannot_be_overridden_to_generic_hard_case_correction(self):
        config = configuration(1, 0, 1)
        config['experiment']['method'] = 'generic_hc'
        with self.assertRaisesRegex(ValueError, 'declared B/C operations'):
            validate_factorial_config(config)


if __name__ == '__main__':
    unittest.main()
