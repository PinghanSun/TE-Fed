"""Zero fixed fidelity mass is excluded before any expert verification work."""
from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from tefed.analysis import case_event_proportions
from tefed.server import construct_cases
from tests.test_server import (FakeBank, FakeRenderer, FakeRouter, STATE,
    calibration, config_for, services, source_descriptor)


class EmptyFidelityPrecheckTests(unittest.TestCase):
    def test_empty_mask_is_rendered_but_never_enters_expert_verification(self):
        for mode in ('stored', 'current'):
            with self.subTest(mode=mode):
                descriptor = source_descriptor()
                descriptor.update(rule_flags=[0] * 5, mask=[0] * 8)
                self.check_preverification_exclusion(descriptor, mode)

    def test_generic_prior_empty_mask_uses_rendering_descriptor_for_precheck(self):
        # The source mask is nonempty, but the frozen generic prior disables it.
        descriptor = source_descriptor()
        cal = calibration()
        cal['generic_priors'] = {'illum': {'attrs': [0.] * 8,
                                         'availability': [1, 1, 1, 0, 1, 1, 1, 1]}}
        self.check_preverification_exclusion(descriptor, 'current', cal, 'generic_hc')

    def check_preverification_exclusion(self, descriptor, mode, cal=None, method='tefed'):
        original = deepcopy(descriptor)
        renderer = FakeRenderer()
        service = services(renderer)
        service['load_case'] = Mock(side_effect=AssertionError('load before precheck'))
        config = config_for(service, mode)
        config['experiment']['method'] = method
        with TemporaryDirectory() as directory, \
                patch('tefed.server._assess', side_effect=AssertionError('expert assessment')) as assess, \
                patch('tefed.server._predict_all', side_effect=AssertionError('expert inference')) as predict, \
                patch('tefed.server.core.verification_posterior', side_effect=AssertionError('coverage')) as coverage, \
                patch('tefed.server.renderer_target_observation', side_effect=AssertionError('RTA')) as rta:
            result = construct_cases([descriptor], FakeBank(), FakeRouter(),
                {descriptor['source_state_key']: STATE}, ['client1'], cal or calibration(),
                config, 3, 8, directory)
        self.assertEqual(len(renderer.requests), 1)
        service['load_case'].assert_not_called()
        for callback in (assess, predict, coverage, rta):
            callback.assert_not_called()
        self.assertEqual(result['cases'], [])
        entry = next(row for row in result['audit'] if row.get('case_id'))
        self.assertEqual(entry['reason'], 'empty_fidelity_mask')
        self.assertEqual(entry['stage'], 'fidelity_precheck')
        self.assertFalse(entry['entered_verification'])
        self.assertFalse(entry['retained'])
        self.assertIsNone(entry['fidelity'])
        self.assertEqual(entry['first_failed_gate'], 'fidelity')
        self.assertEqual(entry['case_id'], entry['rendered_variant_id'])
        self.assertEqual(entry['descriptor_id'], descriptor['id'])
        self.assertEqual(entry['source_route'], descriptor['route'])
        audit = entry['fidelity_audit']
        self.assertEqual(audit['active_mask'], [0] * 8)
        self.assertEqual(audit['active_weight_mass'], 0.)
        self.assertFalse(audit['evaluated'])
        self.assertFalse(audit['valid'])
        self.assertIsNone(audit['score'])
        self.assertNotIn('rta_observation', entry)
        self.assertNotIn('frozen_verification_inputs', entry)
        proportions = case_event_proportions(verification_records=result['audit'])
        self.assertEqual(proportions['retention']['denominator'], 0)
        self.assertIsNone(proportions['retention']['proportion'])
        self.assertEqual(descriptor, original)

    def test_nonempty_mask_preserves_expert_route_rta_and_frozen_inputs(self):
        for mode, reference in (('stored', [0, 1, 2]), ('current', [0, 3, 4])):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                descriptor = source_descriptor()
                original = deepcopy(descriptor)
                service = services(FakeRenderer())
                service['assess_frame'] = Mock(wraps=service['assess_frame'])
                service['predict_outputs'] = Mock(wraps=service['predict_outputs'])
                result = construct_cases([descriptor], FakeBank(), FakeRouter(),
                    {descriptor['source_state_key']: STATE}, ['client1'], calibration(),
                    config_for(service, mode), 3, 8, directory)
                service['assess_frame'].assert_called_once()
                service['predict_outputs'].assert_called_once()
                self.assertEqual(len(result['cases']), 1)
                entry = next(row for row in result['audit'] if row.get('entered_verification'))
                self.assertTrue(entry['retained'])
                self.assertEqual(entry['verification_reference']['route'], reference)
                self.assertIn('rta_observation', entry)
                frozen = entry['frozen_verification_inputs']
                self.assertEqual(frozen['descriptor']['route'], descriptor['route'])
                self.assertEqual(set(frozen['expert_outputs']), set(map(str, range(6))))
                self.assertEqual(frozen['attrs_measured'], entry['fidelity_measurements'])
                proportions = case_event_proportions(verification_records=result['audit'])
                self.assertEqual(proportions['retention']['denominator'], 1)
                self.assertEqual(descriptor, original)


if __name__ == '__main__':
    unittest.main()
