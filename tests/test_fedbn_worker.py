from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from tefed.fedbn import (build_validation_bn_prior, bn_prior_to_arrays, prior_from_arrays,
                         validate_bn_prior, merge_validation_bn_priors)
from tefed.transport import pack_payload, unpack_payload, payload_digest
from tefed.worker_cache import WorkerTaskCache


@unittest.skipUnless(torch is not None, 'Requires native training dependencies')
class FedBNWorkerTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def small_prior(self):
        model = torch.nn.BatchNorm1d(2)
        artifact = build_validation_bn_prior(model, [torch.tensor([[1., 3.], [5., 7.]])],
            validation_ids=['v1', 'v2'], binding={'initial_model_sha256': 'a' * 64, 'preprocessing_sha256': 'b' * 64})
        return model, artifact

    def fixture_model(self, root):
        from tefed.data import load_manifest
        from tefed.model import RoutedYOLO, Router
        from tefed.software_fixture import create_fixture
        config = create_fixture(root)
        config['experiment']['method'] = 'fedbn'
        config['model']['specialists'] = 0
        model, router = RoutedYOLO(config), Router()
        manifest = load_manifest(config['experiment']['manifest'], strict_files=True)
        calibration = json.loads(Path(config['experiment']['calibration']).read_text())
        return config, model, router, manifest, calibration

    def test_transport_roundtrip_converts_only_state_and_moment_arrays(self):
        model, artifact = self.small_prior()
        wire = bn_prior_to_arrays(artifact)
        self.assertIsInstance(wire['state']['num_batches_tracked'], np.ndarray)
        self.assertEqual(wire['state']['num_batches_tracked'].shape, ())
        self.assertEqual(wire['moments']['']['mean'].dtype, np.float64)
        json.dumps(wire['protocol'], allow_nan=False)
        loaded_wire = unpack_payload(pack_payload({'fedbn_prior': wire}))['fedbn_prior']
        self.assertEqual(payload_digest(wire), payload_digest(loaded_wire))
        restored = prior_from_arrays(loaded_wire)
        validate_bn_prior(model, restored)
        self.assertIsInstance(restored['protocol']['validation_ids'], list)
        self.assertIsInstance(restored['protocol']['validation_ids'][0], str)
        for key, value in artifact['state'].items():
            torch.testing.assert_close(restored['state'][key], value, rtol=0, atol=0)
        restored['state']['running_mean'].add_(1)
        torch.testing.assert_close(artifact['state']['running_mean'], torch.tensor([3., 5.]))
        invalid = deepcopy(wire)
        invalid['moments']['']['mean'] = [1., 2.]
        with self.assertRaisesRegex(ValueError, 'NumPy'):
            prior_from_arrays(invalid)
        invalid = deepcopy(artifact)
        invalid['protocol']['tensor_in_metadata'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'plain finite JSON'):
            bn_prior_to_arrays(invalid)

    def test_prior_cache_action_is_idempotent_and_does_not_advance_round_cursor(self):
        _, artifact = self.small_prior()
        value = {'fedbn_prior': bn_prior_to_arrays(artifact)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            cache = WorkerTaskCache(path, {'source': 'worker0'}, b't' * 32, b's' * 32)
            calls = []
            first = cache.execute('run', 'prior', 'prior-digest', 'fedbn_prior', 0,
                                  lambda: (calls.append('prior'), value)[1])
            restarted = WorkerTaskCache(path, {'source': 'worker0'}, b't' * 32, b's' * 32)
            repeated = restarted.execute('run', 'prior', 'prior-digest', 'fedbn_prior', 0,
                                         lambda: self.fail('A committed prior was recomputed'))
            self.assertEqual(calls, ['prior'])
            self.assertEqual(payload_digest(first), payload_digest(repeated))
            restarted.execute('run', 'round0', 'round0-digest', 'client_round', 0, lambda: {'ok': True})
            restarted.execute('run', 'another-prior', 'other-digest', 'fedbn_prior', 99, lambda: value)
            restarted.execute('run', 'round1', 'round1-digest', 'client_round', 1, lambda: {'ok': True})
            with self.assertRaisesRegex(ValueError, 'different input'):
                restarted.execute('run', 'prior', 'changed', 'fedbn_prior', 0, lambda: self.fail('Changed input ran'))

    def test_merge_checks_shared_initial_model_and_preprocessing_bindings(self):
        model, left = self.small_prior()
        right = build_validation_bn_prior(model, [torch.tensor([[8., 9.], [10., 12.]])],
            validation_ids=['v3', 'v4'], binding=left['protocol']['binding'])
        for key in ['initial_model_sha256', 'preprocessing_sha256']:
            invalid = deepcopy(right)
            invalid['protocol']['binding'][key] = 'c' * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'different'):
                merge_validation_bn_priors(model, [left, invalid])
            del invalid['protocol']['binding'][key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, 'different'):
                merge_validation_bn_priors(model, [left, invalid])
            with self.subTest(server=key), self.assertRaisesRegex(ValueError, 'binding differs'):
                merge_validation_bn_priors(model, [left, right], binding={key: 'c' * 64})

    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Requires native detector dependencies')
    def test_client_prior_action_reads_only_this_clients_validation_with_native_forward(self):
        from tefed.client import execute_client_task, module_state
        from tefed.engine import prepared_sample
        from tefed.evaluation_manifest import identity_sha256
        with tempfile.TemporaryDirectory() as folder:
            config, model, router, manifest, calibration = self.fixture_model(Path(folder))
            before = module_state(model)
            read_ids = []
            def validation_only(record, settings):
                self.assertEqual(record.split, 'validation')
                self.assertEqual(str(record.client_id), 'client0')
                read_ids.append(record.id)
                return prepared_sample(record, settings)
            with patch('tefed.engine.prepared_sample', side_effect=validation_only):
                output = execute_client_task(config, manifest, model, router, calibration,
                                             action='fedbn_prior', client_id='client0')
            expected = sorted(record.id for record in manifest.by_split('validation', 'client0'))
            self.assertEqual(read_ids, expected)
            artifact = prior_from_arrays(unpack_payload(pack_payload(output))['fedbn_prior'])
            validate_bn_prior(model, artifact, validation_ids=expected)
            self.assertEqual(artifact['protocol']['binding']['initial_model_sha256'], identity_sha256(model.state_dict()))
            self.assertEqual(artifact['protocol']['unvisited_modules'], [])
            self.assertEqual(len(artifact['protocol']['visited_modules']), 114)
            for key, value in before.items():
                np.testing.assert_array_equal(module_state(model)[key], value)
            with self.assertRaisesRegex(ValueError, 'no validation pool'):
                execute_client_task(config, manifest, model, router, calibration,
                                    action='fedbn_prior', client_id='missing-client')

    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Requires native detector dependencies')
    def test_client_newclient_installs_validation_prior_and_other_view_uses_local_bn(self):
        from tefed.client import build_client_validation_bn_prior, evaluate_client, module_state
        from tefed.baselines import batchnorm_state_keys
        from tefed.evaluation_manifest import identity_sha256
        with tempfile.TemporaryDirectory() as folder:
            config, model, router, manifest, calibration = self.fixture_model(Path(folder))
            config['evaluation']['views'] = ['participating', 'new-client']
            wire = build_client_validation_bn_prior(model, manifest, config, client_id='client0')['fedbn_prior']
            keys = batchnorm_state_keys(model)
            before = module_state(model)
            local_bn = {key: value.copy() for key, value in before.items() if key in keys}
            for key in local_bn:
                if key.endswith('running_mean'):
                    local_bn[key].fill(17.)
            expected_local = identity_sha256(local_bn)
            expected_prior = identity_sha256(wire['state'])
            observed = []
            def assess(current, *args, **kwargs):
                observed.append(identity_sha256({key: value for key, value in module_state(current).items() if key in keys}))
                return {'candidates': []}
            with patch('tefed.engine.assess_frame', side_effect=assess):
                result = evaluate_client(model, router, manifest, calibration['state_prior'], calibration, config,
                    client_id='client0', persistent_state={'bn': local_bn, 'fedbn_prior': wire})
            self.assertEqual(observed, [expected_local, expected_prior])
            self.assertEqual(set(result['views']), {'participating', 'new-client'})
            for key, value in before.items():
                np.testing.assert_array_equal(module_state(model)[key], value)
            config['evaluation']['views'] = ['new-client']
            with self.assertRaisesRegex(ValueError, 'validation-pooled BN prior'):
                evaluate_client(model, router, manifest, calibration['state_prior'], calibration, config,
                                client_id='client0', persistent_state={'bn': local_bn})
            changed = deepcopy(config)
            changed['model']['depth_scale'] = 123.
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                evaluate_client(model, router, manifest, calibration['state_prior'], calibration, changed,
                                client_id='client0', persistent_state={'fedbn_prior': wire})


if __name__ == '__main__':
    unittest.main()
