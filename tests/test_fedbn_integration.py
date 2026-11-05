"""Native FedBN validation-prior, saved evaluation and recovery integration."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from tefed import cli, engine
from tefed.evaluation_manifest import identity_sha256
from tefed.model import RoutedYOLO, Router
from tefed.resume import load_snapshot
from tefed.software_fixture import create_fixture


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prediction_rows(directory, view):
    return [json.loads(line) for line in (Path(directory) / f'{view}.jsonl').read_text(encoding='utf-8').splitlines()]


def write_fixture_config(path, config):
    """Serialize the fixture's scalar/list TOML sections for the actual CLI loader."""
    lines = [f'{key} = {json.dumps(value)}' for key, value in config.items() if not isinstance(value, dict)]
    for section, values in config.items():
        if isinstance(values, dict):
            lines.extend(['', f'[{section}]'])
            lines.extend(f'{key} = {json.dumps(value)}' for key, value in values.items())
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


class FedBNIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.config = create_fixture(cls.root / 'fixture')
        cls.config['experiment'].update(method='fedbn', rounds=2, save_rounds=[1, 2])
        cls.config['model'].update(specialists=0, class_names=['ripe', 'non-ripe'])
        cls.config['evaluation']['monitor_every'] = 1

        # This validation identity belongs to the coordinator, not either client.
        # It must still contribute to the one fixed validation population.
        manifest_path = Path(cls.config['experiment']['manifest'])
        manifest = read_json(manifest_path)
        server = deepcopy(next(record for record in manifest['records'] if record['split'] == 'validation'))
        server.update(id='server:validation:extra', client_id='server')
        for sensor in ('rgb', 'depth', 'nir'):
            source = Path(server[sensor])
            asset = source.with_name(f'server_validation_extra_{sensor}{source.suffix}')
            asset.write_bytes(source.read_bytes())
            server[sensor] = str(asset)
        manifest['records'].append(server)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        # A nondefault initial affine state exposes construction before inheritance.
        torch.manual_seed(41)
        initial = RoutedYOLO(cls.config)
        cls.first_bn_name, first_bn = next((name, layer) for name, layer in initial.named_modules()
                                           if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm))
        with torch.no_grad():
            first_bn.weight.fill_(1.375)
            first_bn.bias.fill_(.125)
            first_bn.running_mean.fill_(.375)
        initial_config = deepcopy(cls.config)
        initial_config['experiment']['method'] = 'fedavg'
        initial_path = cls.root / 'initial.pt'
        engine.save_checkpoint(initial_path, initial, Router(.5), initial_config, {}, [], -1)
        cls.initial_state = deepcopy(initial.state_dict())
        cls.config['model']['initial_checkpoint'] = str(initial_path)
        cls.config_path = cls.root / 'fedbn.toml'
        write_fixture_config(cls.config_path, cls.config)
        cls.output = cls.root / 'continuous'
        cls.result = engine.run_experiment(cls.config, seed=11, output_dir=cls.output)

    def loaded(self):
        return engine.load_checkpoint(self.result['checkpoint'], device='cpu')

    def assert_same_predictions(self, expected, actual):
        for view in self.config['evaluation']['views']:
            left, right = prediction_rows(expected, view), prediction_rows(actual, view)
            self.assertEqual(left, right, view)
            self.assertTrue(all(row['state_frozen'] for row in right), view)

    def assert_bn_selectors(self, audit_path, payload):
        audit = read_json(audit_path)
        prior_digest = identity_sha256(payload['fedbn_prior']['state'])
        for view, summary in audit['views'].items():
            self.assertTrue(summary['records'], view)
            for record in summary['records']:
                selector = record['model_selector']
                if view == 'new-client':
                    self.assertEqual(selector['kind'], 'global_with_validation_bn')
                    self.assertEqual(selector['bn_sha256'], prior_digest)
                else:
                    self.assertEqual(selector['kind'], 'global_with_client_bn')
                    self.assertEqual(selector['bn_sha256'], identity_sha256(payload['client_bn'][record['client_id']]))

    def test_saved_prior_covers_all_validation_and_stays_fixed_after_initial_inheritance(self):
        _, _, payload = self.loaded()
        prior = payload['fedbn_prior']
        self.assertEqual(payload['checkpoint_kind'], 'fedbn_bundle')
        self.assertEqual(set(prior), {'schema_version', 'state', 'moments', 'protocol'})
        protocol = prior['protocol']
        records = engine.load_manifest(self.config['experiment']['manifest']).by_split('validation')
        expected_ids = sorted(record.id for record in records)
        self.assertEqual(protocol['source_split'], 'validation')
        self.assertEqual(protocol['validation_ids'], expected_ids)
        self.assertIn('server:validation:extra', expected_ids)
        self.assertEqual(protocol['input_count'], len(expected_ids))
        self.assertEqual(protocol['binding']['initial_model_sha256'], identity_sha256(self.initial_state))
        for field in ('weight', 'bias'):
            key = f'{self.first_bn_name}.{field}'
            torch.testing.assert_close(prior['state'][key], self.initial_state[key], rtol=0, atol=0)
        self.assertNotEqual(identity_sha256(prior['state']), identity_sha256(payload['client_bn']['client0']))
        digest = identity_sha256(prior)
        for completed in (1, 2):
            saved = torch.load(self.output / f'checkpoints/round_{completed:06d}.pt', map_location='cpu', weights_only=True)
            self.assertEqual(identity_sha256(saved['fedbn_prior']), digest)
            self.assert_bn_selectors(self.output / f'monitoring/round_{completed - 1:03d}/predictions/evaluation_manifest.json', saved)
        self.assertEqual(identity_sha256(load_snapshot(self.output / 'resume.pt')['fedbn_prior']), digest)
        self.assert_bn_selectors(self.output / 'predictions/evaluation_manifest.json', payload)

    def test_direct_saved_checkpoint_and_cli_predictions_match_without_state_changes(self):
        model, router, payload = self.loaded()
        manifest = engine.load_manifest(self.config['experiment']['manifest'])
        calibration = read_json(self.config['experiment']['calibration'])
        frozen = identity_sha256({'model': model.state_dict(), 'router': router.state_dict(),
                                  'states': payload['states'], 'prior': payload['fedbn_prior'],
                                  'client_bn': payload['client_bn']})
        direct = self.root / 'direct'
        metrics = engine.evaluate_model(model, router, manifest, payload['states'], calibration,
            payload['config'], direct, client_bn=payload['client_bn'], fedbn_prior=payload['fedbn_prior'])
        self.assertEqual(metrics, self.result['metrics'])
        self.assert_same_predictions(self.output / 'predictions', direct)
        self.assert_bn_selectors(direct / 'evaluation_manifest.json', payload)
        self.assertEqual(frozen, identity_sha256({'model': model.state_dict(), 'router': router.state_dict(),
                                                'states': payload['states'], 'prior': payload['fedbn_prior'],
                                                'client_bn': payload['client_bn']}))

        command_output = self.root / 'cli-evaluation'
        with contextlib.redirect_stdout(io.StringIO()):
            status = cli.main(['evaluate', '--checkpoint', self.result['checkpoint'],
                               '--config', str(self.config_path), '--output', str(command_output)])
        self.assertEqual(status, 0)
        self.assertEqual(read_json(command_output / 'metrics.json'), metrics)
        self.assert_same_predictions(direct, command_output / 'predictions')
        self.assert_bn_selectors(command_output / 'predictions/evaluation_manifest.json', payload)

    def test_missing_validation_prior_is_rejected_in_direct_bundle_and_cli_paths(self):
        model, router, payload = self.loaded()
        manifest = engine.load_manifest(self.config['experiment']['manifest'])
        calibration = read_json(self.config['experiment']['calibration'])
        with self.assertRaisesRegex(ValueError, '(?i)(prior|validation)'):
            engine.evaluate_model(model, router, manifest, payload['states'], calibration,
                payload['config'], self.root / 'missing-direct', client_bn=payload['client_bn'])
        with self.assertRaisesRegex(ValueError, '(?i)(prior|validation)'):
            engine._saved_checkpoint_bundle(model, manifest, payload['config'], {}, payload['client_bn'])
        payload.pop('fedbn_prior')
        missing = self.root / 'missing-prior.pt'
        torch.save(payload, missing)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '(?i)(prior|validation)'):
            cli.main(['evaluate', '--checkpoint', str(missing), '--config', str(self.config_path),
                      '--output', str(self.root / 'missing-cli')])

    def test_checkpoint_payload_stays_on_cpu_when_target_model_device_is_cuda(self):
        # Exercise the CUDA-target load policy on CPU-only CI. Model transfers are
        # mocked; the checkpoint storage and prior tensors are loaded for real.
        load = torch.load
        with patch('tefed.engine.torch.load', wraps=load) as loader, \
             patch.object(RoutedYOLO, 'to', autospec=True, side_effect=lambda current, *args, **kwargs: current) as model_to, \
             patch.object(Router, 'to', autospec=True, side_effect=lambda current, *args, **kwargs: current) as router_to:
            model, router, payload = engine.load_checkpoint(self.result['checkpoint'], device='cuda:0')
        self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')
        self.assertEqual(model_to.call_args.args[-1], 'cuda:0')
        self.assertEqual(router_to.call_args.args[-1], 'cuda:0')
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        self.assertEqual(next(router.parameters()).device.type, 'cpu')
        for moment in payload['fedbn_prior']['moments'].values():
            for field in ('mean', 'm2'):
                self.assertEqual(moment[field].device.type, 'cpu')
                self.assertEqual(moment[field].dtype, torch.float64)

    def test_ordinary_views_reject_missing_or_partial_client_bn_and_restore_state(self):
        model, router, payload = self.loaded()
        manifest = engine.load_manifest(self.config['experiment']['manifest'])
        calibration = read_json(self.config['experiment']['calibration'])
        config = deepcopy(payload['config'])
        config['evaluation']['views'] = ['participating']
        partial = deepcopy(payload['client_bn'])
        partial['client0'].pop(next(iter(partial['client0'])))
        caches = [None, {}, {'client0': payload['client_bn']['client0']}, partial]
        frozen = identity_sha256({'model': model.state_dict(), 'router': router.state_dict()})
        for index, cache in enumerate(caches):
            with self.subTest(cache=index), self.assertRaisesRegex(ValueError, '(?i)(BN|batchnorm|normalization)'):
                engine.evaluate_model(model, router, manifest, payload['states'], calibration, config,
                    self.root / f'incomplete-bn-{index}', client_bn=cache, fedbn_prior=payload['fedbn_prior'])
            self.assertEqual(frozen, identity_sha256({'model': model.state_dict(), 'router': router.state_dict()}))

    def test_exception_after_installing_validation_prior_restores_model_and_router(self):
        model, router, payload = self.loaded()
        manifest = engine.load_manifest(self.config['experiment']['manifest'])
        calibration = read_json(self.config['experiment']['calibration'])
        config = deepcopy(payload['config'])
        config['evaluation']['views'] = ['new-client']
        prior_state = payload['fedbn_prior']['state']
        before = identity_sha256({'model': model.state_dict(), 'router': router.state_dict()})
        self.assertNotEqual(identity_sha256({key: model.state_dict()[key] for key in prior_state}),
                            identity_sha256(prior_state))

        def fail_after_prior_installed(current, *args, **kwargs):
            self.assertEqual(identity_sha256({key: current.state_dict()[key] for key in prior_state}),
                             identity_sha256(prior_state))
            raise RuntimeError('failure after installing the validation prior')

        with patch('tefed.engine.assess_frame', side_effect=fail_after_prior_installed):
            with self.assertRaisesRegex(RuntimeError, 'after installing the validation prior'):
                engine.evaluate_model(model, router, manifest, payload['states'], calibration, config,
                    self.root / 'failure-after-prior', client_bn=payload['client_bn'], fedbn_prior=payload['fedbn_prior'])
        self.assertEqual(before, identity_sha256({'model': model.state_dict(), 'router': router.state_dict()}))

    def test_interrupted_resume_reuses_the_identical_fixed_prior(self):
        output = self.root / 'interrupted'
        original_train = engine.train_local

        def interrupted(model, items, config, **kwargs):
            if kwargs.get('round_id') == 1:
                raise RuntimeError('FedBN integration round-boundary interruption')
            return original_train(model, items, config, **kwargs)

        with patch('tefed.engine.train_local', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'round-boundary interruption'):
                engine.run_experiment(self.config, seed=11, output_dir=output)
        boundary = load_snapshot(output / 'resume.pt')
        self.assertEqual(boundary['next_round'], 1)
        prior_digest = identity_sha256(boundary['fedbn_prior'])
        from tefed.client import execute_client_task

        def forbid_rebuilding_prior(*args, **kwargs):
            if kwargs.get('action') == 'fedbn_prior':
                raise AssertionError('Resume rebuilt the fixed validation BN prior')
            return execute_client_task(*args, **kwargs)

        with patch('tefed.client.execute_client_task', side_effect=forbid_rebuilding_prior):
            resumed = engine.run_experiment(self.config, seed=11, output_dir=output, resume=output / 'resume.pt')
        full = load_snapshot(self.output / 'resume.pt')
        recovered = load_snapshot(output / 'resume.pt')
        self.assertEqual(prior_digest, identity_sha256(full['fedbn_prior']))
        self.assertEqual(prior_digest, identity_sha256(recovered['fedbn_prior']))
        for field in ('model', 'router', 'states', 'client_runtime', 'bn_cache', 'rng'):
            self.assertEqual(identity_sha256(full[field]), identity_sha256(recovered[field]), field)
        self.assertEqual(resumed['metrics'], self.result['metrics'])
        self.assert_same_predictions(self.output / 'predictions', output / 'predictions')


if __name__ == '__main__':
    unittest.main()
