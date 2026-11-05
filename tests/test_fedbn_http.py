"""FedBN native training and fixed validation priors through private HTTP workers."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tefed import engine
from tefed.evaluation_manifest import identity_sha256
from tefed.fedbn import prior_from_arrays, validate_bn_prior
from tefed.resume import load_snapshot
from tefed.software_fixture import create_fixture
from tefed.worker import WorkerService, make_server, RemoteClientExecutor


class LostFedBNCoordinatorCommit(RuntimeError):
    pass


def fedbn_fixture(root):
    config = create_fixture(root)
    config['experiment'].update(method='fedbn', rounds=2, save_rounds=[1, 2])
    config['model']['specialists'] = 0
    config['evaluation']['monitor_every'] = 1
    return config


def prediction_rows(output, view):
    path = Path(output) / 'predictions' / f'{view}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class FedBNHTTPTests(unittest.TestCase):
    def test_remote_binding_hashes_nonworker_validation_without_private_images(self):
        from tefed import resume
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = fedbn_fixture(root / 'coordinator')
            path = Path(config['experiment']['manifest'])
            data = json.loads(path.read_text(encoding='utf-8'))
            server = dict(next(row for row in data['records'] if row['split'] == 'validation'))
            server.update(id='server:validation:extra', client_id='server')
            for sensor in ('rgb', 'depth', 'nir'):
                source = Path(server[sensor])
                target = source.with_name('server_extra_' + source.name)
                target.write_bytes(source.read_bytes())
                server[sensor] = str(target)
            self.assertNotIn(server['id'], data['calibration_ids'])
            workers = {'client0': {'partition_sha256': 'a' * 64},
                       'client1': {'partition_sha256': 'b' * 64}}
            private = root / 'unavailable-private-worker-inputs'
            for row in data['records']:
                for sensor in ('rgb', 'depth', 'nir'):
                    if sensor in row:
                        row[sensor] = str(private / Path(row[sensor]).name)
                        self.assertFalse(Path(row[sensor]).exists())
            data['records'].append(server)
            path.write_text(json.dumps(data), encoding='utf-8')
            manifest = engine.load_manifest(path, strict_files=False)
            original_hash = resume.file_hash
            def public_only(asset):
                self.assertFalse(Path(asset).is_relative_to(private))
                return original_hash(asset)
            with patch('tefed.resume.file_hash', side_effect=public_only):
                first = resume.run_binding(config, manifest, labels_only=False,
                                           secret=b's' * 32, workers=workers)
                rgb = Path(server['rgb'])
                rgb.write_bytes(rgb.read_bytes() + b'\0')
                second = resume.run_binding(config, manifest, labels_only=False,
                                            secret=b's' * 32, workers=workers)
            rgb_key = str(rgb.resolve())
            self.assertIn(rgb_key, first['input_files'])
            self.assertIn(rgb_key, second['input_files'])
            self.assertNotEqual(first['input_files'][rgb_key], second['input_files'][rgb_key])
            self.assertEqual(first['workers'], second['workers'])

    def test_http_prior_predictions_and_precommit_recovery_match_local_without_coordinator_images(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        servers, threads = [], []
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {'TEFED_TRANSPORT_KEY': 't' * 32}):
                root = Path(directory)
                local_config = fedbn_fixture(root / 'local-data')
                local = engine.run_experiment(local_config, seed=3, output_dir=root / 'local-run')
                remote_config = fedbn_fixture(root / 'coordinator')
                manifest_path = Path(remote_config['experiment']['manifest'])
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                for record in manifest['records']:
                    for sensor in ('rgb', 'depth', 'nir'):
                        if sensor in record:
                            record[sensor] = str(root / 'not-on-coordinator' / Path(record[sensor]).name)
                            self.assertFalse(Path(record[sensor]).exists())
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                settings = {}
                for index in range(2):
                    client_id = f'client{index}'
                    worker_config = fedbn_fixture(root / f'worker-data-{index}')
                    path = Path(worker_config['experiment']['manifest'])
                    data = json.loads(path.read_text(encoding='utf-8'))
                    data['records'] = [record for record in data['records'] if record['client_id'] == client_id]
                    ids = {record['id'] for record in data['records']}
                    data['calibration_ids'] = [uid for uid in data['calibration_ids'] if uid in ids]
                    data['views'] = {view: [uid for uid in values if uid in ids] for view, values in data['views'].items()}
                    path.write_text(json.dumps(data), encoding='utf-8')
                    service = WorkerService(worker_config, client_id, f'node{index}', b't' * 32,
                                            root / f'worker-cache-{index}')
                    server = make_server(service)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    servers.append(server)
                    threads.append(thread)
                    settings[client_id] = {'url': f'http://127.0.0.1:{server.server_address[1]}',
                                           'worker_id': f'node{index}'}
                remote_config['communication'] = {'mode': 'http', 'workers': settings}

                original_execute = RemoteClientExecutor.execute
                prior_clients = []
                def inspect_response(executor, action, model, router, calibration, **kwargs):
                    result = original_execute(executor, action, model, router, calibration, **kwargs)
                    if action == 'fedbn_prior':
                        prior_clients.append(str(kwargs['client_id']))
                        self.assertEqual(set(result), {'fedbn_prior'})
                        wire = result['fedbn_prior']
                        self.assertEqual(set(wire), {'schema_version', 'state', 'moments', 'protocol'})
                        # No RGB/depth/IR image, raw path, or sample/target payload is returned.
                        self.assertTrue(all(isinstance(value, np.ndarray) and value.ndim <= 1
                                            for value in wire['state'].values()))
                        for moment in wire['moments'].values():
                            self.assertEqual(set(moment), {'count', 'mean', 'm2'})
                            self.assertEqual(moment['mean'].ndim, 1)
                            self.assertEqual(moment['m2'].ndim, 1)
                        json.dumps(wire['protocol'], allow_nan=False)
                        metadata = json.dumps(wire['protocol'], sort_keys=True)
                        self.assertNotIn(str(root), metadata)
                        self.assertNotIn('.png', metadata)
                        validate_bn_prior(model, prior_from_arrays(wire), validation_ids=[
                            record.id for record in engine.load_manifest(remote_config['experiment']['manifest']).by_split(
                                'validation', str(kwargs['client_id']))])
                    return result
                with patch.object(RemoteClientExecutor, 'execute', new=inspect_response):
                    remote = engine.run_experiment(remote_config, seed=3, output_dir=root / 'http-run')
                self.assertEqual(prior_clients, ['client0', 'client1'])

                output = root / 'http-interrupted'
                original_snapshot = engine.atomic_snapshot
                original_train = engine.train_local
                training_rounds = []
                def lose_commit(path, payload):
                    if payload.get('phase') == 'training' and payload.get('next_round') == 1:
                        raise LostFedBNCoordinatorCommit('workers committed; coordinator lost its round commit')
                    return original_snapshot(path, payload)
                def count_training(model, items, config, **kwargs):
                    training_rounds.append(kwargs['round_id'])
                    return original_train(model, items, config, **kwargs)
                def forbid_prior_rebuild(executor, action, *args, **kwargs):
                    if action == 'fedbn_prior':
                        raise AssertionError('Resume rebuilt an already committed validation prior')
                    return original_execute(executor, action, *args, **kwargs)
                with patch('tefed.engine.train_local', side_effect=count_training):
                    with patch('tefed.engine.atomic_snapshot', side_effect=lose_commit):
                        with self.assertRaises(LostFedBNCoordinatorCommit):
                            engine.run_experiment(remote_config, seed=3, output_dir=output)
                    boundary = load_snapshot(output / 'resume.pt')
                    self.assertEqual(boundary['next_round'], 0)
                    self.assertEqual(training_rounds, [0, 0])
                    with patch.object(RemoteClientExecutor, 'execute', new=forbid_prior_rebuild):
                        recovered = engine.run_experiment(remote_config, seed=3, output_dir=output,
                                                         resume=output / 'resume.pt')
                    # Retried round 0 comes from authenticated worker cache, not repeated SGD.
                    self.assertEqual(training_rounds, [0, 0, 1, 1])

                reference = load_snapshot(Path(local['output_dir']) / 'resume.pt')
                prior_digest = identity_sha256(reference['fedbn_prior'])
                self.assertEqual(identity_sha256(boundary['fedbn_prior']), prior_digest)
                for result in [remote, recovered]:
                    snapshot = load_snapshot(Path(result['output_dir']) / 'resume.pt')
                    self.assertEqual(identity_sha256(snapshot['fedbn_prior']), prior_digest)
                    for field in ('model', 'router', 'states', 'client_runtime', 'bn_cache', 'rng'):
                        self.assertEqual(identity_sha256(snapshot[field]), identity_sha256(reference[field]), field)
                    self.assertEqual(result['metrics'], local['metrics'])
                    for view in local['metrics']:
                        self.assertEqual(prediction_rows(local['output_dir'], view), prediction_rows(result['output_dir'], view), view)
                    _, _, checkpoint = engine.load_checkpoint(result['checkpoint'], device='cpu')
                    self.assertEqual(identity_sha256(checkpoint['fedbn_prior']), prior_digest)
                    audit = json.loads((Path(result['output_dir']) / 'predictions/evaluation_manifest.json').read_text(encoding='utf-8'))
                    for record in audit['views']['new-client']['records']:
                        self.assertEqual(record['model_selector']['kind'], 'global_with_validation_bn')
                        self.assertEqual(record['model_selector']['bn_sha256'], identity_sha256(reference['fedbn_prior']['state']))
                self.assertTrue(all(not Path(record[sensor]).exists() for record in manifest['records']
                                    for sensor in ('rgb', 'depth', 'nir') if sensor in record))
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(5)
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
