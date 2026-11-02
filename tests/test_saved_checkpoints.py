"""Saved completed-round checkpoints and recovery of unfinished checkpoint writes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tefed import engine
from tefed.resume import file_hash, file_journal, load_snapshot, reconcile_files
from tefed.software_fixture import create_fixture


def assert_tree_equal(test, left, right):
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    elif isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    else:
        test.assertEqual(left, right)


class CheckpointConfigurationTests(unittest.TestCase):
    def test_pretrain_uses_its_own_round_budget_without_adaptation_saves(self):
        config={'experiment':{'rounds':5, 'save_rounds':[300, 500, 1000]}, 'model':{}}
        with patch('tefed.engine.run_experiment', return_value={'checkpoint':'initial.pt'}) as run:
            engine.pretrain(config, output_dir='initial')
        prepared=run.call_args.args[0]
        engine.validate_run_config(prepared)
        self.assertNotIn('save_rounds', prepared['experiment'])
        self.assertEqual(config['experiment']['save_rounds'], [300, 500, 1000])
        self.assertTrue(run.call_args.kwargs['labels_only'])

    def test_rounds_are_distinct_ordered_completed_round_counts(self):
        for values in ([1, 300, 500, 1000], []):
            engine.validate_run_config({'experiment': {'rounds': 1000, 'save_rounds': values}})
        for values in ([0], [True], [1.0], [2, 1], [1, 1], [1001], '300', None):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'save_rounds'):
                engine.validate_run_config({'experiment': {'rounds': 1000, 'save_rounds': values}})

    def test_journal_keeps_completed_checkpoints_and_removes_only_newer_uncommitted_files(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'rounds.jsonl').write_text('{"round_index":0}\n', encoding='utf-8')
            checkpoints=output/'checkpoints';checkpoints.mkdir()
            committed=checkpoints/'round_000001.pt';committed.write_bytes(b'committed')
            pool=committed.with_suffix('.pool.json');pool.write_text('[]', encoding='utf-8')
            journal=file_journal(output, round_index=0)
            self.assertIn('checkpoints/round_000001.pt', journal)
            newer=[checkpoints/name for name in ('round_000002.pt', 'round_000002.pool.json', 'round_000002.pt.tmp')]
            for path in newer:
                path.write_bytes(b'partial')
            unrelated=checkpoints/'notes.txt';unrelated.write_text('keep', encoding='utf-8')
            reconcile_files(output, journal, 1)
            self.assertEqual(committed.read_bytes(), b'committed')
            self.assertTrue(pool.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(all(not path.exists() for path in newer))
            committed.write_bytes(b'changed')
            newer[0].write_bytes(b'keep until committed files validate')
            with self.assertRaisesRegex(ValueError, 'Committed run file changed'):
                reconcile_files(output, journal, 1)
            self.assertTrue(newer[0].exists())


class NativeSavedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_interrupted_uncommitted_checkpoint_recovery_matches_continuous_training(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['experiment']['save_rounds']=[1, 2]
            continuous=engine.run_experiment(config, output_dir=root/'continuous')
            interrupted=root/'interrupted'
            original=engine.atomic_snapshot

            def stop_before_commit(path, payload):
                if Path(path).name=='resume.pt' and payload.get('next_round')==2 and payload.get('phase')=='training':
                    raise RuntimeError('interrupt after saving the second checkpoint')
                return original(path, payload)

            with patch('tefed.engine.atomic_snapshot', side_effect=stop_before_commit):
                with self.assertRaisesRegex(RuntimeError, 'interrupt after saving'):
                    engine.run_experiment(config, output_dir=interrupted)
            boundary=load_snapshot(interrupted/'resume.pt')
            self.assertEqual(boundary['next_round'], 1)
            first=interrupted/'checkpoints/round_000001.pt'
            first_hash=file_hash(first)
            saved=torch.load(first, map_location='cpu', weights_only=True)
            self.assertEqual(saved['completed_rounds'], 1)
            for name in ('model', 'router', 'states'):
                assert_tree_equal(self, saved[name], boundary[name])
            self.assertTrue((interrupted/'checkpoints/round_000002.pt').exists())
            unfinished=interrupted/'checkpoints/round_000002.pt.tmp'
            unfinished.write_bytes(b'uncommitted temporary write')
            resumed=engine.run_experiment(config, output_dir=interrupted, resume=interrupted/'resume.pt')
            self.assertEqual(file_hash(first), first_hash)
            self.assertFalse(unfinished.exists())
            self.assertEqual(continuous['metrics'], resumed['metrics'])
            full=load_snapshot(root/'continuous/resume.pt')
            recovered=load_snapshot(interrupted/'resume.pt')
            for name in ('model', 'router', 'states', 'rng'):
                assert_tree_equal(self, full[name], recovered[name])
            for count in (1, 2):
                relative=f'checkpoints/round_{count:06d}.pt'
                self.assertEqual(recovered['files'][relative]['sha256'], file_hash(interrupted/relative))
            final_saved=torch.load(interrupted/'checkpoints/round_000002.pt', map_location='cpu', weights_only=True)
            self.assertEqual(final_saved['completed_rounds'], 2)
            self.assertEqual(final_saved['round_index'], 1)
            for name in ('model', 'router', 'states'):
                assert_tree_equal(self, final_saved[name], recovered[name])

    def test_saved_local_and_fedbn_bundles_evaluate_with_frozen_client_states(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for method in ('local', 'fedbn'):
                with self.subTest(method=method):
                    config=create_fixture(root/method/'fixture')
                    config['experiment'].update(method=method, rounds=1, save_rounds=[1])
                    output=root/method/'run'
                    result=engine.run_experiment(config, output_dir=output)
                    checkpoint=output/'checkpoints/round_000001.pt'
                    model, router, payload=engine.load_checkpoint(checkpoint, device='cpu')
                    committed=load_snapshot(output/'resume.pt')
                    assert_tree_equal(self, payload['states'], committed['states'])
                    manifest=engine.load_manifest(config['experiment']['manifest'])
                    calibration=json.loads(Path(config['experiment']['calibration']).read_text(encoding='utf-8'))
                    if method=='local':
                        self.assertEqual(payload['checkpoint_kind'], 'local_bundle')
                        self.assertEqual(len(payload['client_models']), 2)
                        self.assertIn(payload['local_new_client_id'], payload['client_models'])
                        extra={'client_models':payload['client_models'], 'local_new_client_id':payload['local_new_client_id']}
                    else:
                        self.assertEqual(payload['checkpoint_kind'], 'fedbn_bundle')
                        self.assertEqual(len(payload['client_bn']), 2)
                        extra={'client_bn':payload['client_bn']}
                    metrics=engine.evaluate_model(model, router, manifest, payload['states'], calibration,
                                                  payload['config'], root/method/'recheck', **extra)
                    self.assertEqual(metrics, result['metrics'])

    def test_absent_save_rounds_does_not_create_checkpoint_directory(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['experiment']['rounds']=1
            result=engine.run_experiment(config, output_dir=root/'run')
            self.assertFalse((Path(result['output_dir'])/'checkpoints').exists())


if __name__=='__main__':
    unittest.main()
