"""Preflight catches missing assets before any model or run-directory creation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tefed.preflight import check_setup
from tefed.software_fixture import create_fixture


class PreflightTests(unittest.TestCase):
    def test_fixture_check_does_not_construct_model_or_write_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('tefed.engine.RoutedYOLO') as constructor:
                report = check_setup(config)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(report['status'], 'passed', report)
            constructor.assert_not_called()
            self.assertEqual(before, after)
            self.assertNotIn('source-identifiers', json.dumps(report))

    def test_reports_independent_missing_inputs_together_without_echoing_secret(self):
        with TemporaryDirectory() as directory:
            config = {'experiment': {'manifest': str(Path(directory) / 'absent.json'),
                                     'calibration': str(Path(directory) / 'absent-cal.json')},
                      'renderer': {'enabled': False}}
            secret = 'do-not-print-this-key-' * 3
            with patch.dict('os.environ', {'TEFED_SOURCE_SECRET': secret}):
                report = check_setup(config)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual({r['check'] for r in report['errors']},
                             {'manifest', 'calibration_and_initial_checkpoint'})
            self.assertNotIn(secret, json.dumps(report))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pretrain_does_not_require_training_calibration_or_scene(self):
        with TemporaryDirectory() as directory:
            config = create_fixture(Path(directory) / 'fixture')
            config['experiment']['calibration'] = 'not-created-yet.json'
            config['renderer'] = {'enabled': True, 'scene': 'not-created-yet.blend'}
            report = check_setup(config, stage='pretrain')
            self.assertEqual(report['status'], 'passed', report)
            self.assertNotIn('renderer', report['checks'])

    def test_frozen_replay_checks_pool_without_requiring_blender(self):
        from tests.test_experiment_protocols import setup_replay
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory), native=True)
            with patch('tefed.engine.validate_renderer_setup') as renderer:
                report = check_setup(config)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['checks']['replay'], 'passed')
            renderer.assert_not_called()
            self.assertEqual(report['warnings'], [])

    def test_frozen_replay_rejects_changed_pool_asset(self):
        from tests.test_experiment_protocols import setup_replay
        from tefed.replay import FrozenPool
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory), native=True)
            pool = FrozenPool(config['replay']['pool_manifest'])
            Path(pool.cases[0]['rgb']).write_bytes(b'changed frozen input')
            report = check_setup(config)
            self.assertEqual(report['status'], 'failed')
            self.assertTrue(any(error['check'] == 'replay' for error in report['errors']))
