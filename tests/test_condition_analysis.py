"""Four effective controls and paired-block contrast oracles."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np

from scripts import analyze_runs, run_suite
from tefed.analysis import condition_contrasts, paired_method_summary
from tefed.experiments import (condition_manifest, condition_plans,
                               factorial_plans, validate_factorial_config)
from tefed.software_fixture import create_fixture


def paired_records():
    blocks = []
    rows = []
    for seed in range(5):
        binding = {'block_id': f'block-{seed}', 'seed': seed,
                   'partition_sha256': f'{seed + 1:064x}',
                   'evaluation_ids_sha256': f'{seed + 11:064x}'}
        blocks.append(binding)
        for a in (0, 1):
            for e in (0, 1):
                rows.append({**binding, 'factors': {'A': a, 'E': e},
                             'value': .4 + .01 * seed + .01 * (seed + 1) * a
                                      + .02 * (seed + 2) * e + .005 * (seed - 2) * a * e,
                             'view': 'hard', 'metric': 'mAP50', 'checkpoint_round': 300})
    return rows, {'schema_version': 1, 'blocks': blocks}


class ConditionProtocolTests(unittest.TestCase):
    def test_native_four_condition_suite_exports_and_resumes_without_retraining(self):
        import torch
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            config['experiment'].update(rounds=1, seeds=[13])
            with contextlib.redirect_stdout(io.StringIO()):
                execution = run_suite.run_suite(config, 'conditions', root / 'suite', execute=True)
                forbidden = Mock(side_effect=AssertionError('completed condition was retrained'))
                resumed = run_suite.run_suite(config, 'conditions', root / 'suite', execute=True,
                                             resume=True, runner=forbidden)
            forbidden.assert_not_called()
            self.assertEqual(execution['status'], 'completed')
            self.assertEqual(resumed['status'], 'completed')
            rows = [json.loads(line) for line in (root / 'suite/conditions_hard_mAP50.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual({(row['factors']['A'], row['factors']['E']) for row in rows},
                             {(0, 0), (1, 0), (0, 1), (1, 1)})
            self.assertTrue(all(row['checkpoint_round'] == 1 for row in rows))

    def test_four_conditions_reuse_the_validated_operations(self):
        jobs = condition_plans()
        self.assertEqual([name for name, _ in jobs], ['A0E0', 'A1E0', 'A0E1', 'A1E1'])
        self.assertEqual(len(factorial_plans()), 8)
        for name, patch in jobs:
            config = run_suite.merge({'experiment': {'test_mode': True}}, patch)
            levels = validate_factorial_config(config)
            self.assertEqual(levels['B'], levels['C'])
            self.assertEqual(config['experiment']['method'], 'tefed' if levels['C'] else 'server_moe')
            self.assertEqual(name, f"A{levels['A']}E{levels['C']}")

    def test_conditions_manifest_and_suite_default_to_five_explicit_seed_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            config['experiment'].pop('seeds', None)
            manifest = run_suite.make_manifest(config, 'conditions', root / 'suite')
            self.assertEqual(manifest['seed_ids'], list(range(5)))
            self.assertEqual(len(manifest['jobs']), 20)
            self.assertEqual(manifest['version'], 'tefed-conditions-v1')
            self.assertIn('separate block manifest', manifest['repetition_unit'])
            for job in manifest['jobs']:
                resolved = run_suite._job_config(config, manifest, job)
                self.assertEqual(resolved['factorial']['plan_sha256'], manifest['protocol_sha256'])
            explicit = condition_manifest(config, [3, 9], root / 'different')
            self.assertEqual(explicit['seed_ids'], [3, 9])
            self.assertEqual(len(explicit['jobs']), 8)

    def test_export_keeps_actual_evaluation_identities_and_raw_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            manifest = condition_manifest(config, [7], root / 'suite')
            manifest['suite_plan'] = 'conditions'
            entries = []
            for job in manifest['jobs']:
                output = Path(job['output'])
                (output / 'predictions').mkdir(parents=True)
                (output / 'configuration.json').write_text(json.dumps({'experiment': {'rounds': 300}}), encoding='utf-8')
                (output / 'predictions/evaluation_manifest.json').write_text(json.dumps({
                    'seed': 7, 'views': {'hard': {'records': [{'id': 'b'}, {'id': 'a'}]}}}), encoding='utf-8')
                entries.append({'status': 'completed', 'result': {'metrics': {'hard': {'mAP50': .75}}}})
            run_suite._publish_factorial(root / 'suite', manifest, {'jobs': entries})
            rows = [json.loads(line) for line in (root / 'suite/conditions_hard_mAP50.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row['seed'] == 7 and row['value'] == .75 and row['checkpoint_round'] == 300 for row in rows))
            self.assertEqual(len({row['evaluation_ids_sha256'] for row in rows}), 1)
            self.assertTrue(all('block_id' not in row for row in rows))


class ConditionAnalysisTests(unittest.TestCase):
    def result(self, rows=None, manifest=None):
        standard_rows, standard_manifest = paired_records()
        return condition_contrasts(rows if rows is not None else standard_rows,
                                   manifest if manifest is not None else standard_manifest,
                                   expected_blocks=[f'block-{seed}' for seed in range(5)], scale=100)

    def test_within_block_conditional_and_interaction_oracle(self):
        result = self.result()
        np.testing.assert_allclose(result['summary']['A_at_E0']['values'], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(result['summary']['A_x_E']['values'], [-1, -.5, 0, .5, 1], atol=1e-12)
        self.assertAlmostEqual(result['summary']['A_x_E']['sample_sd'], np.sqrt(.625))
        self.assertEqual(result['summary']['A_x_E']['n'], 5)
        self.assertAlmostEqual(result['cells']['A0E0']['mean'], .42)
        # Compute SD from within-block contrasts.
        self.assertLess(result['summary']['A_x_E']['sample_sd'],
                        sum(value['sample_sd'] * 100 for value in result['cells'].values()))

    def test_declared_mapping_can_assign_ids_without_relabeling_seed_values(self):
        rows, manifest = paired_records()
        for row in rows:
            row.pop('block_id')
        result = self.result(rows, manifest)
        self.assertEqual(result['per_block'][0]['block_id'], 'block-0')
        self.assertEqual(result['per_block'][0]['seed'], 0)

    def test_protocols_may_differ_between_blocks_but_match_within_each_block(self):
        rows, manifest = paired_records()
        for row in rows:
            row['protocol_sha256'] = f"{row['seed'] + 101:064x}"
        result = self.result(rows, manifest)
        self.assertEqual(len(set(result['block_protocol_sha256'].values())), 5)
        self.assertEqual(result['per_block'][0]['protocol_sha256'], f'{101:064x}')
        changed = deepcopy(rows)
        changed[0]['protocol_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'within block'):
            self.result(changed, manifest)
        missing = deepcopy(rows)
        missing[0].pop('protocol_sha256')
        with self.assertRaisesRegex(ValueError, 'within block'):
            self.result(missing, manifest)

    def test_incomplete_duplicate_mismatched_and_nonfinite_records_are_rejected(self):
        rows, manifest = paired_records()
        variants = [rows[:-1], rows + [rows[0]]]
        for field, value in [('seed', 99), ('partition_sha256', 'f' * 64),
                             ('evaluation_ids_sha256', 'f' * 64), ('value', float('nan')),
                             ('checkpoint_round', 1000), ('view', 'unseen')]:
            changed = deepcopy(rows)
            changed[0][field] = value
            variants.append(changed)
        changed = deepcopy(rows)
        changed[0]['factors']['E'] = True
        variants.append(changed)
        for variant in variants:
            with self.subTest(first=variant[0]), self.assertRaises(ValueError):
                self.result(variant, manifest)
        with self.assertRaises(ValueError):
            self.result(rows, {'schema_version': 1, 'blocks': manifest['blocks'][:-1]})

    def test_cli_uses_five_current_ids_but_keeps_legacy_factorial_defaults(self):
        args = analyze_runs.parser().parse_args(['summary', '--runs', 'r', '--out', 'o'])
        self.assertEqual(args.seeds, ['0', '1', '2', '3', '4'])
        args = analyze_runs.parser().parse_args(['factorial', '--records', 'r', '--definitions', 'd', '--out', 'o'])
        self.assertEqual(args.seeds, list(map(str, range(10))))
        rows, manifest = paired_records()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'records.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
            (root / 'blocks.json').write_text(json.dumps(manifest), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                code = analyze_runs.main(['conditions', '--records', str(root / 'records.jsonl'),
                    '--block-manifest', str(root / 'blocks.json'), '--blocks',
                    *[f'block-{seed}' for seed in range(5)], '--scale', '100', '--out', str(root / 'result.json')])
            self.assertEqual(code, 0)
            result = json.loads((root / 'result.json').read_text())
            self.assertAlmostEqual(result['summary']['A_at_E0']['mean'], 3.)


class PairedMethodTests(unittest.TestCase):
    def inputs(self):
        _, manifest = paired_records()
        rows = []
        for index, binding in enumerate(manifest['blocks']):
            for method, offset in (('tefed', .02), ('proxyfl', 0), ('server_moe', -.03)):
                rows.append({**binding, 'method': method, 'value': .6 + .01 * index + offset,
                             'view': 'hard', 'metric': 'mAP50', 'checkpoint_round': 300})
        return rows, manifest

    def result(self, rows=None, manifest=None):
        original_rows, original_manifest = self.inputs()
        return paired_method_summary(rows if rows is not None else original_rows,
            manifest if manifest is not None else original_manifest,
            ['tefed', 'proxyfl', 'server_moe'],
            expected_blocks=[f'block-{seed}' for seed in range(5)])

    def test_partitions_vary_between_blocks_and_paired_sd_retains_covariance(self):
        result = self.result()
        self.assertAlmostEqual(result['methods']['tefed']['mean'], .64)
        self.assertGreater(result['methods']['tefed']['sample_sd'], 0)
        self.assertAlmostEqual(result['paired_differences']['proxyfl']['mean'], 2.)
        self.assertAlmostEqual(result['paired_differences']['proxyfl']['sample_sd'], 0.)
        self.assertAlmostEqual(result['paired_differences']['server_moe']['mean'], 5.)
        self.assertEqual(result['paired_differences']['proxyfl']['n'], 5)

    def test_missing_duplicate_and_mismatched_pairs_are_rejected(self):
        rows, manifest = self.inputs()
        variants = [rows[:-1], rows + [rows[0]]]
        for field, value in [('partition_sha256', 'f' * 64), ('evaluation_ids_sha256', 'f' * 64),
                             ('seed', 99), ('method', 'missing'), ('checkpoint_round', 500),
                             ('value', None)]:
            changed = deepcopy(rows)
            changed[0][field] = value
            variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):
                self.result(variant, manifest)

    def test_unreached_or_extra_blocks_cannot_change_pair_count(self):
        rows, manifest = self.inputs()
        extra = deepcopy(rows[0]); extra['block_id'] = 'extra'
        with self.assertRaises(ValueError):
            self.result(rows + [extra], manifest)
        with self.assertRaises(ValueError):
            self.result(rows, {'schema_version': 1, 'blocks': manifest['blocks'] + [manifest['blocks'][0]]})

    def test_cli_paired_results_use_the_declared_reference_and_input_scale(self):
        rows, manifest = self.inputs()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'records.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
            (root / 'blocks.json').write_text(json.dumps(manifest), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                code = analyze_runs.main(['paired', '--records', str(root / 'records.jsonl'),
                    '--block-manifest', str(root / 'blocks.json'), '--methods', 'tefed', 'proxyfl', 'server_moe',
                    '--reference', 'proxyfl', '--blocks', *[f'block-{seed}' for seed in range(5)],
                    '--out', str(root / 'result.json')])
            self.assertEqual(code, 0)
            result = json.loads((root / 'result.json').read_text())
            self.assertAlmostEqual(result['paired_differences']['tefed']['mean'], -2.)
            self.assertAlmostEqual(result['methods']['tefed']['mean'], .64)


if __name__ == '__main__':
    unittest.main()
