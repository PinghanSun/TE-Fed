import unittest
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from tefed.measurement import summarize_request_traces


def row(i, value=1., **extra):
    return {'scope': 'forward', 'block_id': '0', 'request_id': str(i),
            'phase': 'timed', 'status': 'success', 'latency_ms': value, **extra}


class RequestMeasurementTests(unittest.TestCase):
    def test_nearest_rank_and_failures_exclude_warmups(self):
        rows = [row(i, i) for i in range(1, 21)]
        rows += [row(0, 9999, phase='warmup'), row(21, status='timeout'), row(22, status='failure')]
        result = summarize_request_traces(rows)
        b = result['blocks'][0]
        self.assertEqual(b['p95_ms'], 19.)  # linear percentile would be 19.05
        self.assertEqual((b['warmup_requests'], b['timed_attempts'], b['failures'], b['timeouts']), (1, 22, 1, 1))
        self.assertIsNone(result['summaries']['forward']['sample_sd_p95_ms'])

    def test_mean_and_sample_sd_use_block_p95_and_keep_scopes_separate(self):
        result = summarize_request_traces([row(0, 1), row(0, 3, block_id='1'), row(0, 100, scope='workflow')])
        self.assertEqual(result['summaries']['forward']['mean_p95_ms'], 2.)
        self.assertAlmostEqual(result['summaries']['forward']['sample_sd_p95_ms'], 2 ** .5)
        self.assertEqual(result['summaries']['workflow']['mean_p95_ms'], 100.)

    def test_no_success_is_undefined_not_zero(self):
        r = summarize_request_traces([row(0, status='failure'), row(1, status='timeout')])
        self.assertIsNone(r['blocks'][0]['p95_ms'])
        self.assertIsNone(r['summaries']['forward']['mean_p95_ms'])
        self.assertEqual(r['summaries']['forward']['defined_blocks'], 0)

    def test_bad_records_and_duplicates_rejected(self):
        for bad in [row(0, float('nan')), row(0, -1), row(0, True), row(0, phase='unknown')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize_request_traces([bad])
        for rows in [[], [row(0), row(0)]]:
            with self.assertRaises(ValueError):
                summarize_request_traces(rows)

    def test_csv_cli_writes_summary_and_preserves_existing_output(self):
        from scripts.summarize_request_traces import main
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'requests.csv', root / 'summary.json'
            with source.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row(0)))
                writer.writeheader()
                writer.writerows([row(i, i) for i in range(1, 21)])
            self.assertEqual(main(['--csv', str(source), '--out', str(output)]), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['blocks'][0]['p95_ms'], 19.)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(['--csv', str(source), '--out', str(output)])
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
