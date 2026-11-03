"""Summarize timed requests without mixing them with node-registry metrics."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


def summarize_request_traces(rows):
    """Compute nearest-rank P95 per scope/block from successful timed attempts.

    Rows require scope, block_id, request_id, phase (warmup/timed), status
    (success/failure/timeout), and latency_ms for successful timed requests.
    Undefined quantiles and SDs stay None, rather than being filled with zero.
    """
    groups = defaultdict(lambda: {'values': [], 'warmup': 0, 'failure': 0, 'timeout': 0})
    seen = set()
    for row in rows:
        keys = []
        for field in ('scope', 'block_id', 'request_id'):
            value = row.get(field)
            if value is None or isinstance(value, bool) or not str(value).strip():
                raise ValueError(f'Request {field} is required')
            keys.append(str(value))
        scope, block, request = keys
        phase, status = row.get('phase'), row.get('status')
        if (not isinstance(phase, str) or phase not in {'warmup', 'timed'}
                or not isinstance(status, str) or status not in {'success', 'failure', 'timeout'}):
            raise ValueError('Request phase/status is invalid')
        key = (scope, block, phase, request)
        if key in seen:
            raise ValueError('Duplicate request within scope, block and phase')
        seen.add(key)
        group = groups[scope, block]
        if phase == 'warmup':
            group['warmup'] += 1
        elif status != 'success':
            group[status] += 1
        else:
            value = row.get('latency_ms')
            if isinstance(value, bool):
                raise ValueError('Successful request latency_ms must be finite and nonnegative')
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError('Successful request latency_ms is required') from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError('Successful request latency_ms must be finite and nonnegative')
            group['values'].append(value)
    if not groups:
        raise ValueError('Request traces are empty')
    blocks = []
    scopes = defaultdict(list)
    for (scope, block), group in sorted(groups.items()):
        values = sorted(group.pop('values'))
        n = len(values)
        p95 = values[math.ceil(.95 * n) - 1] if n else None
        record = {'scope': scope, 'block_id': block, 'p95_ms': p95,
                  'successful_timed_requests': n, 'timed_attempts': n + group['failure'] + group['timeout'],
                  'warmup_requests': group['warmup'], 'failures': group['failure'], 'timeouts': group['timeout']}
        blocks.append(record)
        scopes[scope].append(record)
    summaries = {}
    for scope, records in sorted(scopes.items()):
        values = [r['p95_ms'] for r in records if r['p95_ms'] is not None]
        summaries[scope] = {'mean_p95_ms': statistics.mean(values) if values else None,
                            'sample_sd_p95_ms': statistics.stdev(values) if len(values) > 1 else None,
                            'defined_blocks': len(values), 'total_blocks': len(records),
                            'blocks_without_success': [r['block_id'] for r in records if r['p95_ms'] is None]}
    return {'unit': 'ms', 'quantile_method': 'nearest_rank', 'percentile': 95,
            'statistical_unit': 'block-level P95 within each request scope',
            'blocks': blocks, 'summaries': summaries}
