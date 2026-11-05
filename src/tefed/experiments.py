"""Build experiment plans and input bindings."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256, file_digest
from itertools import product
import json
from pathlib import Path


FACTORIAL_VERSION = 'tefed-factorial-v2'
FACTORIAL_DEFINITIONS = {
    'A': {'on': 'State-dependent reliability qualification with the measured routed reliability weight.',
          'off': 'Fixed posterior qualification at 0.75 with unit pseudo-target and background weights per active head; retain validity checks, routing, measured reliability and correction eligibility.'},
    'B': {'on': 'Enable source-evidence correction only when C=1, with stored-route verification and state-matched reuse.',
          'off': 'Effective correction E=B AND C is off; disable the entire correction path.'},
    'C': {'on': 'Enable correction only when B=1; E=B AND C controls training, rendering, verification, pool insertion, descriptor/case probes and next-round selection.',
          'off': 'Server-MoE; disable the entire correction path, including descriptor collection.'},
}

CONDITION_VERSION = 'tefed-conditions-v2'
CONDITION_DEFINITIONS = {
    'A': deepcopy(FACTORIAL_DEFINITIONS['A']),
    'E': {'on': 'Enable the complete source-evidence correction path (B=1 and C=1).',
          'off': 'Disable correction and train Server-MoE (B=0 and C=0).'},
}


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).expanduser().open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def artifact_bindings(config):
    experiment, model = config['experiment'], config.get('model', {})
    initial = model.get('initial_checkpoint')
    if not initial and not experiment.get('test_mode', False):
        raise ValueError('An experiment protocol requires model.initial_checkpoint')
    return {'manifest_sha256': file_hash(experiment['manifest']),
            'calibration_sha256': file_hash(experiment['calibration']),
            'initial_checkpoint_sha256': file_hash(initial) if initial else None}


def factorial_plans():
    definitions = canonical_hash(FACTORIAL_DEFINITIONS)
    jobs = []
    for a, b, c in product((0, 1), repeat=3):
        factors = {'A': a, 'B': b, 'C': c}
        method = 'tefed' if b and c else 'server_moe'
        patch = {'experiment': {'method': method},
                 'factorial': {'version': FACTORIAL_VERSION, 'definitions_sha256': definitions, 'factors': factors}}
        jobs.append((f'A{a}_B{b}_C{c}', patch))
    return jobs


def condition_plans():
    """Build four A/E conditions with B=C=E."""
    legacy = {tuple(patch['factorial']['factors'][key] for key in ('A', 'B', 'C')): patch
              for _, patch in factorial_plans()}
    return [(f'A{a}E{e}', deepcopy(legacy[(a, e, e)]))
            for e, a in product((0, 1), repeat=2)]


def validate_factorial_config(config):
    protocol = config.get('factorial')
    if protocol is None:
        return None
    if config.get('communication', {}).get('mode', 'local') != 'local':
        raise ValueError('factorial requires communication.mode=local')
    if protocol.get('version') != FACTORIAL_VERSION or protocol.get('definitions_sha256') != canonical_hash(FACTORIAL_DEFINITIONS):
        raise ValueError('Unknown factorial protocol version or operation definitions')
    factors = protocol.get('factors', {})
    if set(factors) != {'A', 'B', 'C'} or any(type(bit) is not int or bit not in (0, 1) for bit in factors.values()):
        raise ValueError('Factorial levels must be integer A/B/C bits')
    effective_correction = bool(factors['B'] and factors['C'])
    expected = 'tefed' if effective_correction else 'server_moe'
    if config['experiment'].get('method') != expected:
        raise ValueError('Factorial method does not implement the declared B/C operations')
    if effective_correction:
        decision = config.get('decision', {})
        if decision.get('verification_route', 'stored') != 'stored' or decision.get('selection_mode', 'matched') != 'matched':
            raise ValueError('E=1 requires stored-route verification and state-matched selection')
    if effective_correction and not config['experiment'].get('test_mode', False):
        if not config.get('renderer', {}).get('enabled', True) or config.get('decision', {}).get('render_budget', 1152) <= 0:
            raise ValueError('E=1 (B=C=1) requires enabled rendering and a positive rendering budget')
        if config.get('loss', {}).get('beta', .5) <= 0:
            raise ValueError('E=1 (B=C=1) requires positive correction supervision')
    return factors


def apply_qualification_control(candidate, config):
    factors = validate_factorial_config(config)
    if factors is None or factors['A']:
        return candidate
    result = deepcopy(candidate)
    original = result.get('disposition')
    if not result.get('valid'):
        disposition = 'suppressed'
    elif max(result['posterior']) >= .75:
        disposition = 'qualified'
    else:
        decision = config.get('decision', {})
        cues = result['cues']
        eligible = result['D_E'] >= decision.get('delta_d', .35) and cues['C'] >= decision.get('delta_c', .5) and cues['G'] >= decision.get('delta_g', .5)
        disposition = 'correction' if eligible else 'suppressed'
    result['disposition'] = disposition
    result['qualification_control'] = {'mode': 'fixed_posterior', 'threshold': .75,
                                       'target_weight_policy': 'unit', 'original_disposition': original}
    return result


def factorial_manifest(config, seeds, output):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Factorial seeds must be distinct nonnegative integers')
    bindings = artifact_bindings(config)
    resolved = deepcopy(config)
    resolved.pop('_config_path', None)
    for key in ('seed', 'seeds', 'output', 'manifest', 'calibration'):
        resolved['experiment'].pop(key, None)
    resolved.get('model', {}).pop('initial_checkpoint', None)
    jobs = []
    for variant, patch in factorial_plans():
        for seed in seeds:
            jobs.append({'variant': variant, 'seed': seed, 'factors': patch['factorial']['factors'],
                         'patch': patch, 'bindings': bindings,
                         'output': str(Path(output) / variant / f'seed_{seed}')})
    body = {'version': FACTORIAL_VERSION, 'operations': FACTORIAL_DEFINITIONS,
            'definitions_sha256': canonical_hash(FACTORIAL_DEFINITIONS), 'seed_ids': seeds,
            'bindings': bindings, 'scope': 'software_fixture' if config['experiment'].get('test_mode') else 'data_experiment',
            'configuration_sha256': canonical_hash(resolved),
            'structural_null': 'B0C0, B0C1 and B1C0 have no training effect relative to one another at fixed A: E=(B AND C)=0 disables the entire correction path.',
            'jobs': jobs}
    return {**body, 'protocol_sha256': canonical_hash(body)}


def condition_manifest(config, seeds, output):
    """Bind four effective conditions to the same declared run inputs."""
    legacy = factorial_manifest(config, seeds, output)
    body = {key: value for key, value in legacy.items()
            if key not in {'version', 'operations', 'definitions_sha256', 'structural_null',
                           'jobs', 'protocol_sha256'}}
    body.update(version=CONDITION_VERSION, operations=deepcopy(CONDITION_DEFINITIONS),
                definitions_sha256=canonical_hash(CONDITION_DEFINITIONS),
                underlying_protocol=FACTORIAL_VERSION,
                condition_mapping='E=0 maps to B=C=0; E=1 maps to B=C=1.',
                repetition_unit='declared seed IDs; paired block IDs require a separate block manifest')
    body['jobs'] = [
        {'variant': variant, 'seed': seed,
         'factors': {'A': patch['factorial']['factors']['A'], 'E': patch['factorial']['factors']['C']},
         'factorial_factors': deepcopy(patch['factorial']['factors']),
         'patch': patch, 'bindings': legacy['bindings'],
         'output': str(Path(output) / variant / f'seed_{seed}')}
        for variant, patch in condition_plans() for seed in legacy['seed_ids']]
    return {**body, 'protocol_sha256': canonical_hash(body)}
