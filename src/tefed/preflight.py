"""Check local training inputs without constructing a model or writing outputs."""
from __future__ import annotations

import json
from pathlib import Path


def check_setup(config, *, stage='train'):
    """Collect actionable setup errors; never expose the source-ID secret."""
    from .baselines import METHOD_REGISTRY, policy_dict
    from .data import load_manifest
    from .engine import (device_for, source_secret, validate_calibration_binding,
                         validate_manifest_config, validate_renderer_setup,
                         validate_run_config)

    if stage not in {'pretrain', 'train'}:
        raise ValueError('stage must be pretrain or train')
    errors, warnings, checks = [], [], {}

    def attempt(name, callback):
        try:
            value = callback()
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            checks[name] = 'failed'
            errors.append({'check': name, 'message': str(error)})
            return None
        checks[name] = 'passed'
        return value

    attempt('configuration', lambda: validate_run_config(config))
    method = 'fedavg' if stage == 'pretrain' else config.get('experiment', {}).get('method', 'tefed')
    if method not in METHOD_REGISTRY:
        errors.append({'check': 'method', 'message': f'Unknown executable method: {method}'})
        checks['method'] = 'failed'
    else:
        checks['method'] = 'passed'
    attempt('source_id_key', lambda: source_secret(config))
    attempt('device', lambda: device_for(config))

    def read_manifest():
        manifest = load_manifest(config['experiment']['manifest'], strict_files=
                                 config.get('communication', {}).get('mode', 'local') == 'local')
        validate_manifest_config(config, manifest)
        return manifest

    manifest = attempt('manifest', read_manifest)
    if stage == 'train':
        def read_calibration():
            path = Path(config['experiment']['calibration']).expanduser()
            artifact = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(artifact, dict):
                raise ValueError('Calibration must be a JSON object')
            validate_calibration_binding(config, artifact)
        attempt('calibration_and_initial_checkpoint', read_calibration)
        if config.get('replay') and manifest is not None:
            from .replay import ReplayController
            clients = sorted({str(record.client_id) for record in manifest.by_split('labeled')})
            experiment = config.get('experiment', {})
            attempt('replay', lambda: ReplayController(config, clients,
                    int(experiment.get('rounds', 30)), int(experiment.get('seed', 0))))
        if method in METHOD_REGISTRY and not config.get('replay'):
            policy = policy_dict(method, 0, 0, config)
            attempt('renderer', lambda: validate_renderer_setup(config, policy))
            renderer = config.get('renderer', {})
            if policy['use_corrections'] and renderer.get('enabled', True) \
                    and config.get('decision', {}).get('render_budget', 1152) > 0:
                if config.get('task', 'tomato') == 'tomato' and not renderer.get('depth_unit_conversion'):
                    warnings.append('Set renderer.depth_unit_conversion from the scene and dataset units; '
                                    'rendered depth is omitted while this conversion is unspecified.')
                if config.get('task') == 'dronevehicle' and not renderer.get('scene_class_map'):
                    errors.append({'check': 'scene_class_map', 'message':
                                   'DroneVehicle rendering requires renderer.scene_class_map mapping '
                                   'scene object or collection root names to all configured class IDs.'})
                    checks['scene_class_map'] = 'failed'
    return {'status': 'failed' if errors else 'passed', 'stage': stage, 'method': method,
            'records': len(manifest.records) if manifest is not None else None,
            'checks': checks, 'errors': errors, 'warnings': warnings,
            'model_constructed': False, 'outputs_written': False}
