"""Fixed BatchNorm priors from an explicitly declared validation input pool.

The caller selects validation records and turns them into tensors; this module
does not load data or labels. Moments are collected at actual BN inputs with the
initial model in eval mode. Downstream BN inputs therefore use the same frozen
initial normalization, independent of validation batch partitioning.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import random


def _identities(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError('An explicit nonempty validation_ids sequence is required')
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError('Validation identities must be nonempty strings or integers')
        ids.append(str(value))
    if len(set(ids)) != len(ids):
        raise ValueError('validation_ids must be unique')
    return sorted(ids)


def _identity_hash(ids):
    value = json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def _bindings(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) or not k or
                                            not isinstance(v, str) or not v for k, v in value.items()):
        raise ValueError('BN prior binding must map nonempty strings to strings')
    for key, item in value.items():
        if key.endswith('_sha256') and (len(item) != 64 or any(c not in '0123456789abcdef' for c in item)):
            raise ValueError(f'BN prior binding {key} must be a lowercase SHA256')
    return dict(sorted(value.items()))


def _validate_merged_sources(protocol, parent_binding):
    merged = 'aggregation' in protocol or 'source_priors' in protocol
    if not merged:
        return
    if protocol.get('aggregation') != 'parallel_channel_moments_from_disjoint_validation_pools':
        raise ValueError('Merged BN prior aggregation protocol is invalid')
    sources = protocol.get('source_priors')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Merged BN prior requires nonempty source_priors')
    hashes, bindings = set(), []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {'validation_ids_sha256', 'binding'}:
            raise ValueError('Merged BN prior source schema differs')
        digest = source['validation_ids_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Merged BN prior source validation hash must be lowercase SHA256')
        if digest in hashes:
            raise ValueError('Merged BN prior source validation hashes must be unique')
        hashes.add(digest)
        bindings.append(_bindings(source['binding']))
    for key in ['initial_checkpoint_sha256', 'initial_model_sha256', 'preprocessing_sha256']:
        values = [binding.get(key) for binding in bindings]
        if key in parent_binding or any(value is not None for value in values):
            if key not in parent_binding or any(value != parent_binding[key] for value in values):
                raise ValueError(f'Merged BN prior source binding differs from its parent: {key}')


def _normalizers(model):
    import torch
    modules = {name: layer for name, layer in model.named_modules()
               if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)}
    if not modules:
        raise ValueError('A validation BN prior requires at least one BatchNorm module')
    for name, layer in modules.items():
        if not layer.track_running_stats or layer.running_mean is None or layer.running_var is None:
            raise ValueError(f'BatchNorm {name!r} cannot carry a fixed running-statistics prior')
    return modules


def _state_key(module_name, field):
    return f'{module_name}.{field}' if module_name else field


def build_validation_bn_prior(model, batches, *, validation_ids, forward=None,
                              source_split='validation', binding=None, require_all=True):
    """Return CPU BN state and provenance without changing model or random state.

    ``batches`` yields prepared N,C,... tensors from validation only, exactly
    once per declared identity. ``forward(model, batch)`` may select a detector
    branch; by default ``model(batch)`` is called. All BN modules must be visited
    unless ``require_all=False`` explicitly requests a partial prior. Unvisited
    modules are listed and never receive invented or fallback statistics.

    Affine parameters come from this initial model. Running variance is unbiased
    over all observed channel activations, not an average of batch variances.
    num_batches_tracked is one for this single pooled population, so repartitioning
    the same observations does not change any stored BN tensor.
    """
    import numpy as np
    import torch
    if source_split != 'validation':
        raise ValueError('BN prior inputs must come only from validation')
    if not isinstance(require_all, bool):
        raise ValueError('require_all must be boolean')
    ids = _identities(validation_ids)
    bound = _bindings(binding)
    modules = _normalizers(model)
    if forward is not None and not callable(forward):
        raise ValueError('forward must be callable')
    run_forward = forward or (lambda current, batch: current(batch))
    original_state = deepcopy(model.state_dict())
    modes = [(layer, layer.training) for layer in model.modules()]
    params = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone())
              for p in model.parameters()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    moments = {}
    handles = []
    batch_count = input_count = 0

    def observe(name, layer, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError(f'BatchNorm {name!r} did not receive a tensor')
        values = args[0].detach()
        if values.ndim < 2 or values.shape[1] != layer.num_features or values.numel() == 0:
            raise ValueError(f'BatchNorm {name!r} received invalid channel dimensions')
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f'BatchNorm {name!r} validation activations must be finite')
        axes = (0, *range(2, values.ndim))
        count = values.numel() // layer.num_features
        variance, mean = torch.var_mean(values.to(torch.float64), dim=axes, unbiased=False)
        mean, m2 = mean.cpu(), (variance * count).cpu()
        if name in moments:
            old_count, old_mean, old_m2 = moments[name]
            total = old_count + count
            delta = mean - old_mean
            mean = old_mean + delta * (count / total)
            m2 = old_m2 + m2 + delta.square() * (old_count * count / total)
            count = total
        moments[name] = (count, mean, m2)

    try:
        model.eval()
        for parameter, _, _, _ in params:
            parameter.requires_grad_(False)
        for name, layer in modules.items():
            handles.append(layer.register_forward_pre_hook(
                lambda current, args, name=name: observe(name, current, args)))
        with torch.no_grad():
            for batch in batches:
                if not isinstance(batch, torch.Tensor) or batch.ndim < 2 or batch.shape[0] < 1:
                    raise ValueError('Validation batches must be nonempty N,C,... tensors')
                if not bool(torch.isfinite(batch).all()):
                    raise ValueError('Validation inputs must be finite')
                batch_count += 1
                input_count += int(batch.shape[0])
                run_forward(model, batch)
        if input_count != len(ids):
            raise ValueError('Validation tensor sample count does not match validation_ids')
        visited = sorted(moments)
        unvisited = sorted(set(modules) - set(visited))
        if not visited:
            raise ValueError('Validation forward did not visit any BatchNorm module')
        if require_all and unvisited:
            raise ValueError(f'Validation forward did not visit BatchNorm modules: {unvisited}')
        state = {}
        for name in visited:
            layer = modules[name]
            count, mean, m2 = moments[name]
            if count < 2:
                raise ValueError(f'BatchNorm {name!r} has fewer than two validation activations per channel')
            variance = m2 / (count - 1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
                raise ValueError(f'BatchNorm {name!r} pooled moments are nonfinite')
            for field, value in layer.state_dict().items():
                key = _state_key(name, field)
                source = original_state[key].detach().cpu()
                if field == 'running_mean':
                    source = mean.to(source.dtype)
                elif field == 'running_var':
                    source = variance.to(source.dtype)
                elif field == 'num_batches_tracked':
                    source = torch.ones_like(source)
                state[key] = source.clone()
        result = {'schema_version': 1, 'state': state,
                  'moments': {name: {'count': moments[name][0], 'mean': moments[name][1].clone(),
                                     'm2': moments[name][2].clone()} for name in visited}, 'protocol': {
            'rule': 'validation_activation_pooled_unbiased_v1', 'source_split': 'validation',
            'forward_mode': 'eval_with_frozen_initial_parameters_and_bn',
            'affine_source': 'initial_model', 'num_batches_tracked_policy': 'one_pooled_population',
            'validation_ids': ids, 'validation_ids_sha256': _identity_hash(ids),
            'input_count': input_count, 'batch_count': batch_count,
            'activation_counts': {name: moments[name][0] for name in visited},
            'visited_modules': visited, 'unvisited_modules': unvisited, 'binding': bound}}
        validate_bn_prior(model, result, require_all=require_all, validation_ids=ids, binding=bound)
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.load_state_dict(original_state, strict=True)
        for layer, training in modes:
            layer.training = training
        for parameter, requires_grad, gradient, saved_gradient in params:
            parameter.requires_grad_(requires_grad)
            if gradient is not None:
                gradient.copy_(saved_gradient)
            parameter.grad = gradient
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def validate_bn_prior(model, prior, *, require_all=True, validation_ids=None, binding=None):
    """Check the complete declared BN scope, tensor schema and optional binding."""
    import torch
    modules = _normalizers(model)
    if not isinstance(prior, Mapping) or prior.get('schema_version') != 1:
        raise ValueError('Unsupported validation BN prior schema')
    state, protocol, moments = prior.get('state'), prior.get('protocol'), prior.get('moments')
    if not isinstance(state, Mapping) or not isinstance(protocol, Mapping) or not isinstance(moments, Mapping):
        raise ValueError('Validation BN prior requires state, moments and protocol mappings')
    if protocol.get('rule') != 'validation_activation_pooled_unbiased_v1' or protocol.get('source_split') != 'validation':
        raise ValueError('BN prior must declare the validation pooled-moments protocol')
    if protocol.get('forward_mode') != 'eval_with_frozen_initial_parameters_and_bn' or \
       protocol.get('affine_source') != 'initial_model' or \
       protocol.get('num_batches_tracked_policy') != 'one_pooled_population':
        raise ValueError('BN prior fixed-forward/affine protocol differs')
    ids = _identities(protocol.get('validation_ids'))
    if protocol.get('validation_ids_sha256') != _identity_hash(ids) or type(protocol.get('input_count')) is not int or protocol.get('input_count') != len(ids):
        raise ValueError('BN prior validation identity binding is invalid')
    if type(protocol.get('batch_count')) is not int or protocol['batch_count'] < 1:
        raise ValueError('BN prior batch count is invalid')
    if validation_ids is not None and ids != _identities(validation_ids):
        raise ValueError('BN prior belongs to a different validation identity pool')
    recorded_binding = _bindings(protocol.get('binding'))
    _validate_merged_sources(protocol, recorded_binding)
    if binding is not None and recorded_binding != _bindings(binding):
        raise ValueError('BN prior input/checkpoint binding differs')
    visited, unvisited = protocol.get('visited_modules'), protocol.get('unvisited_modules')
    if not isinstance(visited, list) or not visited or any(not isinstance(n, str) for n in visited) or len(set(visited)) != len(visited):
        raise ValueError('BN prior visited module list is invalid')
    if not isinstance(unvisited, list) or any(not isinstance(n, str) for n in unvisited) or len(set(unvisited)) != len(unvisited):
        raise ValueError('BN prior unvisited module list is invalid')
    if set(visited) & set(unvisited) or set(visited) | set(unvisited) != set(modules):
        raise ValueError('BN prior module scope differs from the model')
    if require_all and unvisited:
        raise ValueError(f'Incomplete validation BN prior; unvisited modules: {unvisited}')
    expected = {_state_key(name, field): value for name in visited for field, value in modules[name].state_dict().items()}
    if set(state) != set(expected):
        raise ValueError('BN prior state keys differ from its declared module scope')
    counts = protocol.get('activation_counts')
    if not isinstance(counts, Mapping) or set(counts) != set(visited) or any(type(n) is not int or n < 2 for n in counts.values()):
        raise ValueError('BN prior activation counts are invalid')
    if set(moments) != set(visited):
        raise ValueError('BN prior moment modules differ from the declared scope')
    for name in visited:
        moment = moments[name]
        if not isinstance(moment, Mapping) or set(moment) != {'count', 'mean', 'm2'} or \
           type(moment['count']) is not int or moment['count'] != counts[name]:
            raise ValueError(f'BN prior moments/counts differ: {name}')
        for field in ['mean', 'm2']:
            value = moment[field]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != 'cpu' or \
               tuple(value.shape) != (modules[name].num_features,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f'BN prior {field} moments must be finite CPU float64 channel vectors: {name}')
        if bool((moment['m2'] < 0).any()):
            raise ValueError(f'BN prior M2 must be nonnegative: {name}')
    for key, reference in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f'BN prior tensor shape/dtype differs: {key}')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'BN prior tensor is nonfinite: {key}')
        if key.endswith('running_var') and bool((value < 0).any()):
            raise ValueError(f'BN prior variance must be nonnegative: {key}')
        if key.endswith('num_batches_tracked') and value.item() != 1:
            raise ValueError('BN prior must represent one pooled validation population')
    for name in visited:
        for field, value in [('running_mean', moments[name]['mean']),
                             ('running_var', moments[name]['m2'] / (moments[name]['count'] - 1))]:
            key = _state_key(name, field)
            if not torch.equal(state[key].cpu(), value.to(expected[key].dtype)):
                raise ValueError(f'BN prior state does not match its pooled moments: {key}')
    return prior


def apply_bn_prior(model, prior, *, require_all=True, validation_ids=None, binding=None):
    """Install a validated fixed prior, without silently retaining missing BN keys."""
    validate_bn_prior(model, prior, require_all=require_all, validation_ids=validation_ids, binding=binding)
    model.load_state_dict(prior['state'], strict=False)


def merge_validation_bn_priors(model, priors, *, binding=None, require_all=True):
    """Pool worker validation moments on the server without transferring images.

    Workers must evaluate the same initial model and BN scope. Validation IDs
    must be disjoint. Per-worker manifests may differ; supplied ``binding``
    describes the merged artifact. A declared initial-checkpoint hash must agree
    across workers and with the merged binding when supplied. Affine tensors are
    checked against the server's initial model, never averaged across workers.
    """
    import torch
    values = list(priors)
    if not values:
        raise ValueError('At least one worker validation BN prior is required')
    bound = _bindings(binding)
    for prior in values:
        validate_bn_prior(model, prior, require_all=require_all)
    values.sort(key=lambda prior: prior['protocol']['validation_ids_sha256'])
    first = values[0]
    visited = first['protocol']['visited_modules']
    modules = _normalizers(model)
    original = model.state_dict()
    identities, pooled = set(), {}
    for key in ['initial_checkpoint_sha256', 'initial_model_sha256', 'preprocessing_sha256']:
        worker_values = [prior['protocol']['binding'].get(key) for prior in values]
        if any(value is not None for value in worker_values) and len(set(worker_values)) != 1:
            raise ValueError(f'Worker BN priors use different initial checkpoint or preprocessing bindings: {key}')
        if key in bound and worker_values[0] != bound[key]:
            raise ValueError(f'Merged BN prior initial checkpoint binding differs from workers: {key}')
        if key not in bound and worker_values[0] is not None:
            bound[key] = worker_values[0]
    for prior in values:
        protocol = prior['protocol']
        if protocol['visited_modules'] != visited or protocol['unvisited_modules'] != first['protocol']['unvisited_modules']:
            raise ValueError('Worker BN prior scopes differ')
        ids = set(protocol['validation_ids'])
        if identities & ids:
            raise ValueError('Worker validation identity pools overlap')
        identities.update(ids)
        for name in visited:
            for field in ['weight', 'bias']:
                key = _state_key(name, field)
                if key in prior['state'] and not torch.equal(prior['state'][key].cpu(), original[key].detach().cpu()):
                    raise ValueError(f'Worker BN affine differs from the server initial model: {key}')
            item = prior['moments'][name]
            count, mean, m2 = item['count'], item['mean'], item['m2']
            if name in pooled:
                old_count, old_mean, old_m2 = pooled[name]
                total = old_count + count
                delta = mean - old_mean
                mean = old_mean + delta * (count / total)
                m2 = old_m2 + m2 + delta.square() * (old_count * count / total)
                count = total
            pooled[name] = (count, mean.clone(), m2.clone())
    state = {}
    for name in visited:
        count, mean, m2 = pooled[name]
        for field in modules[name].state_dict():
            key = _state_key(name, field)
            value = original[key].detach().cpu().clone()
            if field == 'running_mean': value = mean.to(value.dtype)
            elif field == 'running_var': value = (m2 / (count - 1)).to(value.dtype)
            elif field == 'num_batches_tracked': value = torch.ones_like(value)
            state[key] = value
    ids = sorted(identities)
    protocol = deepcopy(first['protocol'])
    protocol.update(validation_ids=ids, validation_ids_sha256=_identity_hash(ids), input_count=len(ids),
                    batch_count=sum(prior['protocol']['batch_count'] for prior in values),
                    activation_counts={name: pooled[name][0] for name in visited}, binding=bound,
                    aggregation='parallel_channel_moments_from_disjoint_validation_pools',
                    source_priors=[{'validation_ids_sha256': prior['protocol']['validation_ids_sha256'],
                                    'binding': deepcopy(prior['protocol']['binding'])} for prior in values])
    result = {'schema_version': 1, 'state': state, 'protocol': protocol,
              'moments': {name: {'count': pooled[name][0], 'mean': pooled[name][1], 'm2': pooled[name][2]}
                          for name in visited}}
    validate_bn_prior(model, result, require_all=require_all)
    return result


def _transport_protocol(prior):
    if not isinstance(prior, Mapping) or set(prior) != {'schema_version', 'state', 'moments', 'protocol'}:
        raise ValueError('BN prior transport envelope keys differ')
    if type(prior['schema_version']) is not int or prior['schema_version'] != 1 or \
       not isinstance(prior['state'], Mapping) or not isinstance(prior['moments'], Mapping) or \
       not isinstance(prior['protocol'], dict):
        raise ValueError('BN prior transport schema is invalid')
    try:
        # Metadata remains JSON data; do not recursively turn lists or IDs into tensors.
        json.dumps(prior['protocol'], allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('BN prior protocol must contain plain finite JSON data') from error
    return deepcopy(prior['protocol'])


def bn_prior_to_arrays(prior):
    """Convert only BN tensors and channel moments to safe transport arrays."""
    import torch
    protocol = _transport_protocol(prior)
    def array(value):
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError('BN prior transport tensors must be finite')
        return value.detach().cpu().numpy().copy()
    moments = {}
    for name, value in prior['moments'].items():
        if not isinstance(value, Mapping) or set(value) != {'count', 'mean', 'm2'} or type(value['count']) is not int:
            raise ValueError('BN prior transport moment fields differ')
        moments[name] = {'count': value['count'], 'mean': array(value['mean']), 'm2': array(value['m2'])}
    return {'schema_version': 1, 'state': {key: array(value) for key, value in prior['state'].items()},
            'moments': moments, 'protocol': protocol}


def prior_from_arrays(prior):
    """Restore transport state/moments tensors; protocol stays an ordinary dict."""
    import numpy as np
    import torch
    protocol = _transport_protocol(prior)
    def tensor(value):
        if not isinstance(value, np.ndarray) or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
            raise ValueError('BN prior transport values must be finite numeric NumPy arrays')
        return torch.from_numpy(value.copy())
    moments = {}
    for name, value in prior['moments'].items():
        if not isinstance(value, Mapping) or set(value) != {'count', 'mean', 'm2'} or type(value['count']) is not int:
            raise ValueError('BN prior transport moment fields differ')
        moments[name] = {'count': value['count'], 'mean': tensor(value['mean']), 'm2': tensor(value['m2'])}
    return {'schema_version': 1, 'state': {key: tensor(value) for key, value in prior['state'].items()},
            'moments': moments, 'protocol': protocol}


bn_prior_from_arrays = prior_from_arrays
