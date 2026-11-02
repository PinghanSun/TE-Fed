"""ProxyFL ICPL and GPT for native YOLO class convolutions."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from math import isfinite

import torch
from torch import nn
import torch.nn.functional as F


def _classifiers(model):
    """Return classifier names, branches, levels and convolution modules."""
    try:
        detect = model.heads[0].detect
        branches = [('one2many', detect.one2many)]
        if detect.end2end:
            branches.append(('one2one', detect.one2one))
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError('ProxyFL requires an unfused native general detection head') from exc
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for branch, components in branches:
        for level, tower in enumerate(components['cls_head']):
            final = tower[-1]
            if not isinstance(final, nn.Conv2d) or final.kernel_size != (1, 1):
                raise ValueError('ProxyFL requires final 1x1 classification convolutions')
            if final.out_channels != detect.nc or detect.nc < 2:
                raise ValueError('ProxyFL requires at least two consistent detector classes')
            name = names.get(id(final.weight))
            if name is None:
                raise ValueError('Native class proxy is not a named model parameter')
            result.append((name, branch, level, final))
    if not result:
        raise ValueError('Native classification matrices are missing')
    return result


def class_proxy_parameters(model):
    """Return general-head class weights keyed by full parameter names."""
    return {name: final.weight for name, _, _, final in _classifiers(model)}


def _state_parameter(state, name):
    if not isinstance(state, Mapping):
        raise ValueError('Client model state must be a mapping')
    if name in state:
        return state[name]
    if name.startswith('heads.'):
        _, index, suffix = name.split('.', 2)
        try:
            return state['heads'][index][suffix]
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Client class proxy is missing: {name}') from exc
    raise ValueError(f'Client class proxy is missing: {name}')


def optimize_global_proxies(model, client_states, *, steps=10, learning_rate=.005):
    """Tune class weights against equally weighted client/class rows.

    Start from the sample-weighted detector average and fix its adaptive margin.
    Each step uses every client/class row. Biases and other parameters stay fixed.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('ProxyFL server steps must be a positive integer')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('ProxyFL server learning rate must be finite and positive')
    clients = list(client_states)
    if not clients:
        raise ValueError('ProxyFL requires at least one client model state')
    parameters = class_proxy_parameters(model)
    anchors, labels, margins = {}, {}, {}
    for name, parameter in parameters.items():
        classes = parameter.shape[0]
        rows = []
        for state in clients:
            value = torch.as_tensor(_state_parameter(state, name), device=parameter.device, dtype=torch.float32).detach()
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid client class proxy: {name}')
            rows.append(value.flatten(1).clone())
        anchors[name] = torch.cat(rows)
        labels[name] = torch.arange(classes, device=parameter.device).repeat(len(clients))
        initial = parameter.detach().float().flatten(1)
        if not torch.isfinite(initial).all():
            raise ValueError(f'Invalid global class proxy: {name}')
        distances = torch.cdist(initial, initial)
        distances.fill_diagonal_(float('inf'))
        margins[name] = float(distances.amin(1).amax().clamp(max=100).item())

    def objective(name, parameter):
        logits = -torch.cdist(anchors[name], parameter.float().flatten(1))
        margin = F.one_hot(labels[name], num_classes=parameter.shape[0]).to(logits.dtype)
        return F.cross_entropy(logits - margins[name] * margin, labels[name])

    flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    optimizer = torch.optim.SGD(list(parameters.values()), lr=float(learning_rate))
    initial_losses = {name: float(objective(name, parameter).detach()) for name, parameter in parameters.items()}
    try:
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            # Sum matrix losses to preserve the learning rate for each matrix.
            loss = sum(objective(name, parameter) for name, parameter in parameters.items())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ProxyFL server objective')
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters.values()):
                raise FloatingPointError('Nonfinite ProxyFL server gradients')
            optimizer.step()
        final_losses = {name: float(objective(name, parameter).detach()) for name, parameter in parameters.items()}
    finally:
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            parameter.requires_grad_(flags[name])
    count = len(parameters)
    return {'steps': steps, 'learning_rate': float(learning_rate), 'classifier_count': count,
            'client_count': len(clients), 'initial_loss': sum(initial_losses.values()) / count,
            'final_loss': sum(final_losses.values()) / count, 'margins': margins,
            'matrix_initial_losses': initial_losses, 'matrix_final_losses': final_losses}


@contextmanager
def _frozen_batchnorm(model):
    normalizers = [(module, module.training) for module in model.modules()
                   if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in normalizers:
            module.eval()
        yield
    finally:
        for module, training in normalizers:
            module.train(training)


def _class_feature_forward(model, x, classifiers):
    features, hooks = {}, []
    try:
        for _, _, level, final in classifiers:
            def capture(module, inputs, level=level):
                features[level] = inputs[0]
            hooks.append(final.register_forward_pre_hook(capture))
        raw = model(x, (0,))[0]
        raw = raw[1] if isinstance(raw, tuple) else raw
        raw = raw.get('one2many', raw)
    finally:
        for hook in hooks:
            hook.remove()
    if len(features) != len(classifiers) or not isinstance(raw, Mapping) or 'scores' not in raw:
        raise ValueError('Native detector did not expose all classification features')
    ordered = [features[level] for _, _, level, _ in classifiers]
    if any(value.ndim != 4 or value.shape[0] != 1 for value in ordered):
        raise ValueError('ProxyFL region forwards require a single image')
    if sum(value.shape[-2] * value.shape[-1] for value in ordered) != raw['scores'].shape[-1]:
        raise ValueError('Native proposal indices differ from classification features')
    return ordered, raw['scores'][0].float().T


def pool_contrastive_loss(features, positive_targets, possible_classes, anchor_mask, *, return_details=False):
    """Compute contrastive cross-entropy within one native feature space.

    Targets use normalized class rows; low-confidence weighted targets are
    not renormalized. Negatives require disjoint class sets and cosine >= 1e-6.
    Exclude self-pairs, including empty class sets.
    """
    if features.ndim != 2 or positive_targets.shape != features.shape:
        raise ValueError('Pool features and positive targets must share an N-by-D shape')
    if possible_classes.ndim != 2 or possible_classes.shape[0] != len(features) or possible_classes.shape[1] < 2:
        raise ValueError('Possible-class pools require an N-by-C matrix with C >= 2')
    if anchor_mask.shape != (len(features),):
        raise ValueError('Pool anchor mask must have one flag per feature')
    if not torch.isfinite(features).all() or not torch.isfinite(positive_targets).all():
        raise ValueError('Pool features and positive targets must be finite')
    z = F.normalize(features.float(), dim=1, eps=1e-8)
    pools = possible_classes.to(device=z.device, dtype=torch.bool)
    anchors = anchor_mask.to(device=z.device, dtype=torch.bool)
    count = int(anchors.sum().item())
    if not count:
        loss = features.sum() * 0 + positive_targets.sum() * 0
        details = {'anchor_count': 0, 'negative_pair_count': 0}
        return (loss, details) if return_details else loss
    similarities = z @ z.T
    negatives = (pools.float() @ pools.float().T == 0) & (similarities.detach() >= 1e-6)
    negatives.fill_diagonal_(False)
    positive = (z * positive_targets.float()).sum(1)
    logits = torch.cat([positive[anchors, None], similarities[anchors].masked_fill(~negatives[anchors], -torch.inf)], dim=1)
    loss = (torch.logsumexp(logits, dim=1) - positive[anchors]).mean()
    details = {'anchor_count': count, 'negative_pair_count': int(negatives[anchors].sum().item())}
    return (loss, details) if return_details else loss


def _probabilities(value, classes, device, name):
    result = torch.as_tensor(value, device=device, dtype=torch.float32).detach()
    if result.shape != (classes,) or not torch.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError(f'{name} must contain one finite nonnegative probability per class')
    return result / result.sum()


def proxy_objective(model, batch, config, prior, *, strong_view, round_id, epoch):
    """Return unweighted ICPL loss and pool statistics.

    Frozen-global proxy_candidates contain foreground score, class posterior
    and native one-to-many grid index. Strong views preserve geometry;
    prior is the preceding global class distribution.
    """
    from .model import sample_tensor

    classifiers = [entry for entry in _classifiers(model) if entry[1] == 'one2many']
    classes = classifiers[0][3].out_channels
    device = classifiers[0][3].weight.device
    options = config.get('baselines', {}).get('proxyfl', {})
    threshold = float(options.get('pseudo_threshold', .95))
    floor = float(options.get('proposal_floor', .05))
    if not isfinite(threshold) or not 0 <= threshold <= 1 or not isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('ProxyFL probability thresholds must lie in [0, 1]')
    class_prior = _probabilities([1 / classes] * classes if prior is None else prior, classes, device, 'Class prior')
    counts = [0] * classes
    stats = {'labeled_count': 0, 'high_count': 0, 'low_count': 0, 'anchor_count': 0,
             'proposal_count': 0, 'negative_pair_count': 0, 'class_counts': counts, 'loss': 0.}
    items = []
    for item in batch:
        if item['kind'] not in {'labeled', 'pseudo'}:
            raise ValueError('ProxyFL expects labeled or pseudo training items')
        candidates = []
        if item['kind'] == 'labeled':
            for target in item['sample']['targets']:
                label = target['class_id']
                if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < classes:
                    raise ValueError('Annotated ProxyFL class is outside the detector classes')
                counts[label] += 1
                stats['labeled_count'] += 1
        else:
            for candidate in item.get('proxy_candidates', ()):
                score = float(candidate['score'])
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('ProxyFL proposal score must lie in [0, 1]')
                if score >= floor:
                    candidates.append(candidate)
        stats['proposal_count'] += len(candidates)
        items.append((item, candidates))
    zero = classifiers[0][3].weight.sum() * 0
    if not stats['proposal_count']:
        return zero, stats
    levels = [{'features': [], 'positives': [], 'pools': [], 'anchors': []} for _ in classifiers]

    def add(level, feature, positive, pool, anchor):
        group = levels[level]
        for key, value in [('features', feature), ('positives', positive), ('pools', pool), ('anchors', anchor)]:
            group[key].append(value)

    with _frozen_batchnorm(model):
        for item, candidates in items:
            sample = item['sample']
            if item['kind'] == 'pseudo' and not candidates:
                continue
            x = sample_tensor(sample, device, rgb_only=config.get('model', {}).get('detector_rgb_only', False))
            weak, logits = _class_feature_forward(model, x, classifiers)
            if item['kind'] == 'labeled':
                for level, feature_map in enumerate(weak):
                    height, width = feature_map.shape[-2:]
                    for target in sample['targets']:
                        box = target['box']
                        if len(box) != 4 or not all(isfinite(float(v)) for v in box) or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                            raise ValueError('Annotated ProxyFL boxes must use normalized nondegenerate xyxy')
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        column, row = min(int(cx * width), width - 1), min(int(cy * height), height - 1)
                        feature = feature_map[0, :, row, column].float()
                        pool = F.one_hot(torch.tensor(target['class_id'], device=device), classes).bool()
                        add(level, feature, torch.zeros_like(feature), pool, False)
                continue
            augmented = strong_view(sample, round_id * 1000 + epoch)
            strong_x = sample_tensor(augmented, device, rgb_only=config.get('model', {}).get('detector_rgb_only', False))
            strong, _ = _class_feature_forward(model, strong_x, classifiers)
            if [value.shape for value in weak] != [value.shape for value in strong]:
                raise ValueError('ProxyFL strong augmentation must preserve detector grid geometry')
            sizes = [value.shape[-2] * value.shape[-1] for value in weak]
            for candidate in candidates:
                index = candidate['index']
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sum(sizes):
                    raise ValueError('ProxyFL proposal index is outside the native detector grid')
                offset, level = index, 0
                while offset >= sizes[level]:
                    offset -= sizes[level]
                    level += 1
                q = _probabilities(candidate['posterior'], classes, device, 'Global proposal posterior')
                local_q = logits[index].detach().softmax(-1)
                proxies = F.normalize(classifiers[level][3].weight.float().flatten(1), dim=1, eps=1e-8)
                if bool((q.max() >= threshold) | (local_q.max() >= threshold)):
                    label = int(q.argmax().item())
                    pool = F.one_hot(torch.tensor(label, device=device), classes).bool()
                    positive = proxies[label]
                    stats['high_count'] += 1
                    counts[label] += 1
                else:
                    pool = q > class_prior
                    positive = ((q * pool).unsqueeze(1) * proxies).sum(0)
                    stats['low_count'] += 1
                for view in (weak, strong):
                    feature = view[level].flatten(2)[0, :, offset].float()
                    add(level, feature, positive, pool, True)
    terms = []
    for group in levels:
        if not any(group['anchors']):
            continue
        loss, details = pool_contrastive_loss(torch.stack(group['features']), torch.stack(group['positives']),
            torch.stack(group['pools']), torch.tensor(group['anchors'], device=device, dtype=torch.bool), return_details=True)
        terms.append(loss * details['anchor_count'])
        stats['anchor_count'] += details['anchor_count']
        stats['negative_pair_count'] += details['negative_pair_count']
    loss = sum(terms, zero) / max(1, stats['anchor_count'])
    stats['loss'] = float(loss.detach())
    return loss, stats
