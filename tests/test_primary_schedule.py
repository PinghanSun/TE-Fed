from copy import deepcopy
import unittest
import importlib.util
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tefed.baselines import policy_dict, validate_baseline_config, fedsto_round_trainable, orthogonality_penalty


class ToyDetector(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))])
        self.heads = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        self.backbone_layers = 1

    def forward(self, x):
        for layer in self.trunk:
            x = layer(x)
        return self.heads[0](x)


class PrimaryScheduleTests(unittest.TestCase):
    def config(self):
        return {'experiment': {'rounds': 1000, 'local_epochs': 2}, 'baselines': {'fedsto': {
            'schedule': 'round_stages', 'supervised_rounds': 50, 'backbone_only_until': 150}}}

    def test_stage_boundaries_and_long_run_continuation(self):
        config = self.config()
        for r, stage in [(0, 'supervised'), (49, 'supervised'), (50, 'backbone'), (149, 'backbone'), (150, 'full'), (299, 'full'), (999, 'full')]:
            for epoch in [0, 1]:
                p = policy_dict('fedsto', r, epoch, config)
                self.assertEqual(p['fedsto_stage'], stage)
                self.assertEqual(p['use_unlabeled'], stage != 'supervised')
                self.assertEqual(p['orthogonal_coefficient'], .05 if stage == 'full' else 0.)
                self.assertEqual(p['alignment_weight'], .1)
                self.assertEqual(p['alignment_scope'], 'labeled')

    @unittest.skipUnless(torch is not None, 'Requires native training dependencies')
    def test_frozen_neck_and_head_keep_backbone_gradient_and_bn_state(self):
        torch.manual_seed(8)
        model = ToyDetector().train()
        before = deepcopy(model.state_dict())
        fedsto_round_trainable(model, 'backbone', [0])
        optim = torch.optim.SGD(model.parameters(), lr=.1)
        model(torch.randn(8, 2)).square().mean().backward()
        self.assertIsNotNone(model.trunk[0].weight.grad)
        self.assertIsNone(model.trunk[1][0].weight.grad)
        self.assertIsNone(model.heads[0].weight.grad)
        optim.step()
        self.assertFalse(torch.equal(model.trunk[0].weight, before['trunk.0.weight']))
        for name in ['trunk.1.0.weight', 'trunk.1.1.running_mean', 'heads.0.weight']:
            torch.testing.assert_close(model.state_dict()[name], before[name])
        model.train()
        fedsto_round_trainable(model, 'full', [0])
        self.assertTrue(model.trunk[1][1].training)
        self.assertTrue(model.heads[0].weight.requires_grad)
        self.assertFalse(model.heads[1].weight.requires_grad)
        self.assertTrue(torch.isfinite(orthogonality_penalty(model, active_heads=[0], include_neck=True)))

    def test_supervised_warmup_and_legacy_schedule_are_distinct(self):
        for method in ['confidence', 'consistent_teacher_fedawa', 'fsslod', 'proxyfl']:
            config = {'experiment': {'rounds': 300, 'local_epochs': 2}, 'baselines': {method: {'supervised_rounds': 5}}}
            self.assertFalse(policy_dict(method, 4, 0, config)['use_unlabeled'])
            self.assertTrue(policy_dict(method, 5, 0, config)['use_unlabeled'])
        self.assertTrue(policy_dict('fedsto', 0, 0)['freeze_trunk'])
        self.assertFalse(policy_dict('fedsto', 0, 1)['freeze_trunk'])

    @unittest.skipUnless(torch is not None and importlib.util.find_spec('ultralytics'), 'Requires native training dependencies')
    def test_native_optimizer_respects_backbone_stage_then_unfreezes(self):
        from tefed.engine import train_local
        from tefed.model import RoutedYOLO
        torch.set_num_threads(2)
        torch.manual_seed(3)
        config = {'model': {'architecture': 'yolo26n.yaml', 'classes': 2, 'specialists': 0},
                  'experiment': {'rounds': 3, 'local_epochs': 1, 'batch_size': 1},
                  'optimizer': {'amp': False, 'learning_rate': .001, 'warmup_rounds': 0},
                  'baselines': {'fedsto': {'schedule': 'round_stages', 'supervised_rounds': 1, 'backbone_only_until': 2}}}
        model = RoutedYOLO(config)
        sample = {'rgb': np.full((64, 64, 3), 128, dtype=np.uint8), 'depth': None, 'nir': None,
                  'targets': [{'box': [.2, .2, .7, .8], 'class_id': 0}]}
        items = [{'kind': 'labeled', 'sample': sample, 'route_weights': {0: 1.}}]
        neck = model.trunk[model.backbone_layers:]
        before_neck = deepcopy(neck.state_dict())
        before_head = deepcopy(model.heads.state_dict())
        before_backbone = model.trunk[0].conv.weight.detach().clone()
        first = train_local(model, items, config, method='fedsto', round_id=1, reference=deepcopy(model))
        self.assertEqual(first['optimizer_steps'], 1)
        self.assertTrue(first['stage_history'][0]['freeze_neck'])
        self.assertFalse(torch.equal(before_backbone, model.trunk[0].conv.weight))
        for key, value in neck.state_dict().items():
            torch.testing.assert_close(value, before_neck[key], rtol=0, atol=0)
        for key, value in model.heads.state_dict().items():
            torch.testing.assert_close(value, before_head[key], rtol=0, atol=0)
        second = train_local(model, items, config, method='fedsto', round_id=2, reference=deepcopy(model))
        self.assertEqual(second['optimizer_steps'], 1)
        self.assertFalse(second['stage_history'][0]['freeze_neck'])
        self.assertTrue(all(p.requires_grad for p in neck.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_conflicting_or_reversed_schedule_rejected(self):
        for changes in [{'head_only_epochs': 1}, {'backbone_only_until': 50}, {'supervised_rounds': True}, {'schedule': []}]:
            config = self.config()
            config['baselines']['fedsto'].update(changes)
            with self.assertRaises(ValueError):
                validate_baseline_config(config)


if __name__ == '__main__':
    unittest.main()
