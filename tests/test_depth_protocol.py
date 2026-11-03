from copy import deepcopy
import importlib.util
import unittest
import numpy as np

from tefed.depth import clean_depth, normalize_depth, convert_rendered_depth, axial_depth_from_positions


class DepthProtocolTests(unittest.TestCase):
    def test_two_meters_equals_dataset_2000_mm_without_double_conversion(self):
        raw = {'depth': np.array([[2., 100.]], dtype=np.float64)}
        converted = convert_rendered_depth(raw, 1000.)
        self.assertEqual(converted['depth'].dtype, np.float32)
        np.testing.assert_array_equal(converted['depth'], [[2000., 100000.]])
        np.testing.assert_array_equal(convert_rendered_depth(converted, 1000.)['depth'], converted['depth'])
        normalized, _ = normalize_depth(converted['depth'], 65535.)
        self.assertAlmostEqual(float(normalized[0, 0]), 2000 / 65535, places=7)
        self.assertEqual(float(normalized[0, 1]), 1.)
        np.testing.assert_array_equal(raw['depth'], [[2., 100.]])
        with self.assertRaises(ValueError):
            convert_rendered_depth(converted, 1.)

    def test_invalid_depth_and_explicit_mask_stay_zero(self):
        depth = np.array([[1., 2., 0., -1., np.nan, np.inf]])
        valid = np.array([[False, True, True, True, True, True]])
        values, mask = clean_depth(depth, valid)
        np.testing.assert_array_equal(values, [[0, 2, 0, 0, 0, 0]])
        np.testing.assert_array_equal(mask, [[False, True, False, False, False, False]])
        for bad in [0, -1, True, np.inf, np.nan]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_depth([[1]], bad)
        with self.assertRaises(ValueError):
            clean_depth([[1]], [[True, False]])

    def test_world_position_export_handles_camera_translation_scale_and_rotation(self):
        camera = np.eye(4)
        camera[:3, 3] = [0, 0, 5]
        camera[:3, 2] *= 3  # Camera object scaling does not rescale metric Z.
        z, _ = axial_depth_from_positions([[[4., 0, 3.], [0, 0, 6.]]], camera)
        np.testing.assert_array_equal(z, [[2., 0.]])
        camera = np.array([[0., 0, 1, 3], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        z, _ = axial_depth_from_positions([[[1., 8, 0]]], camera, meters_per_unit=.01)
        self.assertAlmostEqual(float(z[0, 0]), .02, places=7)

    def test_export_masks_background_and_rejects_invalid_geometry(self):
        positions = np.array([[[0., 0, -2], [np.nan, 0, -2]]])
        z, valid = axial_depth_from_positions(positions, np.eye(4), [[False, True]])
        np.testing.assert_array_equal(z, [[0, 0]])
        self.assertFalse(valid.any())
        with self.assertRaises(ValueError):
            axial_depth_from_positions(positions, np.eye(3))
        with self.assertRaises(ValueError):
            axial_depth_from_positions(positions, np.eye(4), meters_per_unit=0)

    def test_unseen_geometry_keeps_depth_mask_registered(self):
        from tefed.data import apply_unseen
        depth = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        mask = np.ones_like(depth, dtype=bool)
        mask[0, 1] = False
        sample = {'rgb': np.zeros((3, 4, 3), np.uint8), 'depth': depth,
                  'depth_validity': mask, 'nir': None, 'targets': []}
        result = apply_unseen(sample, perturbation={'horizontal_flip': True, 'translate_x': .25}, identity='x', seed=0)
        expected = np.zeros_like(depth)
        expected[:, 1:] = np.where(mask, depth, 0)[:, ::-1][:, :-1]
        np.testing.assert_array_equal(result['depth'], expected)
        np.testing.assert_array_equal(result['depth_validity'], expected > 0)
        np.testing.assert_array_equal(sample['depth'], depth)
        np.testing.assert_array_equal(sample['depth_validity'], mask)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native training dependencies')
    def test_network_input_uses_mask_and_fixed_scale_for_both_sources(self):
        from tefed.model import sample_tensor
        import torch
        base = {'rgb': np.zeros((2, 2, 3), np.uint8), 'depth': np.full((2, 2), 2000., np.float32),
                'depth_scale': 65535., 'depth_validity': np.array([[True, False], [True, True]]), 'nir': None}
        original = deepcopy(base)
        rendered = convert_rendered_depth({**base, 'depth': np.full((2, 2), 2.)}, 1000.)
        torch.testing.assert_close(sample_tensor(base), sample_tensor(rendered))
        self.assertEqual(float(sample_tensor(base)[0, 3, 0, 1]), 0.)
        np.testing.assert_array_equal(base['depth'], original['depth'])


if __name__ == '__main__':
    unittest.main()
