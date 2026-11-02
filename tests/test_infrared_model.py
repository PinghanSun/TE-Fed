"""Spectral units, surface overrides and deterministic IR readout."""
import hashlib
import importlib.util
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from tefed.blender_scripts import infrared_model as ir


class InfraredModelTests(unittest.TestCase):
    def config(self, mode='lwir'):
        return ir.validate_config(ir.preset_config(mode))

    def test_planck_units_at_ten_micrometres(self):
        self.assertAlmostEqual(float(ir.planck_radiance(10, 300)), 9.924033330070701, places=9)
        self.assertGreater(float(ir.planck_radiance(10, 320)), float(ir.planck_radiance(10, 300)))

    def test_nir_reflectance_illumination_and_shadow_terms(self):
        config = self.config('nir')
        surface = ir.resolve_surface(config, 'Rubber')
        coefficients = ir.surface_coefficients(config, surface)
        full = .05 * 300 * .30 / math.pi
        self.assertAlmostEqual(coefficients['ambient'], full * .2)
        self.assertAlmostEqual(coefficients['direct'], full * .8)
        warmer = dict(surface, temperature_k=350)
        self.assertEqual(coefficients, ir.surface_coefficients(config, warmer))
        config['illumination']['irradiance_w_m2_um'] = 600
        self.assertAlmostEqual(ir.surface_coefficients(config, surface)['direct'], coefficients['direct'] * 2)

    def test_spectral_response_changes_reflectance_weighting(self):
        config = self.config('nir')
        surface = ir.resolve_surface(config, 'Rubber')
        surface['reflectance'] = {'wavelength_um': [.7, 1.1], 'values': [.1, .8]}
        config['sensor']['response'] = [1] * 6 + [0] * 7
        short = ir.surface_coefficients(config, surface)
        config['sensor']['response'] = [0] * 7 + [1] * 6
        long = ir.surface_coefficients(config, surface)
        self.assertGreater(long['direct'], short['direct'])

    def test_lwir_emission_and_environment_reflection(self):
        config = self.config()
        grid, response = config['sensor']['wavelength_um'], config['sensor']['response']
        surface = dict(ir.resolve_surface(config, 'Metal'), emissivity=1, temperature_k=320)
        emitted = ir.surface_coefficients(config, surface)['thermal']
        self.assertAlmostEqual(emitted, ir.integrate(ir.planck_radiance(grid, 320), grid, response))
        surface['emissivity'] = 0
        reflected = ir.surface_coefficients(config, surface)['thermal']
        self.assertAlmostEqual(reflected, ir.integrate(ir.planck_radiance(grid, 290), grid, response))
        self.assertGreater(emitted, reflected)

    def test_opaque_atmosphere_erases_surface_temperature(self):
        config = self.config()
        config['environment']['transmittance'] = 0
        surface = ir.resolve_surface(config, 'Paint_car')
        first = ir.surface_coefficients(config, surface)['thermal']
        surface['temperature_k'] = 800
        self.assertEqual(first, ir.surface_coefficients(config, surface)['thermal'])

    def test_opaque_atmosphere_matches_surface_and_background_in_both_bands(self):
        for mode in ['nir', 'lwir']:
            with self.subTest(mode=mode):
                config = self.config(mode)
                config['environment']['transmittance'] = 0
                coefficients = ir.surface_coefficients(config, ir.resolve_surface(config, 'Rubber'))
                surface = coefficients['thermal'] if mode == 'lwir' else coefficients['ambient']
                self.assertAlmostEqual(surface, ir.background_radiance(config), places=12)
                if mode == 'nir':
                    self.assertEqual(coefficients['direct'], 0)

    def test_material_and_object_properties_have_explicit_precedence(self):
        config = self.config()
        config['materials']['new_finish'] = {'temperature_k': 310}
        config['objects']['car_body'] = {'temperature_k': 315}
        result = ir.resolve_surface(config, 'new_finish', 'car_body',
                                    {'tefed_ir_material': 'rubber', 'ir_temperature_k': 305},
                                    {'tefed_ir_material': 'metal', 'ir_temperature_k': 300})
        self.assertEqual(result['profile'], 'rubber')
        self.assertEqual(result['temperature_k'], 315)
        self.assertEqual(ir.classify_material('Paint_vehicle_blue'), 'painted_metal')
        self.assertEqual(ir.classify_material('Env_Foliage_2'), 'vegetation')

    def test_invalid_physical_parameters_are_rejected(self):
        for update in ({'sensor': {'response': 0}},
                       {'sensor': {'wavelength_um': [9, 8]}},
                       {'profiles': {'rubber': {'emissivity': 1.1}}},
                       {'objects': {'body': {'temperature_k': float('nan')}}},
                       {'objects': {'body': {'profile': 'unregistered'}}},
                       {'environment': {'transmittance': -.1}},
                       {'sensor': {'respnose': [1, 1]}},
                       {'illumination': {'sky_fractoin': .2}},
                       {'environment': {'temprature_k': 290}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                ir.validate_config(ir.merge_config(ir.preset_config(), update))
        with self.assertRaisesRegex(ValueError, 'cover'):
            ir.spectral_values({'wavelength_um': [9, 11], 'values': [.2, .5]}, [8, 10, 14], 'spectrum')

    def test_sensor_readout_is_seeded_and_fixed_window(self):
        config = self.config()
        config['sensor'].update(gain_dn_per_radiance=100, offset_dn=2, noise_std_dn=.5, seed=123)
        radiance = np.array([[30, 40], [50, 1000]], dtype=float)
        first = ir.sensor_readout(radiance, config)
        second = ir.sensor_readout(radiance, config)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[2]['saturated_pixels'], 1)
        narrow = ir.sensor_readout(radiance[:1], config)
        np.testing.assert_array_equal(first[1][0], narrow[1][0])
        with self.assertRaises(ValueError):
            ir.sensor_readout([float('nan')], config)


class InfraredStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'render_scene_ir.py'
        spec = importlib.util.spec_from_file_location('ir_stage_test', path)
        cls.stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.stage)

    def test_configuration_check_needs_no_blender_or_output(self):
        self.assertEqual(self.stage.main(['--preset', 'nir', '--check']), 0)

    def test_grayscale_png_preserves_full_uint16_codes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'ir.png'
            values = np.array([[0, 1, 255], [256, 32768, 65535]], dtype=np.uint16)
            self.stage.write_gray_png(path, values)
            with Image.open(path) as image:
                np.testing.assert_array_equal(np.asarray(image), values)

    def test_preset_export_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'preset.json'
            self.assertEqual(self.stage.main(['--write-preset', str(path)]), 0)
            with self.assertRaises(FileExistsError):
                self.stage.main(['--write-preset', str(path)])

    def test_partial_overrides_keep_unset_values_as_assumptions(self):
        provenance = self.stage.parameter_provenance(ir.preset_config(), {'profiles': {'rubber': {'temperature_k': 310}}})
        self.assertEqual(provenance['user_override_keys'], ['profiles.rubber.temperature_k'])
        self.assertIn('profiles.rubber.emissivity', provenance['assumed_default_keys'])
        self.assertNotIn('profiles.rubber.temperature_k', provenance['assumed_default_keys'])

    def test_configuration_mode_must_match_selected_preset(self):
        import json
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'nir.json'
            path.write_text(json.dumps(ir.preset_config('nir')), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from --preset'):
                self.stage.main(['--preset', 'lwir', '--config', str(path), '--check'])

    def test_calibration_reference_is_checked_by_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'measured.json'
            reference.write_text('{"measurement_id":"example"}', encoding='utf-8')
            config = ir.preset_config()
            config['calibration_reference'] = {'path': reference.name,
                                               'sha256': hashlib.sha256(reference.read_bytes()).hexdigest()}
            checked = self.stage.verify_calibration_reference(ir.validate_config(config), root / 'config.json')
            self.assertTrue(checked['file_verified'])
            reference.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.stage.verify_calibration_reference(config, root / 'config.json')


if __name__ == '__main__':
    unittest.main()
