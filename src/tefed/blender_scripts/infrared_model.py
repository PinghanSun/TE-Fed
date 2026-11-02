"""Band-integrated opaque-surface IR model, independent of visible RGB values."""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np


SOURCES = {
    'planck': 'https://cie.co.at/eilvterm/17-24-005',
    'spectral_materials': 'https://www.usgs.gov/labs/spectroscopy-lab/usgs-spectral-library',
    'dronevehicle': 'https://arxiv.org/pdf/2003.02437',
    'lambertian': 'https://pbr-book.org/3ed-2018/Reflection_Models/Lambertian_Reflection',
}

# These coefficients are illustrative assumptions, not downloaded library spectra.
_PROFILES = {
    'rubber': (.05, .96, 302.0),
    'metal': (.56, .30, 299.0),
    'painted_metal': (.28, .92, 303.0),
    'glass': (.08, .88, 296.0),
    'vegetation': (.48, .97, 295.0),
    'asphalt': (.13, .95, 307.0),
    'concrete': (.32, .93, 301.0),
    'soil': (.22, .95, 300.0),
    'generic': (.25, .90, 300.0),
}
_SURFACE_KEYS = {'profile', 'reflectance', 'emissivity', 'temperature_k'}


def preset_config(mode='lwir'):
    if mode not in {'nir', 'lwir'}:
        raise ValueError('mode must be nir or lwir')
    wavelength = np.linspace(.75, 1.05, 13) if mode == 'nir' else np.linspace(8, 14, 49)
    return {
        'schema_version': 1,
        'mode': mode,
        'parameter_basis': 'assumed_example',
        'sensor': {'wavelength_um': wavelength.tolist(), 'response': [1.0] * len(wavelength),
                   'gain_dn_per_radiance': 1000.0 if mode == 'nir' else 600.0,
                   'offset_dn': 0.0, 'noise_std_dn': 0.0, 'bit_depth': 16, 'seed': 0,
                   'preview_radiance_range': [0.0, 15.0] if mode == 'nir' else [40.0, 65.0]},
        'illumination': {'irradiance_w_m2_um': 300.0, 'sky_fraction': .20,
                         'sun_direction': [-.4, -.6, .7], 'cast_shadows': True},
        'environment': {'reflected_temperature_k': 290.0,
                        'atmosphere_temperature_k': 293.15, 'transmittance': 1.0},
        'profiles': {name: {'reflectance': rho, 'emissivity': eps, 'temperature_k': temp}
                     for name, (rho, eps, temp) in _PROFILES.items()},
        'materials': {}, 'objects': {},
    }


def merge_config(base, update):
    result = deepcopy(base)
    if not isinstance(update, dict):
        raise ValueError('IR configuration must be a JSON object')
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _finite(value, name, lower=None, upper=None):
    if isinstance(value, (bool, str)):
        raise ValueError(f'{name} must be numeric')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(result) or lower is not None and result < lower or upper is not None and result > upper:
        raise ValueError(f'{name} is out of range')
    return result


def spectral_values(value, wavelength_um, name, lower=0.0, upper=None):
    grid = np.asarray(wavelength_um, dtype=float)
    if isinstance(value, dict):
        if set(value) != {'wavelength_um', 'values'}:
            raise ValueError(f'{name} curve needs wavelength_um and values')
        x = np.asarray(value['wavelength_um'], dtype=float)
        y = np.asarray(value['values'], dtype=float)
        if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
            raise ValueError(f'{name} curve needs ordered, finite wavelengths and matching values')
        if x[0] > grid[0] or x[-1] < grid[-1]:
            raise ValueError(f'{name} curve does not cover the sensor band')
        if not np.isfinite(y).all() or np.any(y < lower) or upper is not None and np.any(y > upper):
            raise ValueError(f'{name} curve values are out of range')
        return np.interp(grid, x, y)
    if np.isscalar(value):
        return np.full(grid.shape, _finite(value, name, lower, upper))
    array = np.asarray(value, dtype=float)
    if array.shape != grid.shape or not np.isfinite(array).all() or np.any(array < lower) or upper is not None and np.any(array > upper):
        raise ValueError(f'{name} needs finite values matching the sensor wavelengths')
    return array


def validate_config(config):
    result = deepcopy(config)
    allowed = {'schema_version', 'mode', 'parameter_basis', 'calibration_reference', 'sensor',
               'illumination', 'environment', 'profiles', 'materials', 'objects'}
    if set(result) - allowed:
        raise ValueError(f'Unknown IR configuration keys: {sorted(set(result) - allowed)}')
    if result.get('schema_version') != 1 or result.get('mode') not in {'nir', 'lwir'}:
        raise ValueError('IR schema_version must be 1 and mode must be nir or lwir')
    nested_keys = {
        'sensor': {'wavelength_um', 'response', 'gain_dn_per_radiance', 'offset_dn',
                   'noise_std_dn', 'bit_depth', 'seed', 'preview_radiance_range'},
        'illumination': {'irradiance_w_m2_um', 'sky_fraction', 'sun_direction', 'cast_shadows'},
        'environment': {'reflected_temperature_k', 'atmosphere_temperature_k', 'transmittance'},
    }
    for name, fields in nested_keys.items():
        if not isinstance(result.get(name), dict) or set(result[name]) != fields:
            raise ValueError(f'{name} requires exactly these fields: {sorted(fields)}')
    if 'calibration_reference' in result:
        reference = result['calibration_reference']
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'} or not isinstance(reference['path'], str) or not reference['path'].strip() or not isinstance(reference['sha256'], str) or not re.fullmatch('[a-fA-F0-9]{64}', reference['sha256']):
            raise ValueError('calibration_reference requires a file path and 64-character SHA256')
    sensor = result['sensor']
    grid = np.asarray(sensor['wavelength_um'], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or len(grid) > 4096 or not np.isfinite(grid).all() or grid[0] <= 0 or not np.all(np.diff(grid) > 0):
        raise ValueError('sensor wavelengths must be positive, finite and strictly increasing')
    if result['mode'] == 'nir' and (grid[0] < .7 or grid[-1] > 2.5):
        raise ValueError('nir wavelengths must lie within 0.7-2.5 micrometres')
    if result['mode'] == 'lwir' and (grid[0] < 7 or grid[-1] > 20):
        raise ValueError('lwir wavelengths must lie within 7-20 micrometres')
    response = spectral_values(sensor['response'], grid, 'sensor.response', upper=1)
    if not np.any(response > 0):
        raise ValueError('sensor.response must have positive support')
    sensor['wavelength_um'], sensor['response'] = grid.tolist(), response.tolist()
    for name in ('gain_dn_per_radiance', 'noise_std_dn'):
        sensor[name] = _finite(sensor[name], 'sensor.' + name, 0)
    if sensor['gain_dn_per_radiance'] == 0:
        raise ValueError('sensor gain must be positive')
    sensor['offset_dn'] = _finite(sensor['offset_dn'], 'sensor.offset_dn')
    if type(sensor['seed']) is not int or not 0 <= sensor['seed'] < 2**64:
        raise ValueError('sensor seed must be an unsigned 64-bit integer')
    if type(sensor['bit_depth']) is not int or not 8 <= sensor['bit_depth'] <= 16:
        raise ValueError('sensor bit_depth must be an integer from 8 through 16')
    window = sensor['preview_radiance_range']
    if len(window) != 2 or not 0 <= _finite(window[0], 'preview minimum') < _finite(window[1], 'preview maximum'):
        raise ValueError('preview_radiance_range must have increasing nonnegative endpoints')
    light, env = result['illumination'], result['environment']
    spectral_values(light['irradiance_w_m2_um'], grid, 'illumination.irradiance_w_m2_um')
    light['sky_fraction'] = _finite(light['sky_fraction'], 'illumination.sky_fraction', 0, 1)
    direction = np.asarray(light['sun_direction'], dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
        raise ValueError('sun_direction must be a nonzero finite 3-vector')
    light['sun_direction'] = (direction / np.linalg.norm(direction)).tolist()
    if type(light['cast_shadows']) is not bool:
        raise ValueError('cast_shadows must be boolean')
    spectral_values(env['transmittance'], grid, 'environment.transmittance', upper=1)
    for name in ('reflected_temperature_k', 'atmosphere_temperature_k'):
        env[name] = _finite(env[name], 'environment.' + name, 1, 5000)
    if not isinstance(result['profiles'], dict) or 'generic' not in result['profiles']:
        raise ValueError('profiles must include generic')
    for name, surface in result['profiles'].items():
        _validate_surface(surface, grid, 'profiles.' + name)
    for group in ('materials', 'objects'):
        if not isinstance(result[group], dict):
            raise ValueError(group + ' must map exact names to surface overrides')
        for name, override in result[group].items():
            if not isinstance(name, str) or not name or not isinstance(override, dict) or set(override) - _SURFACE_KEYS:
                raise ValueError(group + ' overrides contain invalid names or fields')
            if override.get('profile', 'generic') not in result['profiles']:
                raise ValueError(f'Unknown profile in {group}.{name}')
            _validate_surface(override, grid, group + '.' + name, partial=True)
    return result


def _validate_surface(surface, grid, name, partial=False):
    if not isinstance(surface, dict) or set(surface) - _SURFACE_KEYS:
        raise ValueError(name + ' contains unknown surface fields')
    for key in ('reflectance', 'emissivity'):
        if key in surface:
            spectral_values(surface[key], grid, name + '.' + key, upper=1)
        elif not partial:
            raise ValueError(name + ' needs ' + key)
    if 'temperature_k' in surface:
        _finite(surface['temperature_k'], name + '.temperature_k', 1, 5000)
    elif not partial:
        raise ValueError(name + ' needs temperature_k')


def classify_material(material_name):
    name = material_name.lower()
    for profile, tokens in (
        ('glass', ('glass', 'window', 'mirror_lens')),
        ('rubber', ('rubber', 'tire', 'tyre', 'tread', 'sidewall')),
        ('vegetation', ('foliage', 'leaves', 'hedge')),
        ('asphalt', ('asphalt', 'road_')),
        ('soil', ('soil', 'mulch', 'bark')),
        ('concrete', ('concrete', 'paving', 'plaster')),
        ('painted_metal', ('paint', 'marking', 'traffic_', 'door', 'cladding')),
        ('metal', ('metal', 'alloy', 'fastener', 'roof', 'seam', 'rib', 'grille')),
    ):
        if any(token in name for token in tokens):
            return profile
    return 'generic'


def resolve_surface(config, material_name, object_name='', object_properties=None, material_properties=None):
    overrides = {}
    for properties in (material_properties or {}, object_properties or {}):
        if 'tefed_ir_material' in properties:
            overrides['profile'] = properties['tefed_ir_material']
        overrides.update({key: properties['ir_' + key] for key in _SURFACE_KEYS if 'ir_' + key in properties})
    overrides.update(config['materials'].get(material_name, {}))
    overrides.update(config['objects'].get(object_name, {}))
    profile = overrides.get('profile', classify_material(material_name))
    if profile not in config['profiles']:
        raise ValueError('Unknown IR material profile: ' + str(profile))
    result = dict(config['profiles'][profile])
    result.update(overrides)
    result['profile'] = profile
    _validate_surface(result, config['sensor']['wavelength_um'], object_name + '/' + material_name)
    return result


def integrate(values, wavelength_um, response):
    x, y = np.asarray(wavelength_um, dtype=float), np.asarray(values, dtype=float) * np.asarray(response, dtype=float)
    return float(np.sum((y[..., 1:] + y[..., :-1]) * np.diff(x) * .5, axis=-1))


def planck_radiance(wavelength_um, temperature_k):
    """Spectral radiance in W m^-2 sr^-1 um^-1; exact SI constants."""
    wavelength = np.asarray(wavelength_um, dtype=float) * 1e-6
    temperature = _finite(temperature_k, 'temperature_k', 1, 5000)
    if np.any(wavelength <= 0) or not np.isfinite(wavelength).all():
        raise ValueError('wavelengths must be finite and positive')
    exponent = 6.62607015e-34 * 299792458 / (wavelength * 1.380649e-23 * temperature)
    with np.errstate(over='ignore'):
        return 2 * 6.62607015e-34 * 299792458**2 / wavelength**5 / np.expm1(exponent) * 1e-6


def surface_coefficients(config, surface):
    """Return band-radiance terms; NIR = ambient + direct*cos(theta)*visibility."""
    sensor, env, light = config['sensor'], config['environment'], config['illumination']
    grid, response = sensor['wavelength_um'], sensor['response']
    tau = spectral_values(env['transmittance'], grid, 'transmittance', upper=1)
    path = integrate((1 - tau) * planck_radiance(grid, env['atmosphere_temperature_k']), grid, response)
    if config['mode'] == 'nir':
        rho = spectral_values(surface['reflectance'], grid, 'reflectance', upper=1)
        irradiance = spectral_values(light['irradiance_w_m2_um'], grid, 'irradiance')
        full = integrate(tau * rho * irradiance / math.pi, grid, response)
        return {'ambient': full * light['sky_fraction'] + path,
                'direct': full * (1 - light['sky_fraction'])}
    epsilon = spectral_values(surface['emissivity'], grid, 'emissivity', upper=1)
    emitted = epsilon * planck_radiance(grid, surface['temperature_k'])
    reflected = (1 - epsilon) * planck_radiance(grid, env['reflected_temperature_k'])
    return {'thermal': integrate(tau * (emitted + reflected), grid, response) + path}


def background_radiance(config):
    sensor, env, light = config['sensor'], config['environment'], config['illumination']
    grid, response = sensor['wavelength_um'], sensor['response']
    tau = spectral_values(env['transmittance'], grid, 'transmittance', upper=1)
    if config['mode'] == 'nir':
        background = spectral_values(light['irradiance_w_m2_um'], grid, 'irradiance') * light['sky_fraction'] / math.pi
    else:
        background = planck_radiance(grid, env['reflected_temperature_k'])
    path = (1 - tau) * planck_radiance(grid, env['atmosphere_temperature_k'])
    return integrate(tau * background + path, grid, response)


def sensor_readout(radiance, config):
    array = np.asarray(radiance, dtype=float)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError('Radiance must be finite and nonnegative')
    sensor = config['sensor']
    linear = array * sensor['gain_dn_per_radiance'] + sensor['offset_dn']
    if sensor['noise_std_dn']:
        linear += np.random.default_rng(sensor['seed']).normal(0, sensor['noise_std_dn'], array.shape)
    maximum = 2**sensor['bit_depth'] - 1
    dn = np.rint(np.clip(linear, 0, maximum)).astype(np.uint16)
    low, high = sensor['preview_radiance_range']
    observed = (dn.astype(float) - sensor['offset_dn']) / sensor['gain_dn_per_radiance']
    preview = np.rint(np.clip((observed - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return dn, preview, {'underflow_pixels': int(np.count_nonzero(linear < 0)),
                         'saturated_pixels': int(np.count_nonzero(linear > maximum))}
