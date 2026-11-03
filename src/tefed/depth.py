"""Depth geometry and the fixed DEPTH-MM-v1 input convention."""
from __future__ import annotations

import math
import numpy as np


def clean_depth(values, validity=None):
    """Return float32 positive finite depths and an explicit validity mask."""
    depth = np.asarray(values, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Depth must be a single-channel image')
    valid = np.isfinite(depth) & (depth > 0)
    if validity is not None:
        mask = np.asarray(validity, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError('Depth validity mask must match depth shape')
        valid &= mask
    return np.where(valid, depth, 0).astype(np.float32), valid


def normalize_depth(values, scale, validity=None):
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('A fixed positive depth_scale is required')
    depth, valid = clean_depth(values, validity)
    return np.clip(depth / float(scale), 0, 1).astype(np.float32), valid


def convert_rendered_depth(sample, conversion):
    """Convert rendered scene depth once, without integer casting or mutation."""
    result = dict(sample)
    result['depth_unit_conversion'] = conversion
    if conversion is None:
        result['depth'] = None
        result['depth_validity'] = None
        result['depth_units_status'] = 'unknown_scene_conversion_depth_omitted'
        return result
    if isinstance(conversion, bool) or not math.isfinite(float(conversion)) or float(conversion) <= 0:
        raise ValueError('depth_unit_conversion must be positive and finite')
    conversion = float(conversion)
    applied = result.get('_depth_unit_conversion_applied')
    if applied is not None and not math.isclose(float(applied), conversion, rel_tol=0, abs_tol=1e-12):
        raise ValueError('sample was already converted using a different depth unit factor')
    if result.get('depth') is not None:
        depth, valid = clean_depth(result['depth'], result.get('depth_validity'))
        if applied is None:
            with np.errstate(over='ignore', invalid='ignore'):
                depth = depth * conversion
        result['depth'], result['depth_validity'] = clean_depth(depth, valid)
    result['_depth_unit_conversion_applied'] = conversion
    result['depth_units_status'] = 'converted_to_dataset_units'
    return result


def axial_depth_from_positions(positions, camera_to_world, validity=None, *, meters_per_unit=1.0):
    """Project world positions onto the camera axis, including orthographic cameras.

    Blender cameras look along local -Z. Camera object scale does not change
    the metric distance. A Position pass avoids confusing ray range with Z.
    """
    points = np.asarray(positions, dtype=np.float64)
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('World positions must have shape H x W x 3')
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Camera transform must be a finite 4 x 4 matrix')
    if isinstance(meters_per_unit, bool) or not math.isfinite(float(meters_per_unit)) or meters_per_unit <= 0:
        raise ValueError('meters_per_unit must be positive and finite')
    axis = -matrix[:3, 2]
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError('Camera forward axis must be nonzero')
    z = (points - matrix[:3, 3]) @ (axis / norm) * float(meters_per_unit)
    valid = np.isfinite(points).all(axis=-1)
    if validity is not None:
        mask = np.asarray(validity, dtype=bool)
        if mask.shape != z.shape:
            raise ValueError('Depth validity mask must match positions')
        valid &= mask
    return clean_depth(z, valid)
