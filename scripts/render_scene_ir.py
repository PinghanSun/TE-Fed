"""Render an optional material-based NIR or LWIR diagnostic from a Blender scene."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np


def load_model():
    root = Path(__file__).resolve().parent.parent
    candidates = (root / 'src' / 'tefed' / 'blender_scripts' / 'infrared_model.py',
                  root / 'renderer' / 'infrared_model.py')
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError('infrared_model.py not found; expected src/tefed/blender_scripts/ or renderer/ beside the script directory')
    spec = importlib.util.spec_from_file_location('tefed_infrared_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def write_gray_png(path, array):
    values = np.asarray(array)
    if values.ndim != 2 or values.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError('PNG input must be a uint8 or uint16 grayscale array')
    depth = 8 if values.dtype == np.uint8 else 16
    encoded = values if depth == 8 else values.astype('>u2')
    raw = b''.join(b'\x00' + row.tobytes() for row in encoded)

    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)

    header = struct.pack('>IIBBBBB', values.shape[1], values.shape[0], depth, 0, 0, 0, 0)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def arguments(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=('nir', 'lwir'), default='lwir')
    parser.add_argument('--config', type=Path, help='JSON overrides, including response and material spectra')
    parser.add_argument('--output', type=Path, help='Fresh output directory')
    parser.add_argument('--scene', default='ParkingLot')
    parser.add_argument('--camera', help='Camera object; defaults to the scene camera')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, help='Default: retain the saved render aspect ratio')
    parser.add_argument('--samples', type=int, choices=(1, 4), default=1, help='Deterministic subpixel rays per pixel')
    parser.add_argument('--with-rgb', action='store_true', help='Also render CPU Cycles RGB with the same camera and geometry')
    parser.add_argument('--rgb-samples', type=int, default=16)
    parser.add_argument('--write-preset', type=Path, help='Write the complete configuration without Blender')
    parser.add_argument('--check', action='store_true', help='Validate configuration without Blender or output creation')
    args = parser.parse_args(argv)
    if not 32 <= args.width <= 4096 or args.height is not None and not 32 <= args.height <= 4096:
        parser.error('Image dimensions must lie within 32-4096')
    if not 1 <= args.rgb_samples <= 256:
        parser.error('RGB samples must lie within 1-256')
    if not args.output and not args.check and not args.write_preset:
        parser.error('--output is required for rendering')
    if args.output:
        args.output = args.output.expanduser().resolve()
        if args.output.exists() or args.output.is_symlink():
            parser.error('Choose a fresh output directory')
    return args


def custom_properties(owner):
    if owner is None:
        return {}
    keys = ('tefed_ir_material', 'ir_profile', 'ir_reflectance', 'ir_emissivity', 'ir_temperature_k')
    return {key: owner[key].to_list() if hasattr(owner[key], 'to_list') else owner[key]
            for key in keys if key in owner}


def parameter_provenance(base, overrides):
    def leaves(value, prefix=''):
        if isinstance(value, dict):
            return {key for name, child in value.items() for key in leaves(child, prefix + name + '.')}
        return {prefix[:-1]}

    supplied = leaves(overrides)
    defaults = leaves(base)
    return {'user_override_keys': sorted(supplied),
            'assumed_default_keys': sorted(defaults - supplied)}


def verify_calibration_reference(config, config_path):
    reference = config.get('calibration_reference')
    if not reference:
        return None
    path = Path(reference['path']).expanduser()
    if not path.is_absolute():
        path = (config_path.parent if config_path else Path.cwd()) / path
    path = path.resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != reference['sha256'].lower():
        raise ValueError('calibration_reference SHA256 does not match its file')
    return {'file': path.name, 'sha256': actual, 'file_verified': True}


def object_record(obj, index):
    ancestor = obj
    while ancestor is not None and 'tefed_class_id' not in ancestor:
        ancestor = ancestor.parent
    record = {'index': index, 'name': obj.name}
    if ancestor is not None:
        record.update(root=ancestor.name, class_name=ancestor.get('tefed_class_name'),
                      class_id=ancestor['tefed_class_id'])
    return record


def build_geometry(scene, config, model):
    import bpy
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    visible_names = set()

    def collect_visible(layer):
        if layer.exclude or layer.collection.hide_render:
            return
        visible_names.update(obj.name for obj in layer.collection.objects)
        for child in layer.children:
            collect_visible(child)

    collect_visible(bpy.context.view_layer.layer_collection)
    vertices, triangles, triangle_surfaces, triangle_objects = [], [], [], []
    surfaces, objects = [], []
    for obj in scene.objects:
        if obj.name not in visible_names or obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'} or obj.hide_render or not getattr(obj, 'visible_camera', True):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                continue
            object_index = len(objects)
            objects.append(object_record(obj, object_index))
            start = len(vertices)
            vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
            material_indices = {}
            for triangle in mesh.loop_triangles:
                slot = triangle.material_index
                if slot not in material_indices:
                    material = mesh.materials[slot] if slot < len(mesh.materials) else None
                    name = material.name if material else '(none)'
                    properties = model.resolve_surface(config, name, obj.name, custom_properties(obj), custom_properties(material))
                    surface_index = len(surfaces)
                    material_indices[slot] = surface_index
                    surfaces.append({'index': surface_index, 'object': obj.name, 'material': name,
                                     'properties': properties,
                                     'coefficients': model.surface_coefficients(config, properties)})
                triangles.append(tuple(start + vertex for vertex in triangle.vertices))
                triangle_surfaces.append(material_indices[slot])
                triangle_objects.append(object_index)
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise ValueError('The chosen scene contains no visible surface geometry')
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    extent = max(max(vertex[axis] for vertex in vertices) - min(vertex[axis] for vertex in vertices) for axis in range(3))
    return tree, triangle_surfaces, triangle_objects, surfaces, objects, max(extent * 1e-6, 1e-6)


def render_ir(scene, camera, config, width, height, samples, model):
    from mathutils import Vector

    tree, tri_surfaces, tri_objects, surfaces, objects, epsilon = build_geometry(scene, config, model)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    frame = camera.data.view_frame(scene=scene)
    xmin, xmax = min(v.x for v in frame), max(v.x for v in frame)
    ymin, ymax = min(v.y for v in frame), max(v.y for v in frame)
    frame_z = frame[0].z
    rotation, translation = camera.matrix_world.to_quaternion(), camera.matrix_world.translation
    camera_forward = rotation @ Vector((0, 0, -1))
    parallel = camera.data.type == 'ORTHO'
    if not parallel and camera.data.type != 'PERSP':
        raise ValueError('IR stage supports perspective and orthographic cameras')
    sun = Vector(config['illumination']['sun_direction'])
    sky = model.background_radiance(config)
    radiance = np.full((height, width), sky, dtype=np.float32)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    material_ids = np.full((height, width), -1, dtype=np.int32)
    object_ids = np.full((height, width), -1, dtype=np.int32)
    offsets = ((.5, .5),) if samples == 1 else ((.25, .25), (.75, .25), (.25, .75), (.75, .75))
    unit_scale = scene.unit_settings.scale_length
    for row in range(height):
        for column in range(width):
            total, hits = 0.0, []
            for dx, dy in offsets:
                x = xmin + (column + dx) / width * (xmax - xmin)
                y = ymax - (row + dy) / height * (ymax - ymin)
                if parallel:
                    origin = translation + rotation @ Vector((x, y, 0))
                    direction = rotation @ Vector((0, 0, -1))
                else:
                    origin = translation
                    direction = rotation @ Vector((x, y, frame_z)).normalized()
                cosine_to_axis = direction.dot(camera_forward)
                near_range = camera.data.clip_start / cosine_to_axis
                far_range = camera.data.clip_end / cosine_to_axis
                start = origin + direction * near_range
                location, normal, face, distance = tree.ray_cast(start, direction, far_range - near_range)
                if location is None:
                    total += sky
                    continue
                if normal.dot(direction) > 0:
                    normal = -normal
                surface_index = tri_surfaces[face]
                coefficients = surfaces[surface_index]['coefficients']
                if config['mode'] == 'lwir':
                    value = coefficients['thermal']
                else:
                    cosine = max(0.0, normal.dot(sun))
                    visible = True
                    if cosine and config['illumination']['cast_shadows']:
                        blocker = tree.ray_cast(location + normal * epsilon + sun * epsilon, sun)
                        visible = blocker[0] is None
                    value = coefficients['ambient'] + coefficients['direct'] * cosine * visible
                total += value
                hits.append((surface_index, tri_objects[face], (distance + near_range) * unit_scale))
            radiance[row, column] = total / samples
            if hits:
                chosen = Counter((hit[0], hit[1]) for hit in hits).most_common(1)[0][0]
                selected = [hit[2] for hit in hits if hit[:2] == chosen]
                material_ids[row, column], object_ids[row, column] = chosen
                depth[row, column] = sum(selected) / len(selected)
        if (row + 1) % max(1, height // 8) == 0:
            print(f'IR rows {row + 1}/{height}', flush=True)
    camera_record = {'name': camera.name, 'type': camera.data.type,
                     'matrix_world': [list(row) for row in camera.matrix_world],
                     'ortho_scale': camera.data.ortho_scale, 'lens_mm': camera.data.lens,
                     'sensor_width_mm': camera.data.sensor_width, 'clip_start': camera.data.clip_start,
                     'clip_end': camera.data.clip_end}
    return radiance, depth, material_ids, object_ids, surfaces, objects, camera_record


def select_scene_camera(args):
    import bpy

    if not bpy.data.filepath:
        raise ValueError('Open a scene .blend before running this script')
    scene = bpy.data.scenes.get(args.scene)
    if scene is None:
        raise ValueError('Scene not found: ' + args.scene)
    camera = scene.objects.get(args.camera) if args.camera else scene.camera
    if camera is None or camera.type != 'CAMERA':
        raise ValueError('The scene needs the requested camera')
    if camera.data.type not in {'ORTHO', 'PERSP'}:
        raise ValueError('IR stage supports perspective and orthographic cameras')
    height = args.height or max(32, round(args.width * scene.render.resolution_y * scene.render.pixel_aspect_y /
                                        (scene.render.resolution_x * scene.render.pixel_aspect_x)))
    if height > 4096:
        raise ValueError('Derived height exceeds 4096')
    return scene, camera, height


def main(argv=None):
    args = arguments(argv)
    model, model_path = load_model()
    base = model.preset_config(args.preset)
    overrides = {}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if isinstance(overrides, dict) and overrides.get('mode', args.preset) != args.preset:
        raise ValueError('Configuration mode differs from --preset; select the matching preset')
    config = model.merge_config(base, overrides)
    config['parameter_basis'] = 'assumed_preset_with_user_overrides' if overrides else 'assumed_example'
    config = model.validate_config(config)
    provenance = parameter_provenance(base, overrides)
    calibration = verify_calibration_reference(config, args.config)
    if args.write_preset:
        if args.write_preset.exists() or args.write_preset.is_symlink():
            raise FileExistsError('Preset output already exists')
        args.write_preset.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    if args.check or args.write_preset:
        checked = {'configuration_valid': True, 'mode': config['mode'], 'scene_checked': False}
        if args.check:
            try:
                import bpy
            except ModuleNotFoundError as error:
                if error.name != 'bpy':
                    raise
            else:
                if bpy.data.filepath:
                    scene, camera, height = select_scene_camera(args)
                    checked.update(scene_checked=True, source_blend=str(Path(bpy.data.filepath).resolve()),
                                   scene=scene.name, camera=camera.name, image_size=[args.width, height])
        print(json.dumps({'configuration_valid': True, 'config': config,
                          'parameter_provenance': provenance, 'calibration_reference_check': calibration}, indent=2))
        print('IR_CHECK_RESULT ' + json.dumps(checked))
        return 0
    import bpy

    scene, camera, height = select_scene_camera(args)
    source = Path(bpy.data.filepath).resolve(strict=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.context.window.scene = scene
    scene.camera = camera
    frame_settings = {'full_frame': True, 'square_pixels': True,
                      'saved_render_border_disabled': bool(scene.render.use_border),
                      'saved_crop_to_border_disabled': bool(scene.render.use_crop_to_border),
                      'clipping': 'camera-axis near and far planes'}
    started = time.perf_counter()
    results = render_ir(scene, camera, config, args.width, height, args.samples, model)
    radiance, depth, material_ids, object_ids, surfaces, objects, camera_record = results
    dn, preview, clipping = model.sensor_readout(radiance, config)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in (('radiance', radiance), ('depth_m', depth), ('material_ids', material_ids), ('object_ids', object_ids), ('ir_dn', dn)):
        np.save(args.output / (name + '.npy'), values, allow_pickle=False)
    write_gray_png(args.output / 'ir.png', dn)
    write_gray_png(args.output / 'ir_preview.png', preview)
    rgb_metadata = None
    if args.with_rgb:
        scene.render.engine = 'CYCLES'
        scene.cycles.device, scene.cycles.samples = 'CPU', args.rgb_samples
        scene.cycles.use_denoising = True
        if hasattr(scene.render, 'compositor_device'):
            scene.render.compositor_device = 'CPU'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode, scene.render.image_settings.color_depth = 'RGB', '8'
        scene.render.filepath = str(args.output / 'rgb.png')
        bpy.ops.render.render(write_still=True)
        rgb_metadata = {'image': 'rgb.png', 'engine': 'Cycles CPU', 'samples': args.rgb_samples,
                        'camera_matches_ir': True, 'geometry_matches_ir': True,
                        'illumination': 'stored Blender scene lighting; IR uses its separate spectral configuration'}
    metadata = {
        'schema_version': 1, 'method': 'opaque_material_spectral_ray_baseline',
        'mode': config['mode'], 'parameter_basis': config['parameter_basis'],
        'parameter_provenance': provenance, 'calibration_reference_check': calibration,
        'dataset_sensor_match': None, 'radiance_unit': 'W m^-2 sr^-1 (response-weighted band integral)',
        'range_definition': 'depth_m.npy is camera-ray range to first surface, not camera-axis Z; misses are NaN',
        'source_blend': source.name, 'source_sha256': source_hash,
        'source_unchanged': hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
        'generator': Path(__file__).name, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'blender_version': bpy.app.version_string,
        'scene': scene.name, 'camera': camera_record, 'image_size': [args.width, height],
        'frame_settings': frame_settings,
        'rgb': rgb_metadata,
        'rays_per_pixel': args.samples, 'seconds': round(time.perf_counter() - started, 3),
        'config': config, 'surfaces': surfaces, 'objects': objects, 'clipping': clipping,
        'limitations': ['opaque Lambertian surfaces; no transmission or directional BRDF',
                        'assigned temperatures; no heat-transfer solution',
                        'direct shadows and isotropic sky; no interreflection',
                        'one common spectral irradiance shape for sun and sky',
                        'mesh evaluation includes modifiers; collection instancing is not expanded'],
        'sources': model.SOURCES,
    }
    (args.output / 'ir.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'mode': config['mode'], 'image_size': metadata['image_size'],
                      'seconds': metadata['seconds'], 'source_unchanged': metadata['source_unchanged']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
