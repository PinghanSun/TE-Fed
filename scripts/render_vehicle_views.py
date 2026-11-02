"""Render vehicle angles and a wheel close-up from the editable asset library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def arguments(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--class-name', choices=['car', 'truck', 'bus', 'van', 'freight_car'], default='car')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--samples', type=int, default=48)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() or args.output.is_symlink():
        parser.error('Choose a fresh output directory')
    if not 640 <= args.width <= 4096 or not 1 <= args.samples <= 256:
        parser.error('Width must be 640-4096 and samples 1-256')
    return args


def main(argv=None):
    args = arguments(argv)
    import bpy
    from mathutils import Vector

    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise ValueError('Open a saved vehicle blend before rendering views')
    scene = bpy.data.scenes.get('Asset_Library')
    if scene is None:
        raise ValueError('The blend must contain the Asset_Library scene')
    bpy.context.window.scene = scene
    roots = [obj for obj in scene.objects if obj.get('tefed_class_name') == args.class_name]
    if len(roots) != 1:
        raise ValueError('Asset_Library must contain one root for the selected class')
    root = roots[0]
    for obj in scene.objects:
        if 'tefed_class_name' in obj:
            obj.hide_render = obj != root
            for child in obj.children_recursive:
                child.hide_render = obj != root
    bpy.context.view_layer.update()
    meshes = [obj for obj in root.children_recursive if obj.type == 'MESH']
    if not meshes:
        raise ValueError('The selected root has no mesh geometry')
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lower = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    center = (lower + upper) * .5
    size = upper - lower
    extent = max(size.x, size.y, size.z)

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    if hasattr(scene.render, 'compositor_device'):
        scene.render.compositor_device = 'CPU'
    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * .625)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    camera_data = bpy.data.cameras.new('PresentationCamera')
    camera = bpy.data.objects.new('PresentationCamera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = 'ORTHO'
    camera_data.clip_start, camera_data.clip_end = .01, 1000
    args.output.mkdir(parents=True, exist_ok=False)

    def render(name, target, direction, scale):
        camera.location = target + Vector(direction).normalized() * extent * 3
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera_data.ortho_scale = scale
        scene.render.filepath = str(args.output / (name + '.png'))
        bpy.ops.render.render(write_still=True)
        return {'image': name + '.png', 'camera_location': list(camera.location),
                'camera_target': list(target), 'ortho_scale_m': scale}

    views = []
    for name, direction in [('front_quarter', (5, -7, 3.3)),
                            ('side', (8, 0, 1.3)), ('rear_quarter', (5, 7, 3.3))]:
        views.append(render(name, center, direction, extent * 1.22))
    wheels = [obj for obj in meshes if 'wheel_0_1' in obj.name and 'rim' in obj.name]
    if not wheels:
        wheels = [obj for obj in meshes if 'wheel' in obj.name and 'rim' in obj.name]
    if wheels:
        wheel = wheels[0]
        wheel_points = [wheel.matrix_world @ Vector(corner) for corner in wheel.bound_box]
        wheel_center = sum(wheel_points, Vector()) / len(wheel_points)
        radius = max((point - wheel_center).length for point in wheel_points)
        views.append(render('wheel_detail', wheel_center, (5, -3, 1.3), max(1.2, radius * 5)))
    report = {'source_blend': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'generator': Path(__file__).name,
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'blender_version': bpy.app.version_string,
              'image_size': [scene.render.resolution_x, scene.render.resolution_y],
              'class_name': args.class_name,
              'root': root.name, 'source_kind': 'native_blender_render',
              'renderer': 'Cycles CPU', 'samples': args.samples, 'views': views,
              'source_blend_modified': False}
    (args.output / 'views.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
