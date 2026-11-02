"""Create a small procedural scene for checking the rendering interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


DEFAULT_CLASSES = {
    'tomato': {'non-ripe': 0, 'ripe': 1},
    'dronevehicle': {'car': 0, 'truck': 1, 'bus': 2, 'van': 3, 'freight_car': 4},
}
ROOT_NAMES = {'non-ripe': 'non_ripe'}


def read_classes(task, path=None):
    classes = dict(DEFAULT_CLASSES[task]) if path is None else json.loads(
        Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(classes, dict) or set(classes) != set(DEFAULT_CLASSES[task]):
        raise ValueError(f'{task} class map requires {sorted(DEFAULT_CLASSES[task])}')
    values = list(classes.values())
    if any(type(value) is not int for value in values) or sorted(values) != list(range(len(values))):
        raise ValueError('Class IDs must be unique consecutive integers starting at zero')
    return classes


def prepare(task, output, class_map=None, resolution=(640, 640)):
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != '.blend':
        raise ValueError('--output must name a new .blend file')
    paths = {'scene': output, 'metadata': output.with_suffix('.scene.json'),
             'request': output.with_suffix('.request.json')}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError('Choose a new scene name; scene and sidecars must not exist')
    if len(resolution) != 2 or any(type(value) is not int or not 32 <= value <= 4096 for value in resolution):
        raise ValueError('Resolution requires two integers in [32,4096]')
    classes = read_classes(task, class_map)
    roots = {ROOT_NAMES.get(name, name): value for name, value in classes.items()}
    return {'task': task, 'paths': paths, 'class_map': classes,
            'class_names': [name for name, _ in sorted(classes.items(), key=lambda row: row[1])],
            'scene_class_map': roots, 'image_size': list(resolution)}


def build_scene(plan):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene['tefed_data_origin'] = 'procedural_example'
    scene['tefed_task'] = plan['task']
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x, scene.render.resolution_y = plan['image_size']
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8

    def material(name, color, roughness=.5):
        value = bpy.data.materials.new(name)
        value.use_nodes = True
        shader = value.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = roughness
        return value

    def cube(name, location, dimensions, surface, parent=None):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        item = bpy.context.object
        item.name = name
        item.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        item.data.materials.append(surface)
        item.parent = parent
        return item

    ground = material('Ground', (.18, .20, .19))
    cube('Ground', (0, 0, -.10), (13, 11, .20), ground)
    if plan['task'] == 'tomato':
        colors = {'non-ripe': (.10, .45, .045), 'ripe': (.65, .035, .02)}
        for index, name in enumerate(('non-ripe', 'ripe')):
            root = bpy.data.objects.new(ROOT_NAMES.get(name, name), None)
            scene.collection.objects.link(root)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                                               radius=.75, location=(-1.25 + 2.5 * index, 0, .8))
            item = bpy.context.object
            item.name = root.name + '_fruit'
            item.scale = (1, 1, .92)
            item.data.materials.append(material(name, colors[name], .42))
            item.parent = root
    else:
        dimensions = {'car': (1.2, 2.4, .75), 'truck': (1.25, 3.2, 1.1),
                      'bus': (1.3, 3.8, 1.2), 'van': (1.25, 2.6, 1.1),
                      'freight_car': (1.3, 3.7, 1.25)}
        colors = [( .18, .32, .52), (.42, .25, .14), (.52, .38, .12),
                  (.29, .43, .30), (.42, .20, .23)]
        glass = material('Windows', (.055, .075, .085), .25)
        for index, name in enumerate(DEFAULT_CLASSES['dronevehicle']):
            root = bpy.data.objects.new(name, None)
            scene.collection.objects.link(root)
            x, y = (-3.2 + 3.2 * (index % 3)), (-2.5 + 5 * (index // 3))
            width, length, height = dimensions[name]
            cube(name + '_body', (x, y, height / 2), (width, length, height),
                 material(name + '_paint', colors[index]), root)
            cube(name + '_cab', (x, y - length * .16, height + .20),
                 (width * .85, length * .40, .40), glass, root)

    world = bpy.data.worlds.new('ExampleWorld')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.25, .25, .25, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .8
    scene.world = world
    lamp = bpy.data.lights.new('Key', 'AREA')
    lamp.energy, lamp.size = 1800, 8
    light = bpy.data.objects.new('Key', lamp)
    scene.collection.objects.link(light)
    light.location = (-3, -4, 9)
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data = bpy.data.cameras.new('Camera')
    camera_data.type, camera_data.ortho_scale = 'ORTHO', 13 if plan['task'] == 'dronevehicle' else 6
    camera_data.clip_start, camera_data.clip_end = .01, 1000
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, -8, 11) if plan['task'] == 'dronevehicle' else (0, -6, 4)
    camera.rotation_euler = (Vector((0, 0, .5)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()


def create(plan):
    import bpy

    paths = plan['paths']
    paths['scene'].parent.mkdir(parents=True, exist_ok=True)
    build_scene(plan)
    metadata = {key: value for key, value in plan.items() if key != 'paths'}
    metadata.update(scene_path=str(paths['scene']), data_origin='procedural_example',
                    purpose='Rendering interface check', unit_scale_metres=1,
                    camera='Camera', root_type='object', renderer_adapter='explicit_scene_class_map')
    request = {'descriptor': {'source_class': min(plan['class_map'].values()),
                             'attrs': [.4, .2, 0, .2, 0, 0, 0, 0]},
               'seed': 0, 'case_id': plan['task'] + '_example',
               'scene_path': str(paths['scene']), 'scene_class_map': plan['scene_class_map'],
               'image_size': plan['image_size']}
    published = []
    with TemporaryDirectory(prefix='.tefed-scene-', dir=paths['scene'].parent) as temporary:
        stage = Path(temporary).resolve()
        assert stage.is_relative_to(paths['scene'].parent)
        blend = stage / paths['scene'].name
        bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
        (stage / paths['metadata'].name).write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        (stage / paths['request'].name).write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
        try:
            for target in paths.values():
                with target.open('xb') as output:
                    published.append(target)
                    with (stage / target.name).open('rb') as source:
                        shutil.copyfileobj(source, output)
        except BaseException:
            for target in reversed(published):
                target.unlink()
            raise
    return metadata


def main(argv=None):
    args = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=tuple(DEFAULT_CLASSES))
    parser.add_argument('--output', required=True, help='New .blend path, with .scene.json and .request.json sidecars')
    parser.add_argument('--class-map', help='Category name to class ID JSON; match the training manifest')
    parser.add_argument('--resolution', nargs=2, type=int, default=(640, 640), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--check', action='store_true', help='Validate arguments without Blender or file creation')
    options = parser.parse_args(args)
    try:
        plan = prepare(options.task, options.output, options.class_map, options.resolution)
        result = {key: value for key, value in plan.items() if key != 'paths'} if options.check else create(plan)
    except ModuleNotFoundError as error:
        if error.name != 'bpy':
            raise
        parser.error('Run scene creation with Blender: blender --background --python this_script -- ...')
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
