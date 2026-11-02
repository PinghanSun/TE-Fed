"""Refine foliage, architectural surfaces and daylight in an editable vehicle scene."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys


def mesh_object(name, vertices, faces, materials, collection, indices=None):
    import bpy
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for material in materials:
        mesh.materials.append(material)
    if indices:
        for face, index in zip(mesh.polygons, indices):
            face.material_index = index
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def surface(name, color, roughness, metallic=0, fine=0, broad=0, ir_type='generic'):
    import bpy
    result = bpy.data.materials.new('Refined_' + name)
    result.use_nodes = True
    result['tefed_ir_material'] = ir_type
    nodes, links = result.node_tree.nodes, result.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    if fine or broad:
        position = nodes.new('ShaderNodeNewGeometry')
        fine_noise = nodes.new('ShaderNodeTexNoise')
        fine_noise.inputs['Scale'].default_value = fine or 2
        fine_noise.inputs['Detail'].default_value = 3
        links.new(position.outputs['Position'], fine_noise.inputs['Vector'])
        colors = nodes.new('ShaderNodeValToRGB')
        colors.color_ramp.elements[0].color = (*(c * .78 for c in color), 1)
        colors.color_ramp.elements[1].color = (*(min(c * 1.16, 1) for c in color), 1)
        links.new(fine_noise.outputs['Fac'], colors.inputs[0])
        output = colors.outputs['Color']
        if broad:
            broad_noise = nodes.new('ShaderNodeTexNoise')
            broad_noise.inputs['Scale'].default_value = broad
            broad_noise.inputs['Detail'].default_value = 2
            links.new(position.outputs['Position'], broad_noise.inputs['Vector'])
            broad_colors = nodes.new('ShaderNodeValToRGB')
            broad_colors.color_ramp.elements[0].color = (.77, .79, .80, 1)
            broad_colors.color_ramp.elements[1].color = (1, 1, 1, 1)
            links.new(broad_noise.outputs['Fac'], broad_colors.inputs[0])
            multiply = nodes.new('ShaderNodeMixRGB')
            multiply.blend_type = 'MULTIPLY'
            multiply.inputs[0].default_value = .55
            links.new(output, multiply.inputs[1])
            links.new(broad_colors.outputs['Color'], multiply.inputs[2])
            output = multiply.outputs[0]
        links.new(output, shader.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .24
        bump.inputs['Distance'].default_value = .0025 if fine > 20 else .0010
        links.new(fine_noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    return result


def add_segment(vertices, faces, start, end, radius0, radius1, sides=7):
    from mathutils import Vector
    start, end = Vector(start), Vector(end)
    axis = (end - start).normalized()
    side = axis.cross(Vector((0, 0, 1)))
    if side.length < .1:
        side = axis.cross(Vector((0, 1, 0)))
    side.normalize()
    normal = axis.cross(side).normalized()
    offset = len(vertices)
    for point, radius in ((start, radius0), (end, radius1)):
        for index in range(sides):
            angle = index * math.tau / sides
            vertices.append(tuple(point + radius * (math.cos(angle) * side + math.sin(angle) * normal)))
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append((offset + index, offset + next_index, offset + sides + next_index, offset + sides + index))
    faces.append(tuple(offset + index for index in range(sides - 1, -1, -1)))
    faces.append(tuple(offset + sides + index for index in range(sides)))


def add_leaf(vertices, faces, indices, position, length, width, yaw, pitch, color):
    from mathutils import Vector
    forward = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
    side = Vector((-math.sin(yaw), math.cos(yaw), 0))
    normal = forward.cross(side)
    points = [(-.50, 0, 0), (-.18, -.50, 0), (.25, -.36, 0), (.50, 0, 0),
              (.25, .36, 0), (-.18, .50, 0), (0, 0, .045)]
    offset = len(vertices)
    center = Vector(position)
    for x, y, z in points:
        vertices.append(tuple(center + forward * (x * length) + side * (y * width) + normal * (z * length)))
    for index in range(6):
        faces.append((offset + index, offset + (index + 1) % 6, offset + 6))
        indices.append(color)


def foliage(scene):
    import bpy
    from mathutils import Vector
    collection = bpy.data.collections['Vegetation']
    trunks = sorted((obj for obj in scene.objects if re.fullmatch(r'Env_Tree_\d+_trunk', obj.name)), key=lambda x: x.name)
    if len(trunks) != 8:
        raise ValueError('Expected the eight original tree trunks')
    trees = [(obj.name.split('_')[2], obj.location.x, obj.location.y, obj.dimensions.z / .68) for obj in trunks]
    hedges = [(obj.name, tuple(obj.location), tuple(obj.dimensions)) for obj in scene.objects if obj.name.startswith('Env_Hedge_')]
    removed = [obj for obj in list(collection.objects)
               if re.fullmatch(r'Env_Tree_\d+_(trunk|branch_\d+|leaves_\d+)', obj.name) or obj.name.startswith('Env_Hedge_')]
    for obj in removed:
        bpy.data.objects.remove(obj, do_unlink=True)
    bark = surface('Bark', (.12, .078, .042), .90, fine=12, ir_type='vegetation')
    palette = [surface(f'Leaf_{i}', color, .55, fine=85, ir_type='vegetation') for i, color in enumerate(
        ((.052, .14, .025), (.085, .205, .036), (.12, .255, .045), (.16, .29, .07), (.095, .18, .032)))]
    for material in palette:
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Subsurface Weight'].default_value = .035
        shader.inputs['Sheen Weight'].default_value = .15
    leaf_total = 0
    for index, x, y, height in trees:
        rng = random.Random(8129 + int(index))
        vertices, faces, leaf_vertices, leaf_faces, leaf_indices = [], [], [], [], []
        base = Vector((x, y, .08))
        trunk_path = [base + Vector((.065 * math.sin(z * 1.6), .045 * math.cos(z), z))
                      for z in (0, .5, 1.3, 2.25, 3.25, height * .75, height * .95)]
        for i in range(len(trunk_path) - 1):
            radius0 = .19 * (1 - i / len(trunk_path)) + .025
            radius1 = .19 * (1 - (i + 1) / len(trunk_path)) + .020
            add_segment(vertices, faces, trunk_path[i], trunk_path[i + 1], radius0, radius1, 9)
        for branch in range(11):
            angle = branch * 2.399 + rng.uniform(-.4, .4)
            level = branch / 10
            start = base + Vector((0, 0, height * (.39 + level * .40)))
            spread = (1.83 - level * .60) * rng.uniform(.82, 1.16)
            tip = start + Vector((math.cos(angle) * spread, math.sin(angle) * spread,
                                  height * rng.uniform(.10, .23)))
            bend = start.lerp(tip, .57) + Vector((0, 0, -.15))
            add_segment(vertices, faces, start, bend, .082 * (1 - level * .45), .045)
            add_segment(vertices, faces, bend, tip, .045, .014)
            for twig in range(5):
                attach = bend.lerp(tip, .24 + twig * .18)
                direction = angle + rng.uniform(-1.1, 1.1)
                end = attach + Vector((math.cos(direction) * rng.uniform(.35, .72),
                                       math.sin(direction) * rng.uniform(.35, .72), rng.uniform(.10, .45)))
                add_segment(vertices, faces, attach, end, .022, .006, 5)
                for leaf in range(96):
                    position = attach.lerp(end, rng.uniform(.2, 1.1)) + Vector((rng.gauss(0, .24), rng.gauss(0, .24), rng.gauss(0, .22)))
                    length = rng.uniform(.18, .28)
                    add_leaf(leaf_vertices, leaf_faces, leaf_indices, position, length,
                             length * rng.uniform(.40, .57), rng.uniform(0, math.tau), rng.uniform(-.85, .85), rng.randrange(len(palette)))
                    leaf_total += 1
        wood = mesh_object(f'Env_NaturalTree_{index}_branches', vertices, faces, [bark], collection)
        for face in wood.data.polygons:
            face.use_smooth = True
        mesh_object(f'Env_NaturalTree_{index}_foliage', leaf_vertices, leaf_faces, palette, collection, leaf_indices)
    for index, (name, xyz, size) in enumerate(hedges):
        rng = random.Random(4312 + index)
        vertices, faces, indices = [], [], []
        for leaf in range(2600):
            position = (xyz[0] + rng.uniform(-.53, .53) * size[0],
                        xyz[1] + rng.uniform(-.52, .52) * size[1],
                        xyz[2] + rng.uniform(-.48, .56) * size[2])
            add_leaf(vertices, faces, indices, position, rng.uniform(.12, .19), rng.uniform(.06, .10),
                     rng.uniform(0, math.tau), rng.uniform(-1, 1), rng.randrange(len(palette)))
            leaf_total += 1
        mesh_object(name + '_leaf_mesh', vertices, faces, palette, collection, indices)
    return {'trees': len(trees), 'hedges': len(hedges), 'leaf_blades': leaf_total,
            'removed_primitive_objects': len(removed), 'foliage_method': 'branch meshes and folded leaf blades', 'collection_instances': 0}


def update_surfaces(scene):
    from collections import Counter
    replacements = {
        'Env_Asphalt': surface('Asphalt', (.080, .087, .093), .86, fine=85, broad=.18, ir_type='asphalt'),
        'Env_Concrete': surface('Concrete', (.29, .30, .28), .85, fine=24, broad=.7, ir_type='concrete'),
        'Env_Paving': surface('Paving', (.38, .37, .33), .78, fine=18, broad=.5, ir_type='concrete'),
        'Env_Cladding': surface('Cladding', (.25, .32, .35), .52, .35, fine=3, broad=.18, ir_type='painted_metal'),
        'Env_Panel_seams': surface('Panel_seams', (.14, .19, .21), .50, .45, ir_type='painted_metal'),
        'Env_Roof': surface('Roof', (.19, .23, .25), .51, .55, fine=10, broad=.4, ir_type='painted_metal'),
        'Env_Roller_door': surface('Roller_door', (.43, .45, .43), .48, .38, fine=12, ir_type='painted_metal'),
        'Env_Office_plaster': surface('Office_plaster', (.42, .39, .33), .82, fine=14, broad=.15, ir_type='concrete'),
        'Env_Galvanized': surface('Galvanized', (.30, .35, .37), .34, .78, fine=18, ir_type='metal'),
    }
    windows = []
    for index, roughness in enumerate((.09, .12, .16)):
        material = surface(f'Office_glass_{index}', (.035, .075, .105), roughness, .10, ir_type='glass')
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['IOR'].default_value = 1.46
        shader.inputs['Coat Weight'].default_value = .55
        shader.inputs['Coat Roughness'].default_value = .06
        windows.append(material)
    unchanged_types = {'Env_Mulch': 'soil', 'Env_Paving_joints': 'concrete',
                       'Env_Dark_hardware': 'metal', 'Env_Traffic_white': 'generic',
                       'Env_Traffic_yellow': 'generic', 'Env_Traffic_red': 'generic'}
    counts = Counter()
    for obj in scene.objects:
        if obj.type != 'MESH' or not obj.name.startswith(('Env_', 'Site_base', 'Parking_asphalt', 'Access_road', 'Curb_', 'North_curb', 'South_curb')):
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material.name in unchanged_types:
                slot.material['tefed_ir_material'] = unchanged_types[slot.material.name]
            if slot.material and slot.material.name in replacements:
                old = slot.material.name
                slot.material = replacements[old]
                counts[old] += 1
            elif slot.material and slot.material.name == 'Env_Office_glass':
                slot.material = windows[sum(obj.name.encode()) % len(windows)]
                counts['Env_Office_glass'] += 1
    return dict(counts)


def curb_joints():
    import bpy
    collection = bpy.data.collections['Ground']
    material = surface('Curb_joint', (.12, .13, .12), .9, ir_type='concrete')
    vertices, faces = [], []
    for x in (-26.65, 26.65):
        for y in range(-15, 17, 2):
            start = len(vertices)
            vertices.extend(((x - .17, y - .007, .244), (x + .17, y - .007, .244),
                             (x + .17, y + .007, .244), (x - .17, y + .007, .244)))
            faces.append(tuple(start + i for i in range(4)))
    for y, x_start, x_end in ((16.70, -26, 27), (-16.75, -23, 6), (-16.75, 21, 27)):
        for x in range(x_start, x_end, 2):
            start = len(vertices)
            vertices.extend(((x - .007, y - .17, .244), (x + .007, y - .17, .244),
                             (x + .007, y + .17, .244), (x - .007, y + .17, .244)))
            faces.append(tuple(start + i for i in range(4)))
    mesh_object('Env_Refined_Curb_joints', vertices, faces, [material], collection)
    terrain = surface('Background_paving', (.22, .235, .21), .90, fine=18, broad=.12, ir_type='concrete')
    mesh_object('Env_Background_ground', [(-500, -500, -.115), (500, -500, -.115),
                                          (500, 500, -.115), (-500, 500, -.115)],
                [(0, 1, 2, 3)], [terrain], collection)
    return len(faces)


def daylight_and_views(scene):
    import bpy
    from mathutils import Vector
    world = bpy.data.worlds.new('Refined_Daylight')
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    sky = nodes.new('ShaderNodeTexSky')
    sky_types = {item.identifier for item in sky.bl_rna.properties['sky_type'].enum_items}
    sky.sky_type = 'MULTIPLE_SCATTERING' if 'MULTIPLE_SCATTERING' in sky_types else 'NISHITA'
    sky.sun_elevation, sky.sun_rotation = math.radians(33), math.radians(120)
    sky.sun_disc, sky.altitude, sky.air_density = False, 100 if 'MULTIPLE_SCATTERING' in sky_types else .1, 1.1
    if hasattr(sky, 'aerosol_density'):
        sky.aerosol_density = 1.3
    elif hasattr(sky, 'dust_density'):
        sky.dust_density = 1.3
    links.new(sky.outputs['Color'], nodes['Background'].inputs['Color'])
    nodes['Background'].inputs['Strength'].default_value = .18
    scene.world = world
    sun = scene.objects.get('Environment_Sun')
    sun.data.energy, sun.data.angle = 1.8, math.radians(7)
    sun.data.color = (1.0, .94, .84)
    sun.rotation_euler = (math.radians(29), math.radians(-27), math.radians(-30))
    for obj in scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'AREA':
            obj.data.energy = min(obj.data.energy, 4200)
    settings = [('Environment_Aerial', (58, -74, 54), (0, 4, 1.7), 40),
                ('Environment_Street', (34, -38, 11), (0, 6, -1.8), 32)]
    for name, position, target, lens in settings:
        obj = scene.objects[name]
        obj.location = position
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
        obj.data.type, obj.data.lens, obj.data.sensor_width = 'PERSP', lens, 36
    bind_preview_ui(scene)


def bind_preview_ui(scene):
    """Bind the opening viewport after any cross-scene validation has finished."""
    import bpy
    bpy.context.window.scene = scene
    scene.camera, scene.render.filepath = scene.objects['Camera'], '//preview.png'
    for screen in bpy.data.screens:
        if screen.name == 'Layout' or screen.name.startswith('Layout.'):
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    view = area.spaces.active
                    view.use_local_camera, view.camera = True, scene.objects['Environment_Aerial']
                    view.region_3d.view_perspective = 'CAMERA'
                    view.region_3d.view_camera_zoom, view.region_3d.view_camera_offset = 0, (0, 0)


def tag_legacy_materials():
    """Make the remaining legacy surface categories explicit without changing RGB shaders."""
    import bpy
    categories = {'truck_01_cargo': 'generic', 'truck_02_cargo': 'generic',
                  'catalog_truck_cargo': 'generic', 'Container_roof_ribs': 'metal',
                  'Concrete': 'concrete', 'Marking_yellow': 'painted_metal',
                  'Marking_white': 'painted_metal', 'Env_Galvanized': 'metal'}
    applied = {}
    for name, category in categories.items():
        material = bpy.data.materials.get(name)
        if material is not None:
            material['tefed_ir_material'] = category
            applied[name] = category
    canonical = {'rubber', 'metal', 'painted_metal', 'glass', 'vegetation', 'asphalt', 'concrete', 'soil', 'generic'}
    used = {slot.material for scene in bpy.data.scenes for obj in scene.objects for slot in obj.material_slots if slot.material}
    missing = sorted(material.name for material in used if material.get('tefed_ir_material') not in canonical)
    if missing:
        raise ValueError('Used materials need explicit IR types: ' + ', '.join(missing))
    return {'legacy_tags': applied, 'used_material_count': len(used), 'all_used_materials_explicit': True}


def mesh_hash(obj):
    values = array.array('f', [0]) * (len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get('co', values)
    return hashlib.sha256(values.tobytes()).hexdigest()


def protected_snapshot(scene, all_objects=False):
    selected = set(scene.objects) if all_objects else set()
    if not all_objects:
        for root in scene.objects:
            if 'tefed_class_name' in root:
                selected.update((root, *root.children_recursive))
    return {obj.name: {'matrix': [list(row) for row in obj.matrix_world], 'parent': obj.parent.name if obj.parent else None,
                       'mesh': mesh_hash(obj) if obj.type == 'MESH' else None,
                       'materials': [slot.material.name if slot.material else None for slot in obj.material_slots]}
            for obj in selected}


def refine(output, width, height, samples, render=True):
    import bpy
    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise ValueError('Open the organized vehicle scene before running this script')
    blend = output / 'vehicle_environment.blend'
    if blend.exists() or blend.resolve() == source:
        raise FileExistsError('Choose an output without vehicle_environment.blend')
    scene = bpy.data.scenes['ParkingLot']
    library = bpy.data.scenes['Asset_Library']
    bpy.context.window.scene = scene
    roots_before = {obj.name: obj['tefed_class_id'] for obj in scene.objects if 'tefed_class_name' in obj}
    if len(roots_before) != 12 or not all(name in bpy.data.collections for name in ('Buildings', 'Ground', 'Traffic', 'Vegetation', 'Cameras', 'Lighting')):
        raise ValueError('Expected the organized twelve-vehicle scene')
    vehicle_before, library_before = protected_snapshot(scene), protected_snapshot(library, True)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    upstream = None
    upstream_path = source.with_name('scene.json')
    if upstream_path.is_file():
        upstream_data = json.loads(upstream_path.read_text(encoding='utf-8'))
        upstream = {'metadata': str(upstream_path), 'sha256': hashlib.sha256(upstream_path.read_bytes()).hexdigest(),
                    'generator': upstream_data.get('generator'), 'source_blend_sha256': upstream_data.get('source_blend_sha256')}
        generator = upstream_data.get('generator')
        if isinstance(generator, dict) and generator.get('script') and generator.get('sha256'):
            actual_script = Path(__file__).with_name(Path(generator['script']).name)
            if actual_script.is_file():
                if hashlib.sha256(actual_script.read_bytes()).hexdigest() != generator['sha256']:
                    raise ValueError('Upstream vehicle generator differs from its recorded hash')
                upstream['actual_generator_hash_verified'] = True
    trees = foliage(scene)
    surfaces = update_surfaces(scene)
    joints = curb_joints()
    daylight_and_views(scene)
    scene.render.engine, scene.cycles.device = 'CYCLES', 'CPU'
    scene.cycles.samples, scene.cycles.use_denoising = samples, True
    if hasattr(scene.render, 'compositor_device'):
        scene.render.compositor_device = 'CPU'
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene['asset_license'] = 'CC-BY-4.0'
    assert protected_snapshot(scene) == vehicle_before
    assert protected_snapshot(library, True) == library_before
    output.mkdir(parents=True, exist_ok=True)
    material_coverage = tag_legacy_materials()
    bind_preview_ui(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=True)
    scene = bpy.data.scenes['ParkingLot']
    assert protected_snapshot(scene) == vehicle_before
    assert protected_snapshot(bpy.data.scenes['Asset_Library'], True) == library_before
    assert {obj.name: obj['tefed_class_id'] for obj in scene.objects if 'tefed_class_name' in obj} == roots_before
    assert scene.camera.name == 'Camera' and scene.render.filepath == '//preview.png'
    assert not [image for image in bpy.data.images if image.source == 'FILE' and image.filepath and image.packed_file is None]
    assert not [obj for obj in scene.objects if obj.instance_type == 'COLLECTION']
    result = {'data_origin': 'original_procedural_industrial_park', 'source_blend': str(source), 'source_sha256': source_sha,
              'generator': Path(__file__).name, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'output_blend': blend.name, 'output_sha256': hashlib.sha256(blend.read_bytes()).hexdigest(),
              'model_license': 'CC-BY-4.0',
              'ir_material_coverage': material_coverage,
              'upstream_metadata': upstream,
              'foliage': trees, 'material_slots_updated': surfaces, 'curb_joints': joints,
              'vehicle_roots': roots_before, 'vehicle_geometry_and_matrices_preserved': True, 'asset_library_preserved': True,
              'saved_and_reopened': True, 'external_images': [], 'collection_instances': 0,
              'scene_camera': scene.camera.name, 'render_output': scene.render.filepath,
              'category_counts': {name: len(bpy.data.collections[name].all_objects) for name in ('Buildings', 'Ground', 'Traffic', 'Vegetation', 'Cameras', 'Lighting')},
              'rendered_previews': [], 'render_samples': samples, 'preview_dimensions': [width, height]}
    (output / 'refinement.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if render:
        for filename, camera_name in (('overall_scene.png', 'Environment_Aerial'), ('street_view.png', 'Environment_Street')):
            scene.camera, scene.render.filepath = scene.objects[camera_name], str(output / filename)
            bpy.ops.render.render(write_still=True)
            result['rendered_previews'].append(filename)
        (output / 'refinement.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    arguments = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1800)
    parser.add_argument('--height', type=int, default=1100)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args(arguments)
    try:
        if not 640 <= args.width <= 4096 or not 400 <= args.height <= 4096 or not 1 <= args.samples <= 256:
            raise ValueError('Invalid preview size or sample count')
        result = refine(Path(args.output).expanduser().resolve(), args.width, args.height, args.samples, not args.no_render)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
