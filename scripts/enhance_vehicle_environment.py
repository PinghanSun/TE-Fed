"""Add an editable industrial-park setting to a generated vehicle scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def material(name, color, roughness=.55, metallic=0, noise=0):
    import bpy
    value = bpy.data.materials.new('Env_' + name)
    value.use_nodes = True
    nodes, links = value.node_tree.nodes, value.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    if noise:
        geometry = nodes.new('ShaderNodeNewGeometry')
        grain = nodes.new('ShaderNodeTexNoise')
        grain.inputs['Scale'].default_value = 36 if name == 'Asphalt' else 7
        grain.inputs['Detail'].default_value = 3
        links.new(geometry.outputs['Position'], grain.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .12
        ramp.color_ramp.elements[0].color = (*(component * .73 for component in color), 1)
        ramp.color_ramp.elements[1].position = .88
        ramp.color_ramp.elements[1].color = (*(min(component * 1.22, 1) for component in color), 1)
        links.new(grain.outputs['Fac'], ramp.inputs[0])
        links.new(ramp.outputs['Color'], shader.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .22
        bump.inputs['Distance'].default_value = noise
        links.new(grain.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    return value


def finish(obj, name, surface, collection, bevel=0):
    obj.name = 'Env_' + name
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)
    if surface is not None:
        obj.data.materials.append(surface)
    if bevel:
        modifier = obj.modifiers.new('Manufactured edge', 'BEVEL')
        modifier.width, modifier.segments = bevel, 2
        modifier = obj.modifiers.new('Normals', 'WEIGHTED_NORMAL')
        modifier.keep_sharp = True
    return obj


def box(name, xyz, size, surface, collection, bevel=.025, rotation=0):
    import bpy
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.z = rotation
    return finish(obj, name, surface, collection, bevel)


def cylinder(name, xyz, radius, depth, surface, collection, rotation=(0, 0, 0)):
    import bpy
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth,
                                      location=xyz, rotation=rotation)
    return finish(bpy.context.object, name, surface, collection, .01)


def line(name, points, radius, surface, collection):
    import bpy
    curve = bpy.data.curves.new('Env_' + name, 'CURVE')
    curve.dimensions, curve.bevel_depth, curve.bevel_resolution = '3D', radius, 2
    poly = curve.splines.new('POLY')
    poly.points.add(len(points) - 1)
    for point, xyz in zip(poly.points, points):
        point.co = (*xyz, 1)
    obj = bpy.data.objects.new('Env_' + name, curve)
    collection.objects.link(obj)
    curve.materials.append(surface)
    return obj


def text(name, body, xyz, size, surface, collection, rotation=(math.pi / 2, 0, 0)):
    import bpy
    curve = bpy.data.curves.new('Env_' + name, 'FONT')
    curve.body, curve.size, curve.align_x = body, size, 'CENTER'
    curve.extrude, curve.bevel_depth = .001, .001
    obj = bpy.data.objects.new('Env_' + name, curve)
    collection.objects.link(obj)
    obj.location, obj.rotation_euler = xyz, rotation
    curve.materials.append(surface)
    return obj


def arrow(name, xy, angle, surface, collection):
    import bpy
    points = [(-.14, -1), (.14, -1), (.14, .36), (.55, .05),
              (0, 1.04), (-.55, .05), (-.14, .36)]
    cosine, sine = math.cos(angle), math.sin(angle)
    vertices = [(xy[0] + x * cosine - y * sine, xy[1] + x * sine + y * cosine, .002)
                for x, y in points]
    mesh = bpy.data.meshes.new('Env_' + name)
    mesh.from_pydata(vertices, [], [list(range(len(vertices)))])
    mesh.update()
    obj = bpy.data.objects.new('Env_' + name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(surface)


def tree(name, xy, height, surfaces, collection, rng):
    import bpy
    x, y = xy
    cylinder(name + '_trunk', (x, y, height * .34), .17, height * .68,
             surfaces['bark'], collection)
    for branch in range(5):
        angle = branch * math.tau / 5 + rng.uniform(-.4, .4)
        end = (x + math.cos(angle) * .92, y + math.sin(angle) * .92, height * .79)
        line(name + f'_branch_{branch}', [(x, y, height * .46), end], .06,
             surfaces['bark'], collection)
    for crown in range(23):
        angle = crown * 2.399
        level = crown / 22
        radius = 1.45 * math.sqrt(max(0, 1 - (2 * level - 1) ** 2))
        xyz = (x + math.cos(angle) * radius, y + math.sin(angle) * radius,
               height * .73 + (2 * level - 1) * 1.23 + rng.uniform(-.17, .17))
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1, location=xyz)
        obj = bpy.context.object
        obj.scale = (rng.uniform(.46, .77), rng.uniform(.46, .77), rng.uniform(.53, .88))
        for face in obj.data.polygons:
            face.use_smooth = True
        finish(obj, name + f'_leaves_{crown}', surfaces['leaves'][crown % 3], collection)


def buildings(s, collection):
    # North-side warehouse leaves a service walk behind the heavy-vehicle bays.
    box('Warehouse', (5, 30, 4.3), (54, 17, 8.8), s['cladding'], collection, .08)
    box('Warehouse_roof', (5, 30, 8.80), (54.8, 17.7, .25), s['roof'], collection, .05)
    for index in range(40):
        box(f'Roof_standing_seam_{index}', (-21.2 + index * 1.35, 30, 8.945),
            (.025, 17.55, .040), s['rib'], collection, .004)
    for x in range(-21, 33, 3):
        box(f'Wall_rib_{x}', (x, 21.47, 4.4), (.055, .055, 8.55), s['rib'], collection, .005)
    for index, x in enumerate((-15, -5, 5, 15, 25)):
        box(f'Dock_frame_{index}', (x, 21.37, 2.55), (5.0, .22, 5.10), s['dark'], collection)
        box(f'Dock_door_{index}', (x, 21.22, 2.49), (4.46, .11, 4.68), s['door'], collection)
        for row in range(13):
            box(f'Door_joint_{index}_{row}', (x, 21.151, .45 + row * .34),
                (4.38, .014, .018), s['rib'], collection, 0)
        for side in (-1, 1):
            cylinder(f'Dock_bollard_{index}_{side}', (x + side * 2.7, 20.72, .59),
                     .12, 1.30, s['yellow'], collection)
        text(f'Dock_number_{index}', f'{index + 1:02d}', (x, 21.11, 5.55), .44,
             s['white'], collection)
    text('Warehouse_sign', 'LOGISTICS', (5, 21.25, 7.0), 1.02, s['dark'], collection)
    for index, x in enumerate((-10, 6, 22)):
        box(f'HVAC_{index}', (x, 31, 9.32), (3.5, 2.3, 1.04), s['metal'], collection)
        for fan in (-.85, .85):
            cylinder(f'HVAC_fan_{index}_{fan}', (x + fan, 31, 9.88), .62, .055,
                     s['dark'], collection)
            for rib in range(6):
                box(f'HVAC_grille_{index}_{fan}_{rib}', (x + fan, 30.51 + rib * .19, 9.92),
                    (1.02, .035, .025), s['metal'], collection, .003)
    # A separate office volume and window reveals make the west background readable.
    box('Office', (-37, 8, 5.75), (13, 22, 11.5), s['office'], collection, .08)
    box('Office_plinth', (-37, 8, .35), (13.4, 22.4, .7), s['dark'], collection)
    box('Office_roof_edge', (-37, 8, 11.62), (13.6, 22.6, .22), s['metal'], collection)
    for floor in range(3):
        z = 2.22 + floor * 3.18
        for index in range(6):
            y = -1.0 + index * 3.50
            box(f'Office_reveal_{floor}_{index}', (-30.46, y, z), (.15, 2.6, 2.10), s['dark'], collection)
            box(f'Office_glass_{floor}_{index}', (-30.36, y, z), (.035, 2.34, 1.86), s['glass'], collection, .008)
            box(f'Office_mullion_{floor}_{index}', (-30.31, y, z), (.04, .055, 1.91), s['metal'], collection, .006)
            box(f'Office_sill_{floor}_{index}', (-30.24, y, z - 1.03), (.31, 2.72, .11), s['white'], collection)
        for index in range(3):
            x = -41.0 + index * 4
            box(f'Office_south_reveal_{floor}_{index}', (x, -3.035, z), (2.9, .10, 2.10), s['dark'], collection)
            box(f'Office_south_glass_{floor}_{index}', (x, -3.101, z), (2.66, .025, 1.86), s['glass'], collection, .008)
    box('Office_entry', (-37, -3.17, 1.30), (2.65, .12, 2.60), s['glass'], collection)
    box('Office_canopy', (-37, -4.60, 3.3), (5.7, 3.5, .20), s['metal'], collection)
    for x in (-39.6, -34.4):
        cylinder(f'Office_pillar_{x}', (x, -5.85, 1.55), .07, 3.1, s['metal'], collection)


def ground_details(scene, s, collection):
    base = scene.objects.get('Site_base')
    if base:
        base.location = (0, 7, -.46)
        base.dimensions = (92, 90, .70)
        base.data.materials.clear()
        base.data.materials.append(s['concrete'])
    for name in ('Parking_asphalt', 'Access_road'):
        obj = scene.objects.get(name)
        if obj:
            obj.data.materials.clear()
            obj.data.materials.append(s['asphalt'])
    # Painted markings rest on the asphalt, without floating geometry or raised shadows.
    for obj in list(scene.objects):
        if obj.name.startswith(('Bay_side_', 'Small_bay_end_', 'Heavy_bay_', 'Aisle_center_')):
            obj.dimensions.z = .002
            obj.location.z = -.0025
        elif obj.name.startswith(('Road_center_', 'Road_edge_')):
            obj.dimensions.z = .002
            obj.location.z = .007
    box('Warehouse_walk', (5, 19.15, .045), (54, 4.1, .20), s['paving'], collection)
    box('Office_walk', (-28.7, 3, .045), (3.5, 31, .20), s['paving'], collection)
    for x in range(-22, 33, 2):
        box(f'North_paving_joint_{x}', (x, 19.15, .150), (.014, 4.0, .006), s['joint'], collection, 0)
    for y in range(-12, 19, 2):
        box(f'West_paving_joint_{y}', (-28.7, y, .150), (3.4, .014, .006), s['joint'], collection, 0)
    for index, y in enumerate((-12, -6, 0, 6, 12)):
        box(f'Drain_rim_{index}', (26.25, y, .027), (.40, .72, .035), s['metal'], collection)
        for bar in range(8):
            box(f'Drain_slot_{index}_{bar}', (26.25, y - .30 + bar * .085, .048),
                (.32, .035, .012), s['dark'], collection, .002)
    # Small wheel stops sit at the ends of bays, clear of the vehicles.
    for y in (-12.8, 9.8):
        for x in (-20.25, -16.75, -13.25, -9.75, -6.25):
            box(f'Wheel_stop_{x}_{y}', (x, y, .11), (1.55, .20, .17), s['dark'], collection, .04)
            for stripe in (-.50, 0, .50):
                box(f'Wheel_stop_stripe_{x}_{y}_{stripe}', (x + stripe, y, .201),
                    (.20, .19, .012), s['yellow'], collection, .003)
    # Preserve the existing entrance gap in the south curb.
    box('Entrance_apron', (13, -17.65, -.015), (14.6, 2.0, .08), s['asphalt'], collection, .02)
    for stripe in range(12):
        box(f'Crosswalk_{stripe}', (6.0 + stripe * 1.25, -16.0, -.002),
            (.57, 2.30, .003), s['white'], collection, .001)
    arrow('Entry_arrow', (13, -10.2), 0, s['white'], collection)
    for index, x in enumerate((-17, 1, 19)):
        arrow(f'Aisle_arrow_{index}', (x, -4.15), -math.pi / 2, s['white'], collection)
    for x, y in ((5.45, -15.95), (20.55, -15.95)):
        box(f'Gate_pedestal_{x}', (x, y, .61), (.42, .50, 1.22), s['yellow'], collection)
        box(f'Gate_reader_{x}', (x, y - .264, .90), (.24, .035, .31), s['dark'], collection)
        cylinder(f'Gate_hinge_{x}', (x, y, 1.13), .11, .61, s['metal'], collection, (math.pi / 2, 0, 0))
    # Open barriers leave the entrance visible and usable.
    for x in (5.45, 20.55):
        box(f'Gate_raised_arm_{x}', (x, -15.95, 3.26), (.10, .12, 4.2), s['white'], collection)
        for z in (1.65, 2.65, 3.65, 4.65):
            box(f'Gate_red_stripe_{x}_{z}', (x, -16.017, z), (.105, .014, .28), s['red'], collection, .002)
    cylinder('Speed_sign_post', (23.8, -17, 1.55), .045, 3.10, s['metal'], collection)
    cylinder('Speed_sign_red', (23.8, -17, 2.80), .44, .045, s['red'], collection, (math.pi / 2, 0, 0))
    cylinder('Speed_sign_white', (23.8, -17.03, 2.80), .365, .045, s['white'], collection, (math.pi / 2, 0, 0))
    text('Speed_sign_number', '10', (23.8, -17.067, 2.63), .35, s['dark'], collection)


def landscape(s, collection):
    rng = random.Random(184)
    for index, (x, y, height) in enumerate(((-28.5, -11, 5.4), (-28.5, -3, 6.2),
                                           (-28.5, 7, 5.8), (-27.8, 15, 6.3),
                                           (30.7, -11, 5.9), (30.7, 0, 6.1),
                                           (30.7, 12, 6.5), (30.7, 20, 6.0))):
        box(f'Tree_bed_{index}', (x, y, -.015), (2.8, 3.6, .12), s['soil'], collection)
        tree(f'Tree_{index}', (x, y), height, s, collection, rng)
    for x in (-29.95, 29.5):
        for index in range(10):
            y = -12.5 + index * 3.2
            box(f'Hedge_{x}_{index}', (x, y, .47), (.76, 2.5, .85), s['leaves'][index % 3], collection, .24)
    for index, x in enumerate(range(-25, 30, 3)):
        cylinder(f'Fence_post_{index}', (x, -28.0, .97), .045, 2.00, s['metal'], collection)
    for z in (.25, .80, 1.35, 1.90):
        line(f'Fence_wire_{z}', [(-25, -28, z), (29, -28, z)], .015, s['metal'], collection)


def camera(scene, name, xyz, target, collection, scale=None, lens=48):
    import bpy
    from mathutils import Vector
    data = bpy.data.cameras.new(name)
    if scale is None:
        data.type, data.lens = 'PERSP', lens
    else:
        data.type, data.ortho_scale = 'ORTHO', scale
    data.clip_start, data.clip_end = .05, 1000
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = xyz
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    return obj


def snapshot(scene):
    return {obj.name: {'class_name': obj['tefed_class_name'], 'class_id': obj['tefed_class_id'],
                       'matrix': [list(row) for row in obj.matrix_world],
                       'parts': sorted(part.name for part in obj.children_recursive)}
            for obj in scene.objects if 'tefed_class_name' in obj}


def organize_scene_ui(scene):
    """Group the setting and frame the Layout without changing the render camera."""
    import bpy
    environment = scene.collection.children.get('Environment_Detail')
    aerial = scene.objects.get('Environment_Aerial')
    runtime_camera = scene.objects.get('Camera')
    if environment is None or aerial is None or runtime_camera is None:
        raise ValueError('Expected the generated environment and its cameras')
    categories = ('Buildings', 'Ground', 'Traffic', 'Vegetation', 'Cameras', 'Lighting')
    groups = {name: environment.children.get(name) for name in categories}
    for name, group in groups.items():
        if group is None:
            group = bpy.data.collections.new(name)
            environment.children.link(group)
            groups[name] = group
    protected = set()
    for obj in scene.objects:
        if 'tefed_class_name' in obj:
            protected.update((obj, *obj.children_recursive))
    library = bpy.data.scenes.get('Asset_Library')
    if library is not None:
        protected.update(library.objects)
    existing = set(environment.all_objects)
    for obj in list(scene.objects):
        if obj in protected:
            continue
        name = obj.name.removeprefix('Env_')
        if obj.type == 'CAMERA':
            category = 'Cameras'
        elif obj.type == 'LIGHT' or name.startswith('Lamp_'):
            category = 'Lighting'
        elif name.startswith(('Tree_', 'Hedge_')):
            category = 'Vegetation'
        elif name.startswith(('Crosswalk_', 'Entry_arrow', 'Aisle_arrow_', 'Gate_',
                              'Speed_sign_', 'Wheel_stop_', 'Fence_', 'Road_center_',
                              'Road_edge_', 'Bay_side_', 'Small_bay_', 'Heavy_bay_', 'Aisle_center_')):
            category = 'Traffic'
        elif name in ('Site_base', 'Parking_asphalt', 'Access_road', 'Warehouse_walk', 'Office_walk') or name.startswith(
                ('Curb_', 'North_curb', 'South_curb_', 'North_paving_', 'West_paving_', 'Drain_', 'Entrance_apron')):
            category = 'Ground'
        elif name.startswith(('Warehouse', 'Roof_', 'Wall_', 'Dock_', 'Door_', 'HVAC_', 'Office')):
            category = 'Buildings'
        elif obj in existing:
            category = 'Ground'
        else:
            continue
        target = groups[category]
        if obj not in target.objects.values():
            target.objects.link(obj)
        for current in list(obj.users_collection):
            if current != target:
                current.objects.unlink(obj)
    scene.camera = runtime_camera
    scene.render.filepath = '//preview.png'
    layout_views = 0
    for screen in bpy.data.screens:
        if screen.name != 'Layout' and not screen.name.startswith('Layout.'):
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                view = area.spaces.active
                view.use_local_camera, view.camera = True, aerial
                view.region_3d.view_perspective = 'CAMERA'
                view.region_3d.view_camera_zoom = 0
                view.region_3d.view_camera_offset = (0, 0)
                layout_views += 1
    return {'environment_object_count': len(environment.all_objects),
            'category_counts': {name: len(group.all_objects) for name, group in groups.items()},
            'runtime_camera': runtime_camera.name, 'layout_camera': aerial.name,
            'layout_views': layout_views, 'default_render_output': scene.render.filepath}


def enhance(output, width, samples):
    import bpy
    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise ValueError('Open a generated vehicle .blend before running this script')
    scene = bpy.data.scenes.get('ParkingLot')
    if scene is None or 'Asset_Library' not in bpy.data.scenes:
        raise ValueError('Input must contain ParkingLot and Asset_Library')
    if bpy.data.collections.get('Environment_Detail'):
        raise ValueError('This file already has an Environment_Detail collection')
    bpy.context.window.scene = scene
    before = snapshot(scene)
    library_objects = sorted(obj.name for obj in bpy.data.scenes['Asset_Library'].objects)
    if len(before) != 12:
        raise ValueError('Expected the original twelve vehicle roots')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    collection = bpy.data.collections.new('Environment_Detail')
    scene.collection.children.link(collection)
    s = {
        'asphalt': material('Asphalt', (.20, .22, .235), .86, noise=.010),
        'concrete': material('Concrete', (.39, .41, .40), .85, noise=.004),
        'paving': material('Paving', (.55, .54, .50), .78, noise=.004),
        'joint': material('Paving_joints', (.19, .20, .19), .9),
        'cladding': material('Cladding', (.48, .57, .61), .60, .27),
        'rib': material('Panel_seams', (.24, .30, .32), .6, .24),
        'roof': material('Roof', (.32, .37, .39), .65, .35, noise=.003),
        'door': material('Roller_door', (.62, .64, .62), .50, .35),
        'office': material('Office_plaster', (.67, .64, .56), .8, noise=.002),
        'glass': material('Office_glass', (.09, .18, .23), .16, .72),
        'metal': material('Galvanized', (.38, .43, .45), .37, .78),
        'dark': material('Dark_hardware', (.035, .045, .05), .51, .4),
        'white': material('Traffic_white', (.82, .82, .74), .65),
        'yellow': material('Traffic_yellow', (.90, .61, .12), .6),
        'red': material('Traffic_red', (.61, .055, .035), .51),
        'soil': material('Mulch', (.13, .085, .042), .95, noise=.018),
        'bark': material('Bark', (.20, .13, .075), .92, noise=.016),
        'leaves': [material(f'Foliage_{index}', color, .85, noise=.025)
                   for index, color in enumerate(((.12, .24, .075), (.18, .31, .10), (.24, .37, .13)))],
    }
    buildings(s, collection)
    ground_details(scene, s, collection)
    landscape(s, collection)
    world = bpy.data.worlds.new('Environment_Daylight')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (.71, .81, .96, 1)
    background.inputs['Strength'].default_value = .55
    scene.world = world
    sun_data = bpy.data.lights.new('Environment_Sun', 'SUN')
    sun_data.energy, sun_data.angle = 2.3, math.radians(12)
    sun = bpy.data.objects.new('Environment_Sun', sun_data)
    collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28), math.radians(-24), math.radians(-30))
    scene.render.engine, scene.cycles.device = 'CYCLES', 'CPU'
    scene.cycles.samples, scene.cycles.use_denoising = samples, True
    if hasattr(scene.render, 'compositor_device'):
        scene.render.compositor_device = 'CPU'
    scene.render.resolution_x, scene.render.resolution_y = width, round(width * .66)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform, scene.view_settings.exposure = 'AgX', 0
    aerial = camera(scene, 'Environment_Aerial', (55, -66, 59), (0, 5, 2), collection, scale=107)
    street = camera(scene, 'Environment_Street', (37, -41, 12), (2, 6, 2), collection, lens=35)
    camera(scene, 'Environment_Detail', (-25, -19, 3.8), (-16.5, -10, .8), collection, lens=52)
    close = camera(scene, 'Environment_Close', (30, -24, 7.2), (21.8, -14.8, .8), collection, lens=40)
    if snapshot(scene) != before:
        raise RuntimeError('Vehicle roots or their parts changed during environment creation')
    if sorted(obj.name for obj in bpy.data.scenes['Asset_Library'].objects) != library_objects:
        raise RuntimeError('Asset_Library changed during environment creation')
    for view, name in ((aerial, 'overall_scene.png'), (street, 'street_view.png'),
                       (close, 'environment_detail.png')):
        scene.camera, scene.render.filepath = view, str(output / name)
        bpy.ops.render.render(write_still=True)
    organized = organize_scene_ui(scene)
    scene['environment_origin'] = 'original_procedural_industrial_park'
    blend = output / 'vehicle_environment.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    reopened = bpy.data.scenes['ParkingLot']
    if snapshot(reopened) != before:
        raise RuntimeError('Vehicle roots changed after saving and reopening')
    external_images = [image.filepath for image in bpy.data.images
                       if image.source == 'FILE' and image.filepath and image.packed_file is None]
    if external_images:
        raise RuntimeError('The scene depends on unpacked external image files')
    metadata = {
        'source_blend': str(source), 'source_sha256': source_sha,
        'generator': Path(__file__).name,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_blend': blend.name, 'data_origin': 'original_procedural_industrial_park',
        'scene': 'ParkingLot', 'environment_collection': 'Environment_Detail',
        'environment_object_count': len(bpy.data.collections['Environment_Detail'].all_objects),
        'organization': organized,
        'vehicle_roots_preserved': True, 'vehicle_root_count': len(before),
        'asset_library_objects_preserved': True, 'external_images': external_images,
        'reopened_validation': True, 'cpu_renderer': True, 'cpu_compositor': True,
        'views': {'overall_scene.png': 'Environment_Aerial', 'street_view.png': 'Environment_Street',
                  'environment_detail.png': 'Environment_Close'},
        'cameras': ['Camera', 'Environment_Aerial', 'Environment_Street', 'Environment_Detail', 'Environment_Close'],
        'detail_groups': ['asphalt grain', 'paving joints', 'drains', 'wheel stops', 'crosswalk',
                          'open entrance barriers', 'traffic signs', 'warehouse panels and loading doors',
                          'office windows and canopy', 'rooftop HVAC', 'trees and hedges', 'perimeter fence'],
    }
    mapping = {name: record['class_id'] for name, record in before.items()}
    request_source = source.with_name('render_request.json')
    if request_source.is_file():
        request = json.loads(request_source.read_text(encoding='utf-8'))
    else:
        car_id = next(record['class_id'] for record in before.values() if record['class_name'] == 'car')
        request = {'descriptor': {'source_class': car_id, 'attrs': [.35, .15, 0, .15, 0, 0, 0, 0]},
                   'seed': 0, 'case_id': 'vehicle_environment_check', 'image_size': [640, 640]}
    request['scene_path'], request['scene_class_map'] = str(blend), mapping
    (output / 'render_request.json').write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
    mapping_lines = '\n'.join(f'{name} = {value}' for name, value in mapping.items())
    config = ('[renderer]\nenabled = true\nblender = "blender"\nscene = "' + blend.as_posix()
              + '"\nsamples = 16\n\n[renderer.scene_class_map]\n' + mapping_lines + '\n')
    (output / 'renderer_config.toml').write_text(config, encoding='utf-8')
    (output / 'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Editable vehicle environment\n\nOpen `vehicle_environment.blend` in Blender. '
        'The original `Camera` and twelve vehicle roots are retained. '
        '`Environment_Aerial`, `Environment_Street`, `Environment_Close` and `Environment_Detail` provide additional views. '
        'Background objects are grouped under `Environment_Detail`; the separate `Asset_Library` is unchanged.\n\n'
        'The industrial-park background is original procedural geometry with node-based materials. '
        'It is a new demonstration setting, with no external image dependencies.\n', encoding='utf-8')
    return metadata


def main(argv=None):
    arguments = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh output directory')
    parser.add_argument('--preview-width', '--width', type=int, default=1800)
    parser.add_argument('--samples', type=int, default=48)
    args = parser.parse_args(arguments)
    try:
        output = Path(args.output).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError('Choose a fresh output directory')
        if not 640 <= args.preview_width <= 4096 or not 1 <= args.samples <= 256:
            raise ValueError('Preview width must be 640-4096; samples must be 1-256')
        import bpy
        if not bpy.data.filepath:
            raise ValueError('Open a generated vehicle .blend before running this script')
        output.mkdir(parents=True, exist_ok=False)
        result = enhance(output, args.preview_width, args.samples)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
