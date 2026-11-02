"""Build editable vehicle assets and a parking-lot rendering example."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


DEFAULT_CLASSES = {'car': 0, 'truck': 1, 'bus': 2, 'van': 3, 'freight_car': 4}
PAINTS = [(0.12, 0.28, 0.40), (0.66, 0.60, 0.49), (0.43, 0.065, 0.09),
          (0.25, 0.38, 0.46), (0.82, 0.82, 0.78)]


def read_classes(path=None):
    classes = dict(DEFAULT_CLASSES) if path is None else json.loads(
        Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(classes, dict) or set(classes) != set(DEFAULT_CLASSES):
        raise ValueError('Class map requires car, truck, bus, van and freight_car')
    values = list(classes.values())
    if any(type(value) is not int for value in values) or sorted(values) != list(range(5)):
        raise ValueError('Class IDs must be unique integers 0 through 4')
    return classes


def material(name, color, roughness=.45, metallic=0, emission=0):
    import bpy
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    shader = value.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    if 'Coat Weight' in shader.inputs and name.startswith('Paint'):
        shader.inputs['Coat Weight'].default_value = .65
        shader.inputs['Coat Roughness'].default_value = .16
        noise = value.node_tree.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 350
        noise.inputs['Detail'].default_value = 2
        bump = value.node_tree.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .025
        bump.inputs['Distance'].default_value = .008
        value.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
        value.node_tree.links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    if emission:
        shader.inputs['Emission Color'].default_value = (*color, 1)
        shader.inputs['Emission Strength'].default_value = emission
    return value


def finish(obj, name, surface, root=None, bevel=0):
    obj.name = name
    obj.data.materials.append(surface)
    if root is not None:
        obj.parent = root
    if bevel:
        modifier = obj.modifiers.new('Edge radius', 'BEVEL')
        modifier.width, modifier.segments = bevel, 3
        modifier.affect = 'EDGES'
        modifier = obj.modifiers.new('Surface normals', 'WEIGHTED_NORMAL')
        modifier.keep_sharp = True
    return obj


def box(name, xyz, dimensions, surface, root=None, bevel=.04):
    import bpy
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    obj = bpy.context.object
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, surface, root, bevel)


def cylinder(name, xyz, radius, depth, surface, root=None, rotation=(0, 0, 0), bevel=.015):
    import bpy
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth,
                                       location=xyz, rotation=rotation)
    return finish(bpy.context.object, name, surface, root, bevel)


def polygon(name, points, surface, root):
    import bpy
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(points, [], [list(range(len(points)))])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return finish(obj, name, surface, root)


def cabin(name, width, bottom_y, top_y, z0, z1, surface, root):
    import bpy
    x0, x1 = width / 2, width * .39
    y0, y1 = bottom_y
    t0, t1 = top_y
    vertices = [(-x0, y0, z0), (x0, y0, z0), (x0, y1, z0), (-x0, y1, z0),
                (-x1, t0, z1), (x1, t0, z1), (x1, t1, z1), (-x1, t1, z1)]
    faces = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return finish(obj, name, surface, root, .045)


def mesh_object(name, vertices, faces, surface, root):
    import bpy
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return finish(obj, name, surface, root)


def boxes_mesh(name, specs, surface, root):
    from mathutils import Euler, Vector
    vertices, faces = [], []
    corners = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    indices = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5),
               (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    for center, size, rotation in specs:
        start = len(vertices)
        transform = Euler(rotation).to_matrix()
        for corner in corners:
            point = transform @ Vector(tuple(corner[i] * size[i] / 2 for i in range(3)))
            vertices.append(tuple(point[i] + center[i] for i in range(3)))
        faces.extend(tuple(start + index for index in face) for face in indices)
    return mesh_object(name, vertices, faces, surface, root)


def lines_mesh(name, paths, radius, surface, root):
    from mathutils import Vector
    vertices, faces = [], []
    for points in paths:
        for a, b in zip(points, points[1:]):
            a, b = Vector(a), Vector(b)
            axis = (b - a).normalized()
            helper = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
            first = axis.cross(helper).normalized()
            second = axis.cross(first).normalized()
            start = len(vertices)
            for center in (a, b):
                for index in range(8):
                    angle = math.tau * index / 8
                    vertices.append(tuple(center + radius * (math.cos(angle) * first + math.sin(angle) * second)))
            faces.extend((start + index, start + (index + 1) % 8,
                          start + (index + 1) % 8 + 8, start + index + 8) for index in range(8))
            faces.append(tuple(start + index for index in reversed(range(8))))
            faces.append(tuple(start + 8 + index for index in range(8)))
    return mesh_object(name, vertices, faces, surface, root)


def wheel_ring(name, xyz, radius, thickness, surface, root):
    vertices, faces = [], []
    for index in range(48):
        angle = math.tau * index / 48
        for cross in range(8):
            c = math.tau * cross / 8
            vertices.append((xyz[0] + thickness * math.sin(c),
                             xyz[1] + (radius + thickness * math.cos(c)) * math.sin(angle),
                             xyz[2] + (radius + thickness * math.cos(c)) * math.cos(angle)))
    for index in range(48):
        for cross in range(8):
            faces.append((index * 8 + cross, ((index + 1) % 48) * 8 + cross,
                          ((index + 1) % 48) * 8 + (cross + 1) % 8, index * 8 + (cross + 1) % 8))
    return mesh_object(name, vertices, faces, surface, root)


def detailed_wheel(root, prefix, side, x, y, radius, surfaces):
    face_x = x + side * .178
    wheel_ring(prefix + '_rim_lip', (face_x, y, radius), radius * .59,
               .017, surfaces['metal'], root)
    wheel_ring(prefix + '_sidewall_line', (x + side * .146, y, radius), radius * .85,
               .007, surfaces['sidewall'], root)
    spokes = []
    for index in range(5):
        angle = math.tau * index / 5
        spokes.append(((face_x, y - radius * .34 * math.sin(angle), radius + radius * .34 * math.cos(angle)),
                       (.032, radius * .12, radius * .54), (angle, 0, 0)))
    boxes_mesh(prefix + '_five_spokes', spokes, surfaces['metal'], root)
    bolt_vertices, bolt_faces = [], []
    for index in range(5):
        angle = math.tau * index / 5
        center = (face_x + side * .02, y + radius * .18 * math.sin(angle),
                  radius + radius * .18 * math.cos(angle))
        start = len(bolt_vertices)
        for offset in (-.01, .01):
            for corner in range(6):
                theta = math.tau * corner / 6
                bolt_vertices.append((center[0] + offset, center[1] + .014 * math.sin(theta),
                                      center[2] + .014 * math.cos(theta)))
        bolt_faces.extend((start + i, start + (i + 1) % 6, start + (i + 1) % 6 + 6,
                           start + i + 6) for i in range(6))
        bolt_faces.append(tuple(start + i for i in reversed(range(6))))
        bolt_faces.append(tuple(start + i + 6 for i in range(6)))
    mesh_object(prefix + '_hub_bolts', bolt_vertices, bolt_faces, surfaces['brightmetal'], root)
    tread = []
    for band in (-1, 1):
        for index in range(32):
            angle = math.tau * (index + .25 * band) / 32
            tread.append(((x + band * .075, y - (radius + .004) * math.sin(angle),
                           radius + (radius + .004) * math.cos(angle)),
                          (.105, .052, .015), (angle, 0, band * .16)))
    boxes_mesh(prefix + '_tread_blocks', tread, surfaces['tread'], root)


def wheels(root, width, axles, radius, surfaces):
    for axle_index, y in enumerate(axles):
        for side in (-1, 1):
            x = side * (width / 2 - .04)
            prefix = f'{root.name}_wheel_{axle_index}_{side}'
            cylinder(prefix + '_rubber', (x, y, radius), radius, .28,
                     surfaces['rubber'], root, (0, math.pi / 2, 0), .035)
            cylinder(prefix + '_rim', (x + side * .151, y, radius), radius * .60, .035,
                     surfaces['darkmetal'], root, (0, math.pi / 2, 0), .015)
            cylinder(prefix + '_hub', (x + side * .176, y, radius), radius * .23, .028,
                     surfaces['darkmetal'], root, (0, math.pi / 2, 0), .008)
            detailed_wheel(root, prefix, side, x, y, radius, surfaces)
            arc = [(side * (width / 2 + .115), y + radius * 1.09 * math.cos(index * math.pi / 20),
                    radius + radius * 1.09 * math.sin(index * math.pi / 20)) for index in range(21)]
            lines_mesh(prefix + '_wheel_arch', [arc], .025, surfaces['darkmetal'], root)


def lights(root, width, length, z, surfaces):
    for side in (-1, 1):
        box(root.name + f'_headlamp_{side}', (side * width * .32, -length / 2 - .025, z),
            (width * .20, .060, .19), surfaces['darkmetal'], root, .028)
        box(root.name + f'_reflector_{side}', (side * width * .32, -length / 2 - .060, z),
            (width * .17, .022, .135), surfaces['brightmetal'], root, .018)
        box(root.name + f'_led_{side}', (side * width * .32, -length / 2 - .078, z + .031),
            (width * .15, .014, .043), surfaces['headlamp'], root, .012)
        box(root.name + f'_indicator_{side}', (side * width * .405, -length / 2 - .079, z),
            (.045, .015, .105), surfaces['amber'], root, .012)
        box(root.name + f'_taillamp_{side}', (side * width * .35, length / 2 + .025, z),
            (.23, .055, .22), surfaces['darkmetal'], root, .025)
        box(root.name + f'_red_lens_{side}', (side * width * .35, length / 2 + .061, z + .042),
            (.19, .025, .083), surfaces['taillamp'], root, .016)
        box(root.name + f'_reverse_lens_{side}', (side * width * .35, length / 2 + .061, z - .047),
            (.19, .025, .046), surfaces['headlamp'], root, .010)
    box(root.name + '_grille', (0, -length / 2 - .030, z - .18),
        (width * .45, .05, .10), surfaces['darkmetal'], root, .018)
    slats = [((0, -length / 2 - .059, z - .25 + index * .050),
              (width * .43, .022, .016), (0, 0, 0)) for index in range(4)]
    boxes_mesh(root.name + '_grille_slats', slats, surfaces['metal'], root)


def mirrors(root, width, y, z, surfaces):
    for side in (-1, 1):
        box(root.name + f'_mirror_arm_{side}', (side * (width / 2 + .07), y, z),
            (.21, .09, .07), surfaces['darkmetal'], root, .018)
        box(root.name + f'_mirror_{side}', (side * (width / 2 + .19), y, z + .02),
            (.13, .25, .18), surfaces['darkmetal'], root, .035)
        box(root.name + f'_mirror_lens_{side}', (side * (width / 2 + .262), y, z + .027),
            (.012, .195, .133), surfaces['mirror'], root, .012)


def glass_frames_and_wipers(root, surfaces):
    from mathutils import Vector
    panes = [obj for obj in root.children if obj.type == 'MESH'
             and len(obj.data.vertices) == 4 and len(obj.data.polygons) == 1
             and any(term in obj.name for term in ('window', 'windscreen', 'windshield'))]
    shell_suffix = {'car': '_cabin', 'van': '_roof', 'truck': '_cab_upper',
                    'freight_car': '_cab_upper'}.get(root['tefed_class_name'])
    shell = next((obj for obj in root.children if shell_suffix and obj.name == root.name + shell_suffix), None)
    if shell is not None:
        for pane in panes:
            face_index = (1 if 'windscreen' in pane.name or 'windshield' in pane.name else
                          3 if '_rear_window' in pane.name else
                          4 if '_window_-1' in pane.name else 2)
            face = shell.data.polygons[face_index]
            anchor = shell.matrix_local @ shell.data.vertices[face.vertices[0]].co
            normal = (shell.matrix_local.to_3x3() @ face.normal).normalized()
            inverse = pane.matrix_local.inverted()
            for vertex in pane.data.vertices:
                point = pane.matrix_local @ vertex.co
                projected = point - normal * (point - anchor).dot(normal) + normal * .014
                vertex.co = inverse @ projected
            pane.data.update()
    outlines = []
    for pane in panes:
        points = [tuple(pane.matrix_local @ vertex.co) for vertex in pane.data.vertices]
        outlines.append(points + [points[0]])
        if 'windscreen' not in pane.name and 'windshield' not in pane.name:
            continue
        a, b, c, d = [Vector(point) for point in points]
        paths = []
        def on_glass(u, v):
            value = (a.lerp(b, u)).lerp(d.lerp(c, u), v)
            value.y -= .010
            return tuple(value)
        for u in (.28, .72):
            paths.append([on_glass(u, .06), on_glass(u + .11, .51)])
            paths.append([on_glass(u + .02, .46), on_glass(u + .22, .62)])
        lines_mesh(root.name + '_wiper_blades', paths, .009, surfaces['rubber'], root)
    if outlines:
        lines_mesh(root.name + '_window_seals', outlines, .014, surfaces['rubber'], root)


def panel_details(root, class_name, width, length, surfaces):
    name = root.name
    dark, metal = surfaces['panelgap'], surfaces['metal']
    seams, handles = [], []
    if class_name == 'car':
        for side in (-1, 1):
            x = side * (width / 2 + .004)
            for a, b in ((-.91, .11), (.16, 1.25)):
                seams.append([(x, a, .86), (x, a + .08, .54),
                              (x, b - .06, .53), (x, b, .86)])
                handles.append(((side * .942, b - .23, .84), (.037, .22, .033), (0, 0, 0)))
            wheel_ring(name + f'_fuel_seam_{side}', (side * .931, 1.63, .76), .085, .004, dark, root)
        seams.extend([[(-.73, -2.08, 1.022), (-.78, -.98, 1.022),
                       (.78, -.98, 1.022), (.73, -2.08, 1.022)],
                      [(-.73, 1.43, .928), (-.73, 2.05, .928),
                       (.73, 2.05, .928), (.73, 1.43, .928)]])
    elif class_name == 'van':
        for side in (-1, 1):
            x = side * 1.006
            for a, b in ((-2.13, -.32), (-.12, 2.24)):
                seams.append([(x, a, 1.42), (x, a + .07, .58),
                              (x, b, .58), (x, b, 1.42)])
            handles.append(((side * 1.024, .20, 1.30), (.04, .30, .045), (0, 0, 0)))
            box(name + f'_slide_rail_{side}', (side * 1.018, 1.16, 1.56),
                (.025, 2.12, .035), metal, root, .006)
        seams.extend([[(-.81, 2.56, .59), (-.81, 2.56, 1.77),
                       (.81, 2.56, 1.77), (.81, 2.56, .59)],
                      [(0, 2.568, .59), (0, 2.568, 2.00)]])
        for side in (-1, 1):
            box(name + f'_rear_handle_{side}', (side * .19, 2.586, 1.21),
                (.07, .035, .20), metal, root, .012)
    elif class_name in ('truck', 'freight_car'):
        cab_y = -length / 2 + 1.20
        for side in (-1, 1):
            x = side * (width / 2 + .008)
            seams.append([(x, cab_y - .99, 1.57), (x, cab_y - .90, .73),
                          (x, cab_y + .94, .73), (x, cab_y + .94, 1.60)])
            handles.append(((x + side * .025, cab_y + .62, 1.49), (.04, .26, .045), (0, 0, 0)))
            box(name + f'_entry_step_{side}', (side * (width / 2 + .08), cab_y + .05, .58),
                (.31, 1.55, .12), surfaces['darkmetal'], root, .025)
            step_ridges = [((side * (width / 2 + .08), cab_y - .48 + i * .22, .647),
                            (.27, .045, .018), (0, 0, 0)) for i in range(6)]
            boxes_mesh(name + f'_step_grip_{side}', step_ridges, surfaces['metal'], root)
        if class_name == 'truck':
            for side in (-1, 1):
                x = side * 1.187
                for z in (1.34, 1.82):
                    seams.append([(x, -1.40, z), (x, 3.82, z)])
            for x in (-.87, .87):
                for z in (1.30, 1.86):
                    box(name + f'_tailgate_hinge_{x}_{z}', (x, 3.972, z),
                        (.26, .05, .075), metal, root, .01)
        else:
            seams.append([(0, 6.067, .89), (0, 6.067, 3.35)])
            roof = [((0, -3.10 + i * .55, 3.455), (2.28, .050, .035), (0, 0, 0))
                    for i in range(17)]
            boxes_mesh(name + '_roof_corrugations', roof, surfaces['container_edge'], root)
            hinges = [((x, 6.068, z), (.18, .055, .08), (0, 0, 0))
                      for x in (-1.05, 1.05) for z in (1.13, 2.11, 3.12)]
            boxes_mesh(name + '_rear_door_hinges', hinges, metal, root)
            markers = [((side * 1.257, y, .94), (.025, .15, .075), (0, 0, 0))
                       for side in (-1, 1) for y in (-2.5, -.5, 1.5, 3.5, 5.5)]
            boxes_mesh(name + '_side_markers', markers, surfaces['amber'], root)
    else:
        outlines = []
        for side in (-1, 1):
            for index in range(8):
                y = -4.05 + index * 1.12
                x = side * 1.258
                outlines.append([(x, y - .48, 1.695), (x, y + .48, 1.695),
                                 (x, y + .48, 2.605), (x, y - .48, 2.605), (x, y - .48, 1.695)])
        lines_mesh(name + '_bus_window_frames', outlines, .015, surfaces['rubber'], root)
        for index, y in enumerate((-3.38, 1.02)):
            box(name + f'_passenger_door_{index}', (1.270, y, 1.50),
                (.045, 1.02, 1.88), surfaces['glass'], root, .035)
            seams.append([(1.299, y, .56), (1.299, y, 2.44)])
            for delta in (-.50, .50):
                box(name + f'_door_pillar_{index}_{delta}', (1.305, y + delta, 1.50),
                    (.045, .045, 1.90), surfaces['trim'], root, .01)
        box(name + '_destination_panel', (0, -5.343, 2.80),
            (1.74, .035, .20), surfaces['darkmetal'], root, .025)
        vents = [((0, 5.349, .99 + index * .055), (1.24, .022, .022), (0, 0, 0))
                 for index in range(7)]
        boxes_mesh(name + '_rear_air_grille', vents, surfaces['darkmetal'], root)
        roof_grille = [((0, -.29 + index * .19, 3.301), (1.10, .050, .022), (0, 0, 0))
                       for index in range(9)]
        boxes_mesh(name + '_roof_air_grille', roof_grille, surfaces['darkmetal'], root)
        wiper = [[(-.64, -5.346, 1.61), (-.31, -5.346, 2.31)],
                 [(.40, -5.346, 1.61), (.72, -5.346, 2.31)],
                 [(-.50, -5.351, 2.25), (-.12, -5.351, 2.42)],
                 [(.52, -5.351, 2.25), (.90, -5.351, 2.42)]]
        lines_mesh(name + '_bus_wipers', wiper, .012, surfaces['rubber'], root)
    if seams:
        lines_mesh(name + '_panel_gaps', seams, .006, dark, root)
    if handles:
        boxes_mesh(name + '_door_handles', handles, metal, root)
    for end in (-1, 1):
        y = end * (length / 2 + .045)
        box(name + f'_bumper_{end}', (0, y, .47 if class_name == 'car' else .60),
            (width * .89, .12, .12), surfaces['darkmetal'], root, .04)
        box(name + f'_blank_plate_{end}', (0, y + end * .074, .63 if class_name == 'car' else .77),
            (.43, .022, .12), surfaces['plate'], root, .012)
    if class_name in ('car', 'van', 'bus'):
        box(name + '_underbody', (0, 0, .43), (width * .68, length * .74, .14),
            surfaces['darkmetal'], root, .025)
    glass_frames_and_wipers(root, surfaces)
    root['tefed_detail_version'] = 2
    root['tefed_detail_features'] = 'panel_gaps;window_seals;wipers;layered_lamps;mirror_lenses;wheel_arches;wheel_spokes;hex_hub_bolts;tread_blocks;bumper'


def create_vehicle(class_name, name, class_id, paint, surfaces):
    import bpy
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root['tefed_class_name'], root['tefed_class_id'] = class_name, class_id
    root['tefed_instance_id'] = name
    body = material('Paint_' + name, paint, .24, .35)
    glass = surfaces['glass']
    if class_name == 'car':
        width, length = 1.84, 4.5
        box(name + '_lower_body', (0, 0, .64), (width, length, .60), body, root, .09)
        box(name + '_hood', (0, -1.49, .94), (1.73, 1.35, .15), body, root, .08)
        cabin(name + '_cabin', 1.71, (-.95, 1.36), (-.38, .83), .94, 1.47, body, root)
        polygon(name + '_windshield', [(-.76, -.932, 1.00), (.76, -.932, 1.00),
                (.63, -.398, 1.405), (-.63, -.398, 1.405)], glass, root)
        polygon(name + '_rear_window', [(-.74, 1.34, 1.00), (-.62, .85, 1.405),
                (.62, .85, 1.405), (.74, 1.34, 1.00)], glass, root)
        for side in (-1, 1):
            for index, (a, b) in enumerate(((-.78, .08), (.15, 1.17))):
                polygon(name + f'_side_window_{side}_{index}',
                        [(side * .862, a, 1.015), (side * .862, b, 1.015),
                         (side * .677, min(b, .76), 1.40), (side * .677, max(a, -.31), 1.40)], glass, root)
        wheels(root, width, (-1.38, 1.35), .37, surfaces)
        lights(root, width, length, .77, surfaces)
        mirrors(root, width, -.62, 1.12, surfaces)
    elif class_name == 'van':
        width, length = 1.98, 5.05
        box(name + '_body', (0, .10, 1.12), (width, length - .15, 1.40), body, root, .10)
        cabin(name + '_roof', 1.94, (-2.40, 2.54), (-1.84, 2.36), 1.40, 2.13, body, root)
        polygon(name + '_windshield', [(-.87, -2.397, 1.48), (.87, -2.397, 1.48),
                (.73, -1.86, 2.03), (-.73, -1.86, 2.03)], glass, root)
        for side in (-1, 1):
            polygon(name + f'_front_window_{side}', [(side * .978, -2.13, 1.43),
                    (side * .978, -.50, 1.43), (side * .772, -.50, 2.03),
                    (side * .772, -1.74, 2.03)], glass, root)
            box(name + f'_cargo_seam_{side}', (side * 1.001, .25, 1.10),
                (.018, .025, 1.03), surfaces['darkmetal'], root, .003)
            box(name + f'_handle_{side}', (side * 1.014, -.14, 1.26),
                (.035, .25, .04), surfaces['metal'], root, .006)
        wheels(root, width, (-1.57, 1.58), .41, surfaces)
        lights(root, width, length, .80, surfaces)
        mirrors(root, width, -1.24, 1.62, surfaces)
    elif class_name == 'bus':
        width, length = 2.45, 10.6
        box(name + '_body', (0, 0, 1.72), (width, length, 2.43), body, root, .11)
        box(name + '_roof', (0, 0, 2.98), (2.34, 10.25, .23), body, root, .11)
        for side in (-1, 1):
            for index in range(8):
                box(name + f'_window_{side}_{index}', (side * 1.234, -4.05 + index * 1.12, 2.15),
                    (.035, .96, .91), glass, root, .055)
            box(name + f'_belt_{side}', (side * 1.242, 0, 1.48),
                (.035, 10.15, .09), surfaces['darkmetal'], root, .01)
        box(name + '_windscreen', (0, -5.315, 2.11), (2.20, .04, 1.13), glass, root, .09)
        box(name + '_rear_window', (0, 5.315, 2.17), (2.09, .04, .88), glass, root, .07)
        box(name + '_roof_vent', (0, .45, 3.18), (1.36, 2.05, .22), surfaces['trim'], root, .09)
        wheels(root, width, (-3.52, 2.79, 3.70), .50, surfaces)
        lights(root, width, length, .95, surfaces)
        mirrors(root, width, -4.63, 2.16, surfaces)
    else:
        freight = class_name == 'freight_car'
        width, length = (2.50, 12.9) if freight else (2.34, 8.10)
        cab_y = -length / 2 + 1.20
        box(name + '_chassis', (0, 0, .66), (width * .78, length - .40, .34),
            surfaces['darkmetal'], root, .06)
        box(name + '_cab_lower', (0, cab_y, 1.08), (width, 2.32, 1.20), body, root, .085)
        cabin(name + '_cab_upper', width * .98, (cab_y - 1.15, cab_y + 1.13),
              (cab_y - .90, cab_y + .90), 1.45, 2.66 if freight else 2.41, body, root)
        roof_z = 2.54 if freight else 2.29
        polygon(name + '_windscreen', [(-width * .43, cab_y - 1.145, 1.56),
                (width * .43, cab_y - 1.145, 1.56), (width * .37, cab_y - .921, roof_z),
                (-width * .37, cab_y - .921, roof_z)], glass, root)
        for side in (-1, 1):
            polygon(name + f'_cab_window_{side}', [(side * width * .495, cab_y - .89, 1.58),
                    (side * width * .495, cab_y + .87, 1.58),
                    (side * width * .394, cab_y + .79, roof_z),
                    (side * width * .394, cab_y - .72, roof_z)], glass, root)
        if freight:
            cargo = material(name + '_container_paint', (.57, .55, .48), .53, .12)
            box(name + '_container', (0, 1.29, 2.13), (2.43, 9.48, 2.62), cargo, root, .075)
            for side in (-1, 1):
                for rib in range(17):
                    box(name + f'_container_rib_{side}_{rib}', (side * 1.227, -3.10 + rib * .55, 2.13),
                        (.04, .045, 2.39), cargo, root, .01)
            for side in (-1, 1):
                box(name + f'_rear_lock_{side}', (side * .51, 6.05, 2.14),
                    (.045, .035, 2.25), surfaces['metal'], root, .006)
            axles = (-5.16, -2.75, -1.80, 4.65, 5.62)
        else:
            deck_y = 1.20
            box(name + '_cargo_deck', (0, deck_y, 1.10), (2.28, 5.45, .19), body, root, .04)
            for side in (-1, 1):
                box(name + f'_cargo_side_{side}', (side * 1.12, deck_y, 1.61),
                    (.12, 5.45, .94), body, root, .04)
                for rib in range(5):
                    box(name + f'_side_rib_{side}_{rib}', (side * 1.19, -.94 + rib * 1.05, 1.61),
                        (.045, .06, .83), surfaces['trim'], root, .008)
            box(name + '_cargo_back', (0, 3.89, 1.61), (2.28, .12, .94), body, root, .04)
            cargo = material(name + '_cargo', (.30, .29, .25), .75)
            for index in range(3):
                box(name + f'_load_{index}', (0, -.55 + index * 1.5, 1.38),
                    (1.64, 1.1, .40), cargo, root, .05)
            axles = (-2.91, 2.18, 3.10)
        wheels(root, width, axles, .49, surfaces)
        lights(root, width, length, .91, surfaces)
        mirrors(root, width, cab_y - .47, 1.98, surfaces)
    panel_details(root, class_name, width, length, surfaces)
    return root


def settings(scene, width, height, samples):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    if hasattr(scene.render, 'compositor_device'):
        scene.render.compositor_device = 'CPU'
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = .20
    scene['tefed_data_origin'] = 'procedural_vehicle_scene'
    scene.unit_settings.system, scene.unit_settings.scale_length = 'METRIC', 1


def camera_and_light(scene, location, target, scale, world, energy):
    import bpy
    from mathutils import Vector
    scene.world = world
    data = bpy.data.cameras.new(scene.name + '_camera')
    data.type, data.ortho_scale = 'ORTHO', scale
    data.clip_start, data.clip_end = .05, 1000
    camera = bpy.data.objects.new('Camera' if scene.name == 'ParkingLot' else 'CatalogCamera', data)
    scene.collection.objects.link(camera)
    camera.location = location
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    light_data = bpy.data.lights.new(scene.name + '_softbox', 'AREA')
    light_data.energy, light_data.shape, light_data.size = energy * 1.7, 'DISK', 16
    light = bpy.data.objects.new(scene.name + '_softbox', light_data)
    scene.collection.objects.link(light)
    light.location = (-12, -18, 32)
    light.rotation_euler = (Vector(target) - light.location).to_track_quat('-Z', 'Y').to_euler()
    fill_data = bpy.data.lights.new(scene.name + '_fill', 'AREA')
    fill_data.energy, fill_data.shape, fill_data.size = energy * .40, 'DISK', 20
    fill = bpy.data.objects.new(scene.name + '_fill', fill_data)
    scene.collection.objects.link(fill)
    fill.location = (22, 9, 26)
    fill.rotation_euler = (Vector(target) - fill.location).to_track_quat('-Z', 'Y').to_euler()


def parking_environment(surfaces):
    box('Site_base', (0, -4, -.36), (60, 49, .50), surfaces['concrete'], bevel=.28)
    box('Parking_asphalt', (0, 0, -.085), (54, 34, .16), surfaces['asphalt'], bevel=.16)
    box('Access_road', (0, -22, -.065), (60, 8, .14), surfaces['road'], bevel=.08)
    for x in range(-27, 29, 6):
        box(f'Road_center_{x}', (x, -22, .015), (3.1, .13, .018), surfaces['paint_yellow'], bevel=.008)
    for y in (-25.65, -18.35):
        box(f'Road_edge_{y}', (0, y, .017), (59, .12, .018), surfaces['paint_white'], bevel=.006)
    for x in (-22.0, -18.5, -15.0, -11.5, -8.0, -4.5):
        for y in (7.0, -10.0):
            box(f'Bay_side_{x}_{y}', (x, y, .025), (.11, 6.4, .022), surfaces['paint_white'], bevel=.005)
    for y in (10.15, -13.15):
        box(f'Small_bay_end_{y}', (-13.25, y, .025), (17.5, .11, .022), surfaces['paint_white'], bevel=.005)
    for x in (-1.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0):
        box(f'Heavy_bay_{x}', (x, 9.3, .025), (.11, 14.6, .022), surfaces['paint_white'], bevel=.005)
    box('Heavy_bay_end', (11, 16.55, .025), (24, .11, .022), surfaces['paint_white'], bevel=.005)
    for x in range(-20, 24, 6):
        box(f'Aisle_center_{x}', (x, -.9, .025), (2.7, .09, .018), surfaces['paint_yellow'], bevel=.005)
    for x in (-26.65, 26.65):
        box(f'Curb_{x}', (x, 0, .10), (.35, 34, .28), surfaces['concrete'], bevel=.06)
    box('North_curb', (0, 16.70, .10), (53.5, .35, .28), surfaces['concrete'], bevel=.06)
    for x, length in ((-9.0, 29), (23.5, 6)):
        box(f'South_curb_{x}', (x, -16.75, .10), (length, .35, .28), surfaces['concrete'], bevel=.06)
    for index, (x, y) in enumerate(((-24.5, 14.5), (25, 14.5), (-24.5, -14.5), (25, -14.5))):
        cylinder(f'Lamp_base_{index}', (x, y, .10), .34, .20, surfaces['concrete'])
        cylinder(f'Lamp_pole_{index}', (x, y, 2.8), .055, 5.5, surfaces['darkmetal'], bevel=.008)
        box(f'Lamp_arm_{index}', (x - .5, y, 5.51), (1.0, .08, .08), surfaces['darkmetal'], bevel=.01)
        box(f'Lamp_head_{index}', (x - 1, y, 5.48), (.6, .25, .10), surfaces['trim'], bevel=.035)


def inspect_scene(scene, classes):
    roots = [obj for obj in scene.objects if 'tefed_class_name' in obj]
    mapping, counts, instances = {}, Counter(), []
    ownership = set()
    for root in roots:
        class_name = root['tefed_class_name']
        class_id = root['tefed_class_id']
        if classes[class_name] != class_id:
            raise RuntimeError('Scene class ID differs from class map')
        meshes = [obj for obj in root.children_recursive if obj.type == 'MESH']
        if len(meshes) < 10:
            raise RuntimeError('Vehicle asset has too few mesh parts')
        if any(obj.instance_type == 'COLLECTION' for obj in root.children_recursive):
            raise RuntimeError('Vehicle contains a collection instancer')
        if any(obj.name in ownership for obj in meshes):
            raise RuntimeError('Vehicle roots share a mesh object')
        ownership.update(obj.name for obj in meshes)
        mapping[root.name] = class_id
        counts[class_name] += 1
        instances.append({'root': root.name, 'class_name': class_name, 'class_id': class_id,
                          'mesh_parts': len(meshes), 'location_m': list(root.location),
                          'detail_version': root.get('tefed_detail_version', 1),
                          'detail_features': root.get('tefed_detail_features', '').split(';')})
    if set(counts) != set(classes):
        raise RuntimeError('Scene does not contain all five classes')
    if scene.camera is None:
        raise RuntimeError('Scene has no active camera')
    return {'scene_name': scene.name, 'camera': scene.camera.name,
            'counts': dict(counts), 'instances': instances, 'scene_class_map': mapping}


def build(output, classes, samples, width):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parking = bpy.context.scene
    parking.name = 'ParkingLot'
    settings(parking, width, round(width * .70), samples)
    surfaces = {
        'rubber': material('Rubber', (.010, .014, .017), .84),
        'tread': material('Tire_tread', (.024, .028, .031), .79),
        'sidewall': material('Tire_sidewall_detail', (.035, .041, .044), .74),
        'metal': material('Wheel_alloy', (.50, .56, .60), .22, .88),
        'brightmetal': material('Polished_fastener', (.70, .75, .78), .14, .94),
        'darkmetal': material('Dark_trim', (.023, .031, .037), .37, .55),
        'panelgap': material('Panel_gap', (.008, .011, .013), .68),
        'mirror': material('Mirror_lens', (.55, .66, .72), .055, 1),
        'plate': material('Blank_plate', (.77, .79, .75), .39),
        'container_edge': material('Container_roof_ribs', (.57, .57, .53), .43, .22),
        'glass': material('Tinted_glass', (.010, .028, .041), .12, .12),
        'amber': material('Amber_lens', (.91, .31, .025), .20, .05, .10),
        'trim': material('Light_trim', (.58, .61, .61), .40, .2),
        'headlamp': material('Warm_headlamp', (.82, .83, .72), .22, .1, .25),
        'taillamp': material('Red_taillamp', (.39, .025, .022), .22, .1, .15),
        'concrete': material('Concrete', (.44, .46, .44), .9),
        'asphalt': material('Parking_asphalt', (.14, .17, .18), .95),
        'road': material('Road_asphalt', (.10, .12, .13), .95),
        'paint_white': material('Marking_white', (.75, .75, .69), .73),
        'paint_yellow': material('Marking_yellow', (.67, .52, .19), .73),
    }
    world = bpy.data.worlds.new('Soft_sky')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.78, .83, .88, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .45
    parking_environment(surfaces)
    placements = [('car', -20.25, -10, 0), ('car', -16.75, -10, 0),
                  ('car', -13.25, -10, 0), ('car', -9.75, 7, math.pi),
                  ('van', -20.25, 7, math.pi), ('van', -16.75, 7, math.pi),
                  ('truck', 1, 9.1, 0), ('truck', 5, 9.1, 0),
                  ('bus', 9, 9.1, 0), ('bus', 13, 9.1, 0),
                  ('freight_car', 17, 9.1, 0), ('freight_car', 21, 9.1, 0)]
    sequence = Counter()
    for index, (class_name, x, y, heading) in enumerate(placements):
        sequence[class_name] += 1
        root = create_vehicle(class_name, f'{class_name}_{sequence[class_name]:02d}',
                              classes[class_name], PAINTS[index % len(PAINTS)], surfaces)
        root.location, root.rotation_euler.z = (x, y, 0), heading
    camera_and_light(parking, (23, -34, 70), (0, -3.0, .4), 66, world, 13500)
    parking_info = inspect_scene(parking, classes)

    library = bpy.data.scenes.new('Asset_Library')
    bpy.context.window.scene = library
    settings(library, width, round(width * .61), samples)
    box('Catalog_floor', (0, 0, -.14), (34, 20, .26), surfaces['concrete'], bevel=.24)
    for index, class_name in enumerate(DEFAULT_CLASSES):
        root = create_vehicle(class_name, 'catalog_' + class_name, classes[class_name],
                              PAINTS[index], surfaces)
        root.location = (-11 + index * 5.5, 0, 0)
    camera_and_light(library, (8, -35, 33), (0, 0, 1.1), 35, world, 10500)
    library_info = inspect_scene(library, classes)

    bpy.context.window.scene = parking
    blend = output / 'vehicle_parking.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    if bpy.context.scene.name != 'ParkingLot' or bpy.context.scene.camera.name != 'Camera':
        raise RuntimeError('Saved file did not reopen in the parking-lot scene')
    reopened = inspect_scene(bpy.context.scene, classes)
    if reopened['scene_class_map'] != parking_info['scene_class_map']:
        raise RuntimeError('Instance mapping changed after reopening the scene')
    external = [image.filepath for image in bpy.data.images
                if image.source == 'FILE' and image.filepath and image.packed_file is None]
    if external:
        raise RuntimeError('Scene has external image dependencies')
    previews = (('Asset_Library', 'vehicle_catalog.png'), ('ParkingLot', 'parking_lot.png'))
    preview_resolution, preview_samples = {}, {}
    for scene_name, filename in previews:
        scene = bpy.data.scenes[scene_name]
        render = scene.render
        scale = render.resolution_percentage / 100
        preview_resolution[filename] = [int(render.resolution_x * scale),
                                        int(render.resolution_y * scale)]
        preview_samples[filename] = scene.cycles.samples
    metadata = {'data_origin': 'procedural_vehicle_scene', 'active_scene': 'ParkingLot',
                'blender_version': bpy.app.version_string,
                'generator': {'script': Path(__file__).name,
                              'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                'preview_resolution': preview_resolution, 'preview_samples': preview_samples,
                'blend': blend.name, 'class_map': classes,
                'class_names': [key for key, _ in sorted(classes.items(), key=lambda item: item[1])],
                'unit': 'metre', 'geometry': 'Original generic vehicle assets',
                'detail_level': 'vehicle_detail_v2',
                'glass_geometry': 'Planes projected to cabin faces with 14 mm outward seal offset',
                'parking_lot': parking_info, 'asset_library': library_info,
                'scene_class_map': parking_info['scene_class_map'],
                'renderer': {'engine': 'CYCLES', 'device': 'CPU', 'compositor_device': 'CPU', 'samples': samples},
                'reopened_validation': {'passed': True, 'external_images': external,
                                        'vehicle_roots': len(reopened['instances'])}}
    (output / 'scene.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    mapping_lines = '\n'.join(f'{name} = {value}' for name, value in parking_info['scene_class_map'].items())
    config = ('[renderer]\nenabled = true\nblender = "blender"\nscene = "' + blend.as_posix()
              + '"\nsamples = 16\n\n[renderer.scene_class_map]\n' + mapping_lines + '\n')
    (output / 'renderer_config.toml').write_text(config, encoding='utf-8')
    request = {'descriptor': {'source_class': classes['car'], 'attrs': [.35, .15, 0, .15, 0, 0, 0, 0]},
               'seed': 0, 'case_id': 'vehicle_scene_check', 'scene_path': str(blend),
               'scene_class_map': parking_info['scene_class_map'], 'image_size': [640, 640]}
    (output / 'render_request.json').write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Editable vehicle scene\n\n'
        'Open `vehicle_parking.blend` in Blender. `ParkingLot` is the startup scene; '
        '`Asset_Library` contains one asset of each class. All vehicle parts are native meshes '
        'under independent instance roots. Materials have no external textures.\n\n'
        'The parking lot contains 12 vehicles, road markings, curbs and lighting. '
        'Vehicle detail includes door and hood gaps, window seals, wipers, mirror lenses, '
        'layered lamps, grilles, tire tread, wheel spokes and fasteners. '
        'Use `scene.json` for class IDs and object roots; `renderer_config.toml` and '
        '`render_request.json` connect the scene to the generic renderer.\n\n'
        'These are original, generic procedural vehicle assets.\n', encoding='utf-8')
    (output / 'visual_design.md').write_text(
        '# Quiet geometry\n\n'
        'Vehicle proportions carry the class distinction: a low sedan, tall van, '
        'open-bed truck, long windowed bus and articulated freight carrier. Separate '
        'meshes keep the shape and material decisions editable.\n\n'
        'Paint stays within muted blue, warm grey and deep red. Dark glass and rubber '
        'contrast with restrained metallic wheel details. Soft lighting makes these '
        'material boundaries readable without hard highlights.\n\n'
        'Parking bays and road markings establish scale and a regular rhythm. Gaps '
        'between vehicles preserve their silhouettes; the camera keeps the scene '
        'close to an aerial observation.\n\n'
        'The catalog holds one example per class at a common geometric scale. The '
        'parking view gives context with twelve vehicles and limited street furniture. '
        'No labels cover the vehicle bodies.\n', encoding='utf-8')
    for scene_name, filename in previews:
        scene = bpy.data.scenes[scene_name]
        bpy.context.window.scene = scene
        scene.render.filepath = str(output / filename)
        bpy.ops.render.render(write_still=True)
    return metadata


def main(argv=None):
    arguments = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh directory for the blend and previews')
    parser.add_argument('--class-map', help='Five-category class ID JSON')
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--preview-width', type=int, default=1600)
    parser.add_argument('--check', action='store_true', help='Validate arguments without Blender or file creation')
    args = parser.parse_args(arguments)
    try:
        output = Path(args.output).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError('Choose a fresh output directory')
        if not 1 <= args.samples <= 256 or not 640 <= args.preview_width <= 4096:
            raise ValueError('Samples must be 1-256 and preview width 640-4096')
        classes = read_classes(args.class_map)
        if args.check:
            result = {'output': str(output), 'class_map': classes, 'samples': args.samples,
                      'preview_width': args.preview_width}
        else:
            import bpy
            output.mkdir(parents=True, exist_ok=False)
            result = build(output, classes, args.samples, args.preview_width)
    except ModuleNotFoundError as error:
        if error.name != 'bpy':
            raise
        parser.error('Run scene generation with Blender; --check also works with Python')
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
