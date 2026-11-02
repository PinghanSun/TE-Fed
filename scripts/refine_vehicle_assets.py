"""Refine native vehicle shells, panel seams and road wheels in a fresh blend."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


CLASSES = ('car', 'truck', 'bus', 'van', 'freight_car')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def bounds(root):
    from mathutils import Vector
    points = []
    for obj in root.children_recursive:
        if obj.type != 'MESH':
            continue
        matrix = obj.matrix_parent_inverse @ obj.matrix_basis
        ancestor = obj.parent
        while ancestor is not root:
            matrix = ancestor.matrix_parent_inverse @ ancestor.matrix_basis @ matrix
            ancestor = ancestor.parent
        points.extend(matrix @ Vector(corner) for corner in obj.bound_box)
    return [[min(point[i] for point in points), max(point[i] for point in points)]
            for i in range(3)]


def dimensions(root):
    box = bounds(root)
    return {'length_m': round((box[1][1] - box[1][0]) * abs(root.scale.y), 6),
            'width_m': round((box[0][1] - box[0][0]) * abs(root.scale.x), 6),
            'height_m': round((box[2][1] - box[2][0]) * abs(root.scale.z), 6)}


def native_mesh(root, name, vertices, faces, surface, old=None):
    import bpy
    import bmesh
    from mathutils import Matrix
    mesh = bpy.data.meshes.new(name + '_surface')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    if old is None:
        obj = bpy.data.objects.new(name, mesh)
        root.users_collection[0].objects.link(obj)
        obj.parent = root
    else:
        obj = old
        obj.data = mesh
        obj.modifiers.clear()
    obj.matrix_local = Matrix.Identity(4)
    obj.data.materials.append(surface)
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def bevel(obj, width=.012):
    modifier = obj.modifiers.new('Manufactured edge', 'BEVEL')
    modifier.width, modifier.segments = width, 3
    modifier = obj.modifiers.new('Panel normals', 'WEIGHTED_NORMAL')
    modifier.keep_sharp = True


def loft(root, old, sections, bottom, crown=.008):
    vertices, faces = [], []
    for y, width, top in sections:
        height = top - bottom
        ring = [(-.78 * width, bottom), (.78 * width, bottom),
                (.95 * width, bottom + .07 * height), (width, bottom + .21 * height),
                (width, bottom + .66 * height), (.98 * width, bottom + .86 * height),
                (.88 * width, top), (.47 * width, top + crown),
                (0, top + crown * 1.30), (-.47 * width, top + crown),
                (-.88 * width, top), (-.98 * width, bottom + .86 * height),
                (-width, bottom + .66 * height), (-width, bottom + .21 * height),
                (-.95 * width, bottom + .07 * height)]
        vertices.extend((x, y, z) for x, z in ring)
    size = 15
    for row in range(len(sections) - 1):
        for column in range(size):
            following = (column + 1) % size
            faces.append((row * size + column, row * size + following,
                          (row + 1) * size + following, (row + 1) * size + column))
    faces.extend([tuple(reversed(range(size))),
                  tuple((len(sections) - 1) * size + column for column in range(size))])
    return native_mesh(root, old.name, vertices, faces, old.data.materials[0], old)


def wheel_cutouts(root, body, wheels):
    import bpy
    for y, radius in sorted(set((round(y, 5), round(radius, 5)) for _, y, radius in wheels)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius * 1.105,
                                           depth=4.0, location=(0, 0, 0))
        cutter = bpy.context.object
        cutter.name = 'Temporary_wheel_opening'
        cutter.parent = root
        cutter.location = (0, y, radius)
        cutter.rotation_euler = (0, math.pi / 2, 0)
        bpy.context.view_layer.update()
        modifier = body.modifiers.new('Wheel opening', 'BOOLEAN')
        modifier.operation, modifier.solver, modifier.object = 'DIFFERENCE', 'EXACT', cutter
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    bevel(body)


def lathe_wheel(root, name, center, profile, surface, old=None, segments=96, closed=True):
    x, y, z = center
    vertices, faces = [], []
    for axial, radius in profile:
        for index in range(segments):
            angle = math.tau * index / segments
            vertices.append((x + axial, y + radius * math.sin(angle), z + radius * math.cos(angle)))
    for row in range(len(profile) - 1):
        for index in range(segments):
            following = (index + 1) % segments
            faces.append((row * segments + index, row * segments + following,
                          (row + 1) * segments + following, (row + 1) * segments + index))
    if closed:
        last = (len(profile) - 1) * segments
        faces.extend((last + index, last + (index + 1) % segments,
                      (index + 1) % segments, index) for index in range(segments))
    else:
        faces.extend([tuple(reversed(range(segments))),
                      tuple((len(profile) - 1) * segments + index for index in range(segments))])
    return native_mesh(root, name, vertices, faces, surface, old)


def road_wheels(root, groove_depth):
    import bpy
    tires = [obj for obj in root.children_recursive if obj.name.endswith('_rubber')]
    wheels = []
    for tire in tires:
        prefix = tire.name[:-len('_rubber')]
        side = int(prefix.rsplit('_', 1)[1])
        radius = max(math.hypot(v.co.x, v.co.y) for v in tire.data.vertices)
        for part in [obj for obj in root.children_recursive if obj.name.startswith(prefix + '_')]:
            part.location.x -= side * .065
        center = tuple(tire.location)
        wheels.append((center[0], center[1], radius))
        profile = [(-.145, .67 * radius), (-.148, .82 * radius), (-.143, .91 * radius),
                   (-.129, .971 * radius), (-.108, radius)]
        for axial in (-.075, -.025, .025, .075):
            profile.extend([(axial - .006, radius), (axial - .003, radius - groove_depth),
                            (axial + .003, radius - groove_depth), (axial + .006, radius)])
        profile.extend([(.108, radius), (.129, .971 * radius), (.143, .91 * radius),
                        (.148, .82 * radius), (.145, .67 * radius)])
        lathe_wheel(root, tire.name, center, profile, tire.data.materials[0], tire)
        for suffix in ('_tread_blocks', '_five_spokes', '_wheel_arch'):
            part = bpy.data.objects.get(prefix + suffix)
            if part is not None:
                bpy.data.objects.remove(part, do_unlink=True)
        lip = bpy.data.objects.get(prefix + '_rim_lip')
        surface = lip.data.materials[0]
        face_x = center[0] + side * .172
        lip_profile = [(side * .163, .56 * radius), (side * .171, .58 * radius),
                       (side * .181, .585 * radius), (side * .185, .57 * radius)]
        lathe_wheel(root, lip.name, center, lip_profile, surface, lip)
        vertices, faces = [], []
        for index in range(5):
            angle = math.tau * index / 5
            for branch in (-1, 1):
                polygon = []
                for distance, offset in ((.18, -.050), (.36, -.042), (.58, branch * .12 - .025),
                                         (.58, branch * .12 + .025), (.36, .042), (.18, .050)):
                    theta = angle + offset
                    polygon.append((center[1] + radius * distance * math.sin(theta),
                                    center[2] + radius * distance * math.cos(theta)))
                start = len(vertices)
                for depth in (-.010, .010):
                    vertices.extend((face_x + depth, y, z) for y, z in polygon)
                faces.extend([tuple(start + i for i in reversed(range(6))),
                              tuple(start + 6 + i for i in range(6))])
                faces.extend((start + i, start + (i + 1) % 6, start + (i + 1) % 6 + 6,
                              start + i + 6) for i in range(6))
        spoke = native_mesh(root, prefix + '_split_spokes', vertices, faces, surface)
        bevel(spoke, .002)
        ring = bpy.data.objects.get(prefix + '_sidewall_line')
        if ring is not None:
            ring_profile = [(side * .145, .83 * radius), (side * .147, .835 * radius),
                            (side * .145, .84 * radius)]
            lathe_wheel(root, ring.name, center, ring_profile, ring.data.materials[0], ring)
        for part in [obj for obj in root.children_recursive if obj.name.startswith(prefix + '_')]:
            if part.name.endswith(('_hub', '_rim')):
                for modifier in part.modifiers:
                    if modifier.type == 'BEVEL':
                        modifier.width = min(modifier.width, .006)
    return wheels


def refine_shells(root, wheels):
    import bpy
    class_name = root['tefed_class_name']
    candidates = []
    def shell(suffix, sections, bottom, cut=True, crown=.008):
        old = bpy.data.objects.get(root.name + suffix)
        obj = loft(root, old, sections, bottom, crown)
        wheel_cutouts(root, obj, wheels) if cut else bevel(obj, .008)
        candidates.append(obj)
    if class_name == 'car':
        shell('_lower_body', [(-2.25, .78, .84), (-2.13, .86, .91), (-1.82, .916, .945),
              (-1.2, .92, .95), (0, .92, .95), (1.25, .92, .94),
              (1.83, .90, .92), (2.13, .84, .88), (2.25, .77, .82)], .34)
        shell('_hood', [(-2.15, .72, .945), (-1.97, .81, .985), (-1.49, .865, 1.00),
                       (-1.04, .82, .975), (-.815, .77, .946)], .865, False, .006)
    elif class_name == 'van':
        shell('_body', [(-2.35, .86, 1.36), (-2.23, .94, 1.43), (-1.85, .99, 1.45),
              (1.96, .99, 1.45), (2.40, .97, 1.43), (2.55, .90, 1.35)], .42)
    elif class_name == 'bus':
        shell('_body', [(-5.30, 1.11, 2.84), (-5.17, 1.19, 2.91), (-4.87, 1.225, 2.935),
              (4.87, 1.225, 2.935), (5.17, 1.19, 2.91), (5.30, 1.10, 2.84)], .505, crown=.004)
    else:
        length, width = (12.9, 2.50) if class_name == 'freight_car' else (8.1, 2.34)
        cab_y = -length / 2 + 1.2
        shell('_cab_lower', [(cab_y - 1.16, width * .43, 1.40),
              (cab_y - 1.06, width * .48, 1.49), (cab_y - .75, width / 2, 1.52),
              (cab_y + .85, width / 2, 1.52), (cab_y + 1.16, width * .46, 1.43)], .48)
    for obj in root.children_recursive:
        if obj.type != 'MESH' or obj in candidates:
            continue
        if any(obj.name == root.name + suffix for suffix in ('_cabin', '_roof', '_cab_upper', '_container')):
            candidates.append(obj)
            if obj.name.endswith(('_cabin', '_cab_upper')):
                for modifier in obj.modifiers:
                    if modifier.type == 'BEVEL':
                        modifier.segments = 5
                for face in obj.data.polygons:
                    face.use_smooth = True
        elif '_cargo_side_' in obj.name or obj.name.endswith('_cargo_back'):
            candidates.append(obj)
    return candidates


def fine_seams(root, shells, radius):
    import bpy
    from mathutils import Vector
    gaps = bpy.data.objects.get(root.name + '_panel_gaps')
    if gaps is not None:
        for start in range(0, len(gaps.data.vertices), 16):
            for offset in (0, 8):
                ring = list(gaps.data.vertices)[start + offset:start + offset + 8]
                center = sum((vertex.co for vertex in ring), Vector()) / 8
                candidates = []
                for shell in shells:
                    inverse = shell.matrix_local.inverted()
                    hit, point, normal, _ = shell.closest_point_on_mesh(inverse @ center)
                    if hit:
                        point = shell.matrix_local @ point
                        normal = (shell.matrix_local.to_3x3() @ normal).normalized()
                        candidates.append(((point - center).length, point, normal))
                if candidates:
                    _, point, normal = min(candidates, key=lambda item: item[0])
                    projected = point + normal * radius * .35
                    for vertex in ring:
                        vertex.co = projected + (vertex.co - center) * (radius / .006)
        gaps.data.update()
    for obj in root.children_recursive:
        if not obj.name.endswith(('_window_seals', '_bus_window_frames', '_wiper_blades', '_bus_wipers')):
            continue
        for start in range(0, len(obj.data.vertices), 16):
            for offset in (0, 8):
                ring = list(obj.data.vertices)[start + offset:start + offset + 8]
                center = sum((vertex.co for vertex in ring), Vector()) / 8
                for vertex in ring:
                    vertex.co = center + (vertex.co - center) * .55


def fit_rear_lamps(root):
    if root.get('tefed_class_name') != 'car':
        return []
    suffixes = ('_taillamp_-1', '_taillamp_1', '_red_lens_-1', '_red_lens_1',
                '_reverse_lens_-1', '_reverse_lens_1')
    parts = [obj for obj in root.children_recursive if obj.name in
             {root.name + suffix for suffix in suffixes}]
    if len(parts) != 6:
        raise RuntimeError(f'{root.name} requires six rear-lamp parts')
    for obj in parts:
        obj.location.z -= .11
    return [obj.name for obj in parts]


def materials():
    import bpy
    for surface in bpy.data.materials:
        if not surface.use_nodes:
            continue
        shader = surface.node_tree.nodes.get('Principled BSDF')
        if shader is None:
            continue
        name = surface.name.lower()
        semantic = ('painted_metal' if name.startswith('paint_') or 'container_paint' in name else
                    'rubber' if any(token in name for token in ('rubber', 'tire', 'panel_gap')) else
                    'glass' if any(token in name for token in ('glass', 'lens', 'headlamp', 'taillamp')) else
                    'metal' if any(token in name for token in ('metal', 'alloy', 'fastener', 'trim', 'plate', 'corrugation')) else None)
        if semantic is not None:
            surface['tefed_ir_material'] = semantic
        if surface.name.startswith('Paint_'):
            shader.inputs['Metallic'].default_value = .46
            shader.inputs['Roughness'].default_value = .255
            shader.inputs['Coat Weight'].default_value = .58
            shader.inputs['Coat Roughness'].default_value = .20
            for node in surface.node_tree.nodes:
                if node.type == 'BUMP':
                    node.inputs['Strength'].default_value = .008
                    node.inputs['Distance'].default_value = .0015
        elif surface.name in ('Rubber', 'Tire_tread', 'Tire_sidewall_detail'):
            shader.inputs['Base Color'].default_value = (.018, .021, .024, 1)
            shader.inputs['Roughness'].default_value = .68
        elif surface.name == 'Panel_gap':
            shader.inputs['Base Color'].default_value = (.030, .034, .038, 1)
        elif surface.name == 'Wheel_alloy':
            shader.inputs['Roughness'].default_value = .29


def validate_parameters(value):
    if not isinstance(value, dict) or set(value.get('classes', {})) != set(CLASSES):
        raise ValueError('Parameters require all five vehicle classes')
    for item in value['classes'].values():
        for key in ('length_m', 'width_m', 'height_m'):
            if type(item.get(key)) not in (int, float) or not math.isfinite(item[key]) or item[key] <= 0:
                raise ValueError('Each vehicle requires positive finite length_m, width_m and height_m')
    for key, minimum, maximum in (('panel_seam_radius_m', .001, .004), ('tire_groove_depth_m', .002, .010)):
        if not minimum <= value.get(key, 0) <= maximum:
            raise ValueError(f'{key} must be between {minimum} and {maximum}')
    return value


def main(argv=None):
    arguments = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parameters', help='Optional vehicle_parameters.json with editable design dimensions')
    parser.add_argument('--preview-width', type=int, default=1800)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args(arguments)
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if not source.is_file() or output.exists():
        parser.error('Input must exist and output must be a fresh directory')
    if not 640 <= args.preview_width <= 4096 or not 1 <= args.samples <= 128:
        parser.error('Preview width must be 640-4096 and samples 1-128')
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    roots = [obj for obj in bpy.data.objects if 'tefed_class_name' in obj]
    before = {root.name: {'class': root['tefed_class_name'], 'id': root['tefed_class_id'],
                          'location': list(root.location), 'rotation': list(root.rotation_euler)} for root in roots}
    defaults = {'dimension_basis': 'Original asset design bounds in metres, including mirrors and lamps',
                'panel_seam_radius_m': .0023, 'tire_groove_depth_m': .0045, 'classes': {}}
    for root in roots:
        defaults['classes'].setdefault(root['tefed_class_name'], dimensions(root))
    parameters = validate_parameters(json.loads(Path(args.parameters).read_text(encoding='utf-8'))
                                     if args.parameters else defaults)
    output.mkdir(parents=True, exist_ok=False)
    for scene in bpy.data.scenes:
        bpy.context.window.scene = scene
        for root in [obj for obj in scene.objects if 'tefed_class_name' in obj]:
            wheel_data = road_wheels(root, parameters['tire_groove_depth_m'])
            shells = refine_shells(root, wheel_data)
            fine_seams(root, shells, parameters['panel_seam_radius_m'])
            fit_rear_lamps(root)
            root['tefed_detail_version'] = 3
            root['tefed_detail_features'] += ';sculpted_shell;wheel_openings;fine_surface_seams;road_tire_grooves;split_spokes'
    materials()
    for scene in bpy.data.scenes:
        scene['asset_license'] = 'CC-BY-4.0'
        scene['asset_attribution'] = 'TE-Fed vehicle and industrial-park example assets, TE-Fed project contributors, 2026'
    bpy.context.view_layer.update()
    for root in roots:
        desired = parameters['classes'][root['tefed_class_name']]
        present = dimensions(root)
        root.scale = (desired['width_m'] / present['width_m'], desired['length_m'] / present['length_m'],
                      desired['height_m'] / present['height_m'])
    parking = bpy.data.scenes['ParkingLot']
    bpy.context.window.scene = parking
    bpy.context.view_layer.update()
    mapping = {root.name: int(root['tefed_class_id']) for root in parking.objects if 'tefed_class_name' in root}
    if len(mapping) != 12 or set(mapping.values()) != set(range(5)):
        raise RuntimeError('The parking scene must preserve 12 roots and five classes')
    for root in roots:
        record = before[root.name]
        if list(root.location) != record['location'] or list(root.rotation_euler) != record['rotation']:
            raise RuntimeError('Vehicle placement changed')
    blend = output / 'vehicle_refined.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    script = Path(__file__)
    report = {'data_origin': 'procedural_vehicle_scene', 'detail_level': 'vehicle_detail_v3',
              'blender_version': bpy.app.version_string, 'source_blend': source.name,
              'source_blend_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'generator': {'script': script.name, 'sha256': hashlib.sha256(script.read_bytes()).hexdigest()},
              'blend': blend.name, 'unit': 'metre', 'scene_class_map': mapping,
              'asset_license': 'CC-BY-4.0', 'asset_attribution': 'TE-Fed project contributors, 2026',
              'vehicle_parameters': parameters,
              'rear_lamp_fit': {'class': 'car', 'offset_z_m': -.11, 'parts_per_root': 6},
              'preview': {'filename': 'vehicle_refined_preview.png', 'resolution': [args.preview_width, round(args.preview_width * .65)],
                          'samples': args.samples},
              'checks': {'root_count': len(mapping), 'class_ids': sorted(set(mapping.values())),
                         'placements_unchanged': True, 'reopened': True,
                         'collection_instancers': sum(obj.instance_type == 'COLLECTION' for obj in bpy.data.objects),
                         'external_images': [image.filepath for image in bpy.data.images if image.source == 'FILE'
                                             and image.filepath and image.packed_file is None]},
              'dimensions_m': {root.name: dimensions(root) for root in bpy.data.objects if 'tefed_class_name' in root}}
    write_json(output / 'scene.json', report)
    write_json(output / 'vehicle_parameters.json', parameters)
    write_json(output / 'reference_template.json', {'unit': 'metre', 'classes': {
        name: {'reference_name': None, 'source_url': None, 'source_document': None, 'access_date': None,
               'length_m': None, 'width_m': None, 'height_m': None,
               'width_includes_mirrors': None, 'front_image': None, 'side_image': None, 'rear_image': None}
        for name in CLASSES}})
    (output / 'visual_design.md').write_text(
        '# Quiet surfaces\n\n'
        'Continuous body shoulders replace block edges. Wheel openings, slender panel seams and curved '
        'road-tire profiles provide the main scale cues. Craft is expressed through surface alignment, '
        'not a larger number of decorations.\n\n'
        'Muted paint, dark rubber and satin alloy remain distinct under broad soft lighting. Fine '
        'clear-coat reflections describe the metal panels without hiding their geometry.\n\n'
        'Repeated wheel and door details share one restrained scale across the five classes. Accurate '
        'alignment and small edge radii carry the visual precision.\n\n'
        'The preview keeps the body and wheels in the foreground. No labels cover the assets; the native '
        'scenes preserve the full parking composition and individual editable vehicle roots.\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Refined vehicle assets\n\nOpen `vehicle_refined.blend`. The five classes and twelve parking instances remain editable.\n\n'
        '`vehicle_parameters.json` controls length, width, height, panel-seam radius and tire-groove depth. '
        'Dimensions describe the original asset design in metres, including mirrors and lamps. '
        '`reference_template.json` accepts actual dimensions, image paths and source details.\n', encoding='utf-8')
    print('GEOMETRY_FROZEN ' + str(blend), flush=True)
    if not args.no_preview:
        scene = bpy.data.scenes['Asset_Library']
        bpy.context.window.scene = scene
        root = bpy.data.objects['catalog_car']
        for other in [obj for obj in scene.objects if 'tefed_class_name' in obj and obj != root]:
            for part in other.children_recursive:
                part.hide_render = True
        camera = scene.camera
        target = root.matrix_world @ Vector((0, -.10, .80))
        camera.location = target + Vector((5.6, -7.8, 4.7))
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.type, camera.data.ortho_scale = 'ORTHO', 6.4
        scene.render.engine, scene.cycles.device = 'CYCLES', 'CPU'
        scene.cycles.samples, scene.cycles.use_denoising = args.samples, True
        if hasattr(scene.render, 'compositor_device'):
            scene.render.compositor_device = 'CPU'
        scene.render.resolution_x = args.preview_width
        scene.render.resolution_y = round(args.preview_width * .65)
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(output / 'vehicle_refined_preview.png')
        bpy.ops.render.render(write_still=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
