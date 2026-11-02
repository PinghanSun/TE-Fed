# Rendering scenes

Blender scripts create editable scenes, input arrays and instance annotations.

The [2.6.4 release](https://github.com/autumnyfot1898-bot/TE-Fed/releases/tag/v2.6.4) includes a ready-to-open vehicle and industrial-park scene with paired RGB/IR examples and run scripts.

## Vehicle models and parking scene

For five editable vehicle types and a 12-instance parking scene:

```bash
blender --background --factory-startup --python scripts/create_vehicle_scene.py -- \
  --output runs/vehicle_scene --class-map examples/dronevehicle/class_map.json
```

`vehicle_parking.blend` contains `ParkingLot` and `Asset_Library` scenes, with editable bodywork, panel seams, windows, lights and wheel details. Outputs include two PNG previews, `scene.json`, `renderer_config.toml` and `render_request.json`.

To inspect a vehicle from several angles:

```bash
blender --background runs/vehicle_scene/vehicle_parking.blend \
  --python scripts/render_vehicle_views.py -- \
  --class-name car --output runs/vehicle_views
```

The command renders front, side, rear and wheel views. `--class-name` selects the vehicle type.

To add an industrial-park environment:

```bash
blender --background runs/vehicle_scene/vehicle_parking.blend \
  --python scripts/enhance_vehicle_environment.py -- \
  --output runs/vehicle_environment
```

`vehicle_environment.blend` adds buildings, planting, paving, drainage and entrance furniture under `Environment_Detail`. It includes aerial, street and ground-detail previews; `environment.json` records the source and checks.

Render the full environment example with this single-line command, including in PowerShell:

```text
python scripts/render_scene_example.py --request runs/vehicle_environment/render_request.json --output runs/environment_check --blender blender
```

Use the Blender executable's full path if it is not on `PATH`. Add `--check` to validate the paths without rendering. Relative `scene_path` values are resolved from the request file's directory. Outputs contain RGB, depth, the IR cue, instance masks and boxes.

## Surface refinement and material-based infrared

Refine the vehicle surfaces and environment:

```text
blender --background --factory-startup --python scripts/refine_vehicle_assets.py -- --input runs/vehicle_environment/vehicle_environment.blend --output runs/refined_vehicles --no-preview
blender --background runs/refined_vehicles/vehicle_refined.blend --python scripts/refine_vehicle_environment.py -- --output runs/refined_environment
```

`vehicle_parameters.json` stores design dimensions, panel gaps and tyre grooves. Use `reference_template.json` for reference dimensions and views.

For an independent material-based IR preview with a paired RGB image:

```text
python scripts/run_ir_example.py --scene runs/refined_environment/vehicle_environment.blend --preset lwir --output runs/lwir --blender blender
```

Use `--preset nir` for reflected near-infrared light. The [IR guide](infrared.md) describes spectral response, material and calibration parameters. `render_scene_example.py` writes the luminance cue; `run_ir_example.py` uses material spectra. Original scene assets and previews use the [asset license](../ASSET_LICENSE.md).

## Generate a scene

Run from the repository root with Blender installed:

```bash
blender --background --factory-startup --python scripts/create_example_scene.py -- \
  --task dronevehicle --output runs/scenes/dronevehicle.blend
```

Use `--task tomato` for two fruit instances. Each command writes a new `.blend`, `.scene.json` and `.request.json`; existing files are preserved.

The default sample maps are:

| Task | Class IDs | Object roots |
| --- | --- | --- |
| Tomato | `non-ripe=0`, `ripe=1` | `non_ripe`, `ripe` |
| DroneVehicle | `car=0`, `truck=1`, `bus=2`, `van=3`, `freight_car=4` | Matching class names |

Pass `--class-map path/to/class_map.json` to use the training manifest's class IDs. The JSON maps the listed category names to unique consecutive IDs. Scene metadata records both the class order and the root-to-ID map. Check arguments without Blender or file creation using `--check`.

## Connect the scene

Copy the emitted `scene_class_map` into the local training configuration. For the default DroneVehicle sample:

```toml
[renderer]
enabled = true
blender = "blender"
scene = "runs/scenes/dronevehicle.blend"

[renderer.scene_class_map]
car = 0
truck = 1
bus = 2
van = 3
freight_car = 4
```

For Tomato, use `non_ripe=0` and `ripe=1`, or the IDs from the chosen class map. A mapped object root and its child meshes form one annotated instance. Roots must be disjoint, visible and uniquely named. `object:` and `collection:` prefixes are supported when names overlap.

The example sets one scene unit to one metre, a 640×640 orthographic camera named `Camera`, a ground plane and an area light. `--resolution WIDTH HEIGHT` changes the saved size. Render requests can override image size. Set `renderer.depth_unit_conversion` from scene units to the dataset's depth units for training. The generated NIR/IR cue uses rendered luminance and noise, as in the renderer implementation.

## Check a render

The emitted request is ready for the packaged generic renderer:

```bash
mkdir -p runs/render_check
blender --background --factory-startup \
  --python src/tefed/blender_scripts/blender_generic_render.py -- \
  --request runs/scenes/dronevehicle.request.json \
  --response runs/render_check/response.json
```

The CPU renderer returns aligned arrays, visible boxes and measured attributes.
