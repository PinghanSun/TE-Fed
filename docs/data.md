# Data preparation

| Dataset | Source | Inputs | Configuration |
| --- | --- | --- | --- |
| Tomato | [China Scientific Data](https://doi.org/10.11922/11-6035.csd.2023.0154.zh) | RGB, depth, NIR | [tomato.toml](../configs/tomato.toml) |
| DroneVehicle | [Official repository](https://github.com/VisDrone/DroneVehicle) | RGB, IR | [dronevehicle.toml](../configs/dronevehicle.toml) |

Download data from the source and keep images outside the repository. Full datasets and trained checkpoints are separate from this source package.

## Class IDs

Set `class_names` in each data manifest as a list in zero-based class-ID order. This order is copied to the training configuration and checkpoint. All splits, labels, render mappings and display names must use it.

The supplied Tomato class map uses `['non-ripe', 'ripe']`. Class IDs must match the checkpoint. For DroneVehicle, provide a name-to-ID JSON containing its five training classes; use the category spelling in your annotation files.

## Tomato

Pair RGB, depth and NIR by observation ID. RGB is uint8. Under `DEPTH-MM-v1`, dataset depth is uint16 in millimetres, with zero marking invalid pixels. Loading preserves a validity mask and uses float32 values divided by the fixed `model.depth_scale=65535.0`, then clipped to `[0,1]`. Invalid pixels remain zero. NIR is uint8 or a floating-point `.npy` array in `[0,1]`. Prepare 16-bit NIR using the training normalization before loading it. The input example's colors and percentile limits are for display.

Rendered depth is float32 axial camera-Z in metres. It is multiplied by `renderer.depth_unit_conversion=1000.0` once before the same network normalization. Keep it in floating point during conversion. The [rendering guide](rendering.md#depth-convention) defines scene scale and geometry.

The manifest contains image paths, client IDs, explicit splits, class names, calibration IDs and evaluation-view IDs. See the [manifest fields](training.md#data-manifest) and [paired input example](../examples/tomato_trimodal/README.md).

## DroneVehicle

Keep RGB and single-channel IR files under separate directories with matching IDs. The loader accepts uint8 IR or floating-point `.npy` values in `[0,1]`. Its manifest field is named `nir`; for this task it carries IR. RGB feeds the detector and IR supplies auxiliary cues.

Some original IR JPEGs store the same intensity in three channels. Verify exact equality, retain one channel and save uint8 PNG under `prepared_ir/`. Keep the original files. A color image with unequal channels requires its documented sensor preprocessing.

The converter reads native polygon XML or eight-corner text annotations and creates minimum enclosing horizontal boxes. Use annotations in the selected RGB image's coordinate system. It preserves image geometry. The official data include a 100-pixel border; perform any chosen crop, registration and annotation translation together before conversion.

```bash
uv run --locked python scripts/prepare_manifest.py \
  --input /data/DroneVehicle/labels --input-format dronevehicle \
  --image-root /data/DroneVehicle/rgb --ir-root /data/DroneVehicle/prepared_ir \
  --image-suffix .jpg --ir-suffix .png --class-map classes.json \
  --class-aliases examples/dronevehicle/class_aliases.json \
  --client-id client1 --split labeled --output data/dronevehicle/client1.json --check-files
```

Set `--difficult-policy include` or `drop` explicitly when annotations contain difficult objects, using the same policy throughout the study. With `--ir-root`, every selected image must have a valid matching IR file. Omitting it imports RGB-only records.

`--class-aliases` explicitly maps source spellings to the names in your class map. The supplied example maps `feright car`, `feright_car` and `freight car` to `freight_car`; omit or replace it for other label vocabularies.

Convert each declared client and split, then combine the records with your calibration and evaluation ID lists. Keep synchronized inputs and observations from the same acquisition group together. The converter preserves the assignments supplied to it.

Use the [RGB/IR example](../examples/dronevehicle/README.md) to inspect prepared inputs. For correction rendering, follow the [scene interface](rendering.md).
