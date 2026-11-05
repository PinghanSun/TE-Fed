# Data preparation

| Dataset | Source | Inputs | Configuration |
| --- | --- | --- | --- |
| Tomato | [Zhang et al., China Scientific Data (2025)](https://doi.org/10.11922/11-6035.csd.2023.0154.zh) | RGB, depth, NIR | [tomato.toml](../configs/tomato.toml) |
| DroneVehicle | [VisDrone/DroneVehicle](https://github.com/VisDrone/DroneVehicle) | RGB, IR | [dronevehicle.toml](../configs/dronevehicle.toml) |

Download images from the providers and keep datasets outside the repository. Dataset samples retain their providers' usage terms.

## Manifest and classes

Create a JSON manifest with `class_names`, `records`, `calibration_ids` and `views`:

- `class_names`: names in zero-based class-ID order. Tomato uses `non-ripe, ripe`; the vehicle example uses `car, truck, bus, van, freight_car`. Keep this order consistent with the checkpoint and renderer.
- Each record: unique `id`, `client_id`, `split`, `rgb`, optional `depth`/`nir`, and `labels`. Image paths are relative to the manifest or absolute.
- Each label: `class_id` and normalized `[x_min, y_min, x_max, y_max]` box. Use an empty labels list for labeled images without objects.
- Splits: `labeled`, `unlabeled`, `validation`, `test`, `hard`. Unlabeled records supply no supervision labels. Keep synchronized observations and acquisition groups together.
- `calibration_ids`: an explicit list of validation IDs. Give every participating client a validation pool.
- `views`: explicit ID lists for `participating`, `unseen`, `new-client`, `hard`. The first three use `test` records; Hard uses `hard` records. Set `require_equal_standard_views=false` when New-client has separate IDs.

## Tomato inputs

Pair RGB, depth and NIR by observation ID. RGB is uint8. Depth is uint16 millimetres: zero, nonfinite or nonpositive pixels are invalid. Loading preserves validity and divides float32 depth by `model.depth_scale=65535.0`, clipping to `[0,1]`.

Rendered depth is float32 axial camera-Z in metres. Set `renderer.scene_meters_per_unit=1.0` and `renderer.depth_unit_conversion=1000.0`; convert once to millimetres before the same normalization. Keep rendered values floating point.

NIR must be uint8 or a float `.npy` array in `[0,1]`. Normalize 16-bit NIR with the fixed training preprocessing before loading. Preview colors and percentile ranges are display settings.

## DroneVehicle inputs

Store matched RGB and single-channel IR files. The manifest field `nir` carries IR; RGB feeds the detector and IR supplies auxiliary cues. IR accepts uint8 or float `.npy` values in `[0,1]`.

For JPEGs with three exactly equal IR channels, preserve one channel as uint8 PNG. Convert polygon annotations to minimum enclosing horizontal boxes in RGB coordinates. Preserve the border, or apply registration/cropping and the same coordinate transform to every modality and annotation.

Use [prepare_manifest.py](../scripts/prepare_manifest.py) with `--input-format dronevehicle`, `--image-root`, `--ir-root`, `--class-map`, `--client-id` and `--split`. Supply `--class-aliases` for spelling variants and choose `--difficult-policy include` or `drop` consistently. Combine each client's split records with the calibration and view IDs above.

## Included input examples

The [Tomato manifest](../examples/tomato_trimodal/manifest.json) identifies the weak-light folder's `1.png` from Zhang et al.'s dataset, with RGB/depth/NIR hashes and display mappings. Its class order is in [class_names.json](../examples/tomato_trimodal/class_names.json).

The DroneVehicle example is observation `00001` from the [McCheng mirror](https://huggingface.co/datasets/McCheng/DroneVehicle). [source.json](../examples/dronevehicle/source.json) records its source and file hashes; [data_manifest.json](../examples/dronevehicle/data_manifest.json) provides the inputs and labels.
