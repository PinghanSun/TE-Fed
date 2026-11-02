# Single-sample inference

`infer` runs Tomato TE-Fed detection with frozen model, router and client state. Use the five-specialist configuration with auxiliary modalities enabled.

## Inputs

Supply a checkpoint containing the model, router, configuration and client states; the calibration JSON used for that run, with positive `T_C`; registered RGB/depth/NIR inputs; and a client ID stored in the checkpoint.

| Input | Format |
| --- | --- |
| RGB | 8-bit RGB image |
| Depth | Finite nonnegative single-channel values; normalized by `model.depth_scale` |
| NIR | uint8 image or floating-point `.npy` in `[0,1]` |

Use matching original image dimensions. Prepare higher-bit-depth NIR with the training preprocessing; the example's display mapping serves visualization.

## Run

```bash
uv run --locked --extra cpu python -m tefed infer \
  --checkpoint /path/to/final.pt --calibration /path/to/calibration.json \
  --rgb /path/to/prepared/rgb.png --depth /path/to/prepared/depth.png --nir /path/to/prepared/nir.png \
  --sample-id YOUR_SAMPLE_ID --client-id YOUR_CLIENT_ID \
  --device cpu --display-confidence 0.25 --output runs/single_sample
```

Use a new output directory. Class names come from the checkpoint. For checkpoints without names, supply `--class-names classes.json`; the fallback labels are `class_0`, `class_1`, etc. An explicit mapping must agree with any names stored in the checkpoint.

## Outputs

| File | Content |
| --- | --- |
| `predictions.json` | Class IDs, scores, routes and image coordinates for all returned candidates |
| `rgb_with_detections.png` | RGB image with detections above the display threshold |
| `provenance.json` | Input hashes, checkpoint settings, frozen state and coordinate transform |

`box` uses normalized letterbox coordinates. `box_original_normalized_xyxy` and `box_original_xyxy` give original-image normalized and pixel coordinates. The display threshold filters the image overlay; JSON retains all generated candidates.

`--round-id` sets the reference-augmentation context and defaults to `0`. The command verifies unchanged model/state/input data. For dataset AP, use `evaluate` with a held-out manifest.
