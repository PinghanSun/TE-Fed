# Changes

## Unreleased

- Export float32 axial camera-Z in metres, with scene scale and a depth validity mask.
- Share one-time rendered-depth conversion between verification and local training, followed by fixed Tomato millimetre-depth normalization.
- Add FedSTO round stages to the 300-round configurations, including frozen neck/head BN statistics and final-stage neck/head regularization. Retain the separate local-epoch protocol.
- Apply five-round supervised warm-up to the configured semi-supervised baselines, and keep TE-Fed alignment out of conventional baseline objectives.
- Summarize request-level P95 by scope and paired block, with nearest-rank quantiles, attempt/failure counts and sample SD.
- Add regression tests for depth geometry, training-stage boundaries and timing summaries.

## 2.6.4

- Paired Tomato RGB/depth/NIR and DroneVehicle RGB/IR previews, with local sample extraction.
- Checkpoint class names shared by inference, calibration and rendering.
- Two-dataset data preparation and frozen single-sample inference.
- Procedural scene builders, editable vehicles and material-based NIR/LWIR rendering with configurable profiles.
- 300-round configurations with five run IDs.
- Training preflight, saved-round checkpoints and recovery.
- Four A/E conditions and paired-block analysis.
- ProxyFL detector ICPL, GPT and class-prior updates.
- Packaged Blender scripts, wheel checks and Docker checks.
- Full recorded curve points with sparse markers.

## 2.6.3

Source release before the input-preview and single-sample inference commands.
