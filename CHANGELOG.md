# Changes

## Unreleased — 2.6.5.dev0

- Align the displayed manuscript title and repository links with the current TE-Fed manuscript; retain v2.6.4 wheel/scene links as historical release assets.
- Use v2 A/E controls with effective correction `E=B AND C`. At A=0, qualified pseudo-targets and background have unit weights per active head. The B0C0, B0C1 and B1C0 cells all disable correction; Generic HC remains a separate method.
- Omit empty possible-class sets from ProxyFL ICPL anchors and negatives, and count omitted proposals separately.
- Exclude rendered variants with zero fixed fidelity-mask weight before expert assessment, coverage and verification; keep these exclusions outside the verification denominator.
- Form a fixed FedBN cold-start prior by pooling initial-model activation moments from client validation inputs; preserve it in checkpoints and resume state for New-client inference.

- Export float32 axial camera-Z in metres, with scene scale and a depth validity mask.
- Share one-time rendered-depth conversion between verification and local training, followed by fixed Tomato millimetre-depth normalization.
- Add FedSTO round stages to the 300-round configurations, including frozen neck/head BN statistics and final-stage neck/head regularization. Retain the separate local-epoch protocol.
- Apply five-round supervised warm-up and round-reset teachers to the configured semi-supervised baselines. Conventional Tomato baselines use cross-modal contrast on labeled regions only.
- Use unique labeled plus unlabeled observation counts for sample-weighted aggregation, and hold learning-rate and FSSLOD EMA schedules fixed after round 300 during continuation.
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
