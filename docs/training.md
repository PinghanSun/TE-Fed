# Training and evaluation

Run from the repository root with Python 3.12 and `uv`. Use a new output directory for each run.

## Install one environment

CPU (Torch 2.5.1 / Torchvision 0.20.1):

```bash
uv sync --locked --extra cpu
```

RTX 5080 (Torch 2.7.1 / Torchvision 0.22.1, CUDA 12.8):

```bash
uv sync --locked --extra train
uv pip install -r requirements/rtx5080.txt
```

Jetson (Linux aarch64, JetPack 7.2 runtime; Torch 2.12.0 / Torchvision 0.27.0, CUDA 13.2):

```bash
uv sync --locked --extra train
uv pip install -r requirements/jetson-jp72.txt
```

Match the Jetson packages and vendor libraries to the installed JetPack runtime. After installation, use `uv run --no-sync` for all commands so the selected device wheels are retained.

Check the selected environment; use `--device cpu --profile cpu`, or GPU profiles `rtx5080` / `jetson` with `--device cuda:0`:

```bash
uv run --no-sync python scripts/check_environment.py --device cpu --profile cpu --smoke --output runs/environment.json
```

## Prepare configuration

Copy [tomato.toml](../configs/tomato.toml) or [dronevehicle.toml](../configs/dronevehicle.toml) to `local.toml`. Prepare the [data manifest](data.md) and set:

- `experiment.manifest`: manifest path; `experiment.calibration`: calibration JSON to create below.
- `experiment.device`: `cpu` or `cuda:0`; `model.initial_checkpoint`: empty for initialization, then the saved initialization checkpoint.
- `renderer.blender` and `renderer.scene`: installed Blender executable and your source `.blend` scene for correction rendering.
- `renderer.scene_class_map`: scene-root names mapped to training class IDs when using custom scenes; DroneVehicle requires all five class IDs. Match object names in the scene to this mapping.
- For Tomato depth, keep `model.depth_scale=65535.0`, `renderer.scene_meters_per_unit=1.0` and `renderer.depth_unit_conversion=1000.0`.

The default configurations use 300 rounds, five seed IDs, two local epochs, batch size four and CUDA AMP. A `train --seed` command runs one seed; repeat it with separate output directories for the other IDs. Select methods with `--method`; list them with `uv run --no-sync python -m tefed methods`.

Set a private source-ID key of at least 32 UTF-8 bytes and retain it for recovery. Bash:

```bash
export TEFED_SOURCE_SECRET=$(uv run --no-sync python -c 'import secrets; print(secrets.token_hex(32))')
```

PowerShell:

```powershell
$env:TEFED_SOURCE_SECRET = uv run --no-sync python -c 'import secrets; print(secrets.token_hex(32))'
```

## Initialize, calibrate and train

Start with an empty `model.initial_checkpoint`:

```bash
uv run --no-sync python -m tefed validate data/tomato/manifest.json --check-files
uv run --no-sync python -m tefed check --config local.toml --stage pretrain
uv run --no-sync python -m tefed pretrain --config local.toml --seed 0 --rounds 5 --output runs/init
```

Set `model.initial_checkpoint` to `runs/init/final.pt` and `experiment.calibration` to `data/tomato/calibration.json`. Calibration defaults to 15 validation IDs; use `--strict-count N` to match your declared calibration pool.

```bash
uv run --no-sync python -m tefed calibrate --config local.toml --checkpoint runs/init/final.pt --seed 0 --output data/tomato/calibration.json
uv run --no-sync python -m tefed check --config local.toml
uv run --no-sync python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0
uv run --no-sync python -m tefed evaluate --config local.toml --checkpoint runs/tefed_0/final.pt --output runs/tefed_0_eval
```

For DroneVehicle, use its manifest and calibration paths. Blender and scene assets must be available when checking or running correction methods.

Recover an interrupted run with the same configuration, inputs, seed, key and output directory:

```bash
uv run --no-sync python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0 --resume runs/tefed_0/resume.pt
```

## Outputs

| Files | Contents |
| --- | --- |
| `configuration.json`, `calibration.json`, `environment.json` | Settings and input/runtime identities |
| `rounds.jsonl`, `selected_events.jsonl`, `verification_round_*.json` | Training, correction and selection records |
| `final.pt`, `final.pool.json`, `resume.pt`, `checkpoints/round_*.pt` | Models, correction pool and recovery state |
| `metrics.json`, `predictions/`, `monitoring/` | Evaluation scores and per-image predictions |
