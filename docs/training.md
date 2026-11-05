# Training and evaluation

Run commands from the repository root, with a new output directory for each run.

## Environment

CPU installation and runtime check:

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python scripts/check_environment.py --device cpu --profile cpu --smoke --output runs/environment_cpu.json
```

For the RTX 5080 package pair:

```bash
uv sync --locked --extra train
uv pip install -r requirements/rtx5080.txt
uv run --no-sync python scripts/check_environment.py --device cuda:0 --profile rtx5080 --smoke --output runs/environment_gpu.json
```

After installing device-specific wheels, replace the CPU command prefix below with `uv run --no-sync`. On Jetson, match the packages and vendor libraries to the installed JetPack runtime; see [the package list](../requirements/jetson-jp72.txt).

## Data manifest

The [data guide](data.md) lists dataset sources, class mappings and RGB/IR import commands.

The manifest contains `records`, a `class_names` list, `calibration_ids` and evaluation `views` mapping view names to ID lists. Each record specifies `id`, `client_id`, `split`, `rgb`, optional `depth`/`nir`, and `labels`, with paths relative to the manifest. NIR accepts uint8 images or floating-point `.npy` arrays in `[0,1]`. Each label contains a zero-based `class_id` and normalized `[x_min,y_min,x_max,y_max]` box.

Splits are `labeled`, `unlabeled`, `validation`, `test` and `hard`. Unlabeled annotations are excluded from supervision. Calibration IDs belong to validation. Keep synchronized modalities and repeated observations in the same acquisition group.

`class_names` defines the training class order saved with the checkpoint. Tomato rendering uses it for ripe and non-ripe classes. DroneVehicle uses minimum enclosing horizontal boxes; registered IR occupies the manifest's `nir` field and supplies auxiliary cues while RGB feeds the detector.

Set explicit identity lists for `participating`, `unseen`, `new-client` and `hard`. Under the manuscript protocol, Participating and Unseen share test identities, while New-client uses a separate set. Leave `require_equal_standard_views=false`; its true setting requires all present standard views to share identities.

```bash
uv run --locked --extra cpu python -m tefed validate data/tomato/manifest.json --check-files
```

Use [prepare_manifest.py](../scripts/prepare_manifest.py) for JSON/JSONL/CSV, COCO or DroneVehicle conversion. Supply the image-level assignments for each class-skew setting.

## Configuration

Copy [Tomato](../configs/tomato.toml) or [DroneVehicle](../configs/dronevehicle.toml) to `local.toml` and set:

| Field | Input |
| --- | --- |
| `experiment.manifest`, `experiment.calibration` | Data manifest and calibration JSON |
| `experiment.device` | CPU or CUDA device |
| `model.initial_checkpoint` | Shared pretrained detector |
| `model.depth_scale` | Fixed input divisor: `65535.0` for Tomato millimetre depth |
| `renderer.blender`, `renderer.scene` | Blender executable and scene |
| `renderer.scene_meters_per_unit` | Metres per scene coordinate unit: `1.0` for Tomato |
| `renderer.depth_unit_conversion` | Exported metres to dataset millimetres: `1000.0` for Tomato |
| `renderer.scene_class_map` | Scene roots mapped to class IDs |

Correction methods use the rendering fields. Map all five DroneVehicle classes. See the [rendering guide](rendering.md) for scene builders, axial-depth export and the asset interface.

### Baseline stages

The supplied 300-round configurations select `baselines.fedsto.schedule="round_stages"`:

| Completed-round labels | Training |
| --- | --- |
| 1–50 | Labeled supervision, all detector modules trainable |
| 51–150 | Labeled and pseudo supervision, backbone trainable, neck and heads frozen including BN statistics |
| 151 onward | All detector modules trainable, neck/head orthogonality penalty enabled |

`supervised_rounds=50` and `backbone_only_until=150` set the boundaries. The full stage continues for 500/1000-round runs. The resolved policy and trainable-stage flags are saved in each local training ledger.

The separate local-epoch protocol uses `schedule="local_epochs"` and `head_only_epochs=1`, with no `backbone_only_until`. It freezes shared modules in the first local epoch and unfreezes them thereafter. Do not combine these two schedule types.

Confidence self-training, Consistent-Teacher + FedAWA, FSSLOD and ProxyFL use `supervised_rounds=5` before unlabeled training in the supplied configurations. This labeled-only stage is separate from the optimizer's learning-rate warm-up. Consistent-Teacher + FedAWA and FSSLOD reset their EMA teachers from the broadcast detector each round, as does FedSTO.

On Tomato, conventional baselines apply the configured cross-modal contrastive weight to labeled regions only. TE-Fed, Server-MoE and Generic HC use the contrastive term over their assembled training regions. RGB-only detector inputs disable this term. Sample-weighted aggregation uses unique labeled plus unlabeled training observation counts, including during labeled-only stages.

`experiment.schedule_rounds=300` fixes the learning-rate and FSSLOD EMA schedules independently of the total run length. Increasing `experiment.rounds` to 500 or 1000 keeps the same first 300 rounds. Later rounds retain learning rate 0.001 and FSSLOD EMA decay 0.999. For a separate 30-round schedule, set `schedule_rounds=30` or omit it.

## Initialization and calibration

Set a private source-ID key of at least 32 UTF-8 bytes. Retain it for resumed runs, outside configuration files and logs.

Bash:

```bash
export TEFED_SOURCE_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

PowerShell:

```powershell
$env:TEFED_SOURCE_SECRET = python -c "import secrets; print(secrets.token_hex(32))"
```

Keep `model.initial_checkpoint=""` during initialization. This example uses five initialization rounds:

```bash
uv run --locked --extra cpu python -m tefed check --config local.toml --stage pretrain
uv run --locked --extra cpu python -m tefed pretrain --config local.toml --seed 0 --rounds 5 --output runs/tomato_init
```

Then set `model.initial_checkpoint="runs/tomato_init/final.pt"` and `experiment.calibration="data/tomato/calibration.json"`:

```bash
uv run --locked --extra cpu python -m tefed calibrate --config local.toml --checkpoint runs/tomato_init/final.pt --strict-count 15 --seed 0 --output data/tomato/calibration.json
```

Methods within a paired block share its partition, evaluation identities, initialization and calibration.

### Fixed FedBN prior for New-client inference

Before communication rounds, each client supplies only its validation inputs to the shared initial model in evaluation mode and collects per-layer activation moments. Labels are not used for this collection, and model parameters remain unchanged. The server pools the client moments to form one fixed BN prior for cold-start inference.

The fixed prior is saved with evaluation checkpoints and `resume.pt` and is restored when training resumes. FedBN New-client inference reads that prior from the checkpoint. Keep the validation identities and shared initialization fixed within a paired block; evaluation inputs do not update the prior.

## Train, resume and evaluate

```bash
uv run --locked --extra cpu python -m tefed check --config local.toml
uv run --locked --extra cpu python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0
uv run --locked --extra cpu python -m tefed evaluate --config local.toml --checkpoint runs/tefed_0/final.pt --output runs/tefed_0_eval
```

Select a strategy with `--method`, including `proxyfl`. Resume with the same configuration, inputs, key and output directory:

```bash
uv run --locked --extra cpu python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0 --resume runs/tefed_0/resume.pt
```

## Outputs

| File | Content |
| --- | --- |
| `configuration.json`, `calibration.json`, `environment.json` | Configuration, calibration and runtime/input identities |
| `rounds.jsonl` | Client training and selection records |
| `monitoring/round_*/metrics.json` | Per-round evaluation at the configured cadence |
| `verification_round_*.json`, `selected_events.jsonl` | Verification and case-selection records |
| `final.pt`, `final.pool.json`, `resume.pt` | Final checkpoint, retained cases and recovery snapshot |
| `metrics.json`, `predictions/` | Evaluation scores and per-image predictions |
| `checkpoints/round_*.pt` | Saved completed-round checkpoints |

## Docker

Build `tefed:cpu` using the root Dockerfile and mount the output directory:

```bash
mkdir -p runs
docker run --rm --mount "type=bind,source=$(pwd)/runs,target=/output" tefed:cpu preview --example examples/tomato_trimodal --output /output/tomato_preview
```

PowerShell mount syntax:

```powershell
New-Item -ItemType Directory -Path runs -Force | Out-Null
docker run --rm --mount "type=bind,source=$((Get-Location).Path)/runs,target=/output" tefed:cpu preview --example examples/tomato_trimodal --output /output/tomato_preview
```

Correction rendering requires Blender and mounted scene assets.
