# TE-Fed

TE-Fed provides training and evaluation for federated multimodal object detection, with rendered correction cases, source-route verification and state-matched reuse.

## Compatible environments

Package combinations for Python 3.12 and Ultralytics 8.4.0:

| Platform | PyTorch / torchvision | Runtime / package list |
| --- | --- | --- |
| CPU | 2.5.1 / 0.20.1 | `cpu` installation extra |
| NVIDIA RTX 5080 | 2.7.1 / 0.22.1 | CUDA 12.8; [requirements](requirements/rtx5080.txt) |
| NVIDIA Jetson | 2.12.0 / 0.27.0 | JetPack 7.2 / CUDA 13.2; [requirements](requirements/jetson-jp72.txt) |

Correction rendering requires Blender and local scene assets. See the [training guide](docs/training.md) for device setup and configuration.

## Install

```bash
git clone https://github.com/PinghanSun/TE-Fed.git
cd TE-Fed
uv sync --locked --extra cpu
uv run --no-sync python -m tefed methods
```

For RTX 5080, install the training dependencies and CUDA package pair instead:

```bash
uv sync --locked --extra train
uv pip install -r requirements/rtx5080.txt
```

For Jetson, match the package pair to the installed JetPack runtime as described in the [training guide](docs/training.md). Use `uv run --no-sync` after installing device-specific packages.

## Train and evaluate

Follow [data preparation](docs/data.md) and the [training guide](docs/training.md) to prepare a manifest, shared initialization, calibration and source-ID key. Copy a configuration from `configs/` to `local.toml` and set the data paths, device, checkpoint and rendering assets.

```bash
uv run --no-sync python -m tefed check --config local.toml
uv run --no-sync python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0
uv run --no-sync python -m tefed evaluate --config local.toml --checkpoint runs/tefed_0/final.pt --output runs/tefed_0_eval
```

Use a new output directory for each run. Full datasets and trained checkpoints are prepared separately; dataset sources, class mappings and preprocessing are in the [data guide](docs/data.md).

## License

Code: [AGPL-3.0-only](LICENSE). Original scene assets: [CC BY 4.0](ASSET_LICENSE.md). Dataset samples and third-party components retain their respective terms; see [third-party information](THIRD_PARTY.md). Software citation metadata are in [CITATION.cff](CITATION.cff).
