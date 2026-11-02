# TE-Fed

[![Source checks](https://github.com/autumnyfot1898-bot/TE-Fed/actions/workflows/tests.yml/badge.svg)](https://github.com/autumnyfot1898-bot/TE-Fed/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**TE-Fed: Correction with Source Evidence and Reuse by State Matching for Federated Semi-Supervised Multisensor Object Detection.**

TE-Fed verifies rendered correction cases against their recorded source expert routes and selects cases for each client's current state.

<p align="center"><img src="assets/overview.png" width="100%" alt="TE-Fed overview"></p>

## Install

```bash
git clone https://github.com/autumnyfot1898-bot/TE-Fed.git
cd TE-Fed
uv sync --locked --extra cpu
uv run --locked --extra cpu python -m tefed methods
```

The registry contains 12 methods, including [ProxyFL](docs/proxyfl.md). For GPU installation, follow [training and evaluation](docs/training.md).

## Example inputs

| Dataset | Inputs | Configuration | Example |
| --- | --- | --- | --- |
| Tomato | RGB, depth, NIR | [tomato.toml](configs/tomato.toml) | [Paired inputs](examples/tomato_trimodal/README.md) |
| DroneVehicle | RGB, IR | [dronevehicle.toml](configs/dronevehicle.toml) | [Paired inputs](examples/dronevehicle/README.md) |

![RGB, depth and NIR](examples/tomato_trimodal/images/rgb_depth_nir.png)

Generate the [Tomato input preview](examples/tomato_trimodal/README.md):

```bash
uv run --locked python -m tefed preview --example examples/tomato_trimodal --output runs/tomato_preview
```

Full datasets and trained checkpoints are separate from this source release. Download the source archive, Python wheel and editable vehicle scene from [release 2.6.4](https://github.com/autumnyfot1898-bot/TE-Fed/releases/tag/v2.6.4). The [rendering guide](docs/rendering.md) covers procedural vehicle scenes; the [IR guide](docs/infrared.md) covers material-based NIR/LWIR rendering.

## Train and evaluate

The main experiments used 300 rounds, five run IDs, two local epochs and batch size four. Start with the [Tomato](configs/tomato.toml) or [DroneVehicle](configs/dronevehicle.toml) configuration.

Follow [data preparation](docs/data.md) and the [training guide](docs/training.md) to create the manifest, shared initialization and calibration, then run:

```bash
uv run --locked --extra cpu python -m tefed check --config local.toml
uv run --locked --extra cpu python -m tefed train --config local.toml --method tefed --seed 0 --output runs/tefed_0
uv run --locked --extra cpu python -m tefed evaluate --config local.toml --checkpoint runs/tefed_0/final.pt --output runs/tefed_0_eval
```

See [single-sample inference](docs/single_sample_inference.md) for detection, [experimental protocols](docs/experiments.md) for comparisons and statistics, and the [paper-to-code map](docs/paper_to_code.md) for module locations.

## Checks and Docker

```bash
uv lock --check
uv run --locked --extra cpu python -m tefed smoke --output runs/software_smoke --device cpu
uv run --locked --extra cpu python -m unittest discover -s tests -t . -v
uv build --wheel --out-dir dist
uv run --locked python scripts/check_wheel.py --dist dist
docker build -t tefed:cpu .
docker run --rm tefed:cpu --help
```

[GitHub Actions](.github/workflows/tests.yml) runs source and packaging checks. Full CPU and Docker checks are available through `workflow_dispatch`.

## Citation and license

Use [CITATION.cff](CITATION.cff) for software citation. Code uses [AGPL-3.0](LICENSE); original vehicle scenes and rendered previews use [CC BY 4.0](ASSET_LICENSE.md). See [CHANGELOG.md](CHANGELOG.md) and [third-party components](THIRD_PARTY.md).
