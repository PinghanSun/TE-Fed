# TE-Fed

[![Source checks](https://github.com/PinghanSun/TE-Fed/actions/workflows/tests.yml/badge.svg)](https://github.com/PinghanSun/TE-Fed/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**TE-Fed: From Rejected Predictions to Reusable Supervision with Source Evidence and State Matching for Federated Multimodal Object Detection.**

TE-Fed renders correction cases from eligible rejected predictions. Verification recomputes expert responses with the recorded source expert identities and normalized route weights held fixed. Retained cases become available for state-matched reuse from the next round.

<p align="center"><img src="assets/overview.png" width="100%" alt="TE-Fed overview"></p>

## Install

```bash
git clone https://github.com/PinghanSun/TE-Fed.git
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

Download the [current source archive](https://github.com/PinghanSun/TE-Fed/archive/refs/heads/main.zip). This source tree is version `2.6.5.dev0` (unreleased). The Python wheel and editable vehicle scene for version 2.6.4 remain available from [the historical release page](https://github.com/PinghanSun/TE-Fed/releases/tag/v2.6.4). The manuscript identifies its versioned reference as v2.6.3 and separately fixes additional interfaces by snapshot hashes; retain those identifiers when comparing artifacts. Dataset sources and preparation are listed in the [data guide](docs/data.md). Training uses a local checkpoint prepared through the [training guide](docs/training.md). The [rendering guide](docs/rendering.md) covers procedural vehicle scenes; the [IR guide](docs/infrared.md) covers material-based NIR/LWIR rendering.

## Train and evaluate

For the manuscript's primary protocol, use 300 communication rounds, five paired blocks, two local epochs per round and batch size four. Start with the [Tomato](configs/tomato.toml) or [DroneVehicle](configs/dronevehicle.toml) configuration. Use the corresponding manifests and specific protocol configurations for the primary AP panels, complete test pools and additional controls; see [experimental protocols](docs/experiments.md).

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
