# Paper-to-code map

| Operation | Module |
| --- | --- |
| Model, losses and training | [model.py](../src/tefed/model.py), [losses.py](../src/tefed/losses.py), [engine.py](../src/tefed/engine.py) |
| Cues, reliability and routing | [cues.py](../src/tefed/cues.py), [core.py](../src/tefed/core.py) |
| Source verification and rendering | [server.py](../src/tefed/server.py), [rendering.py](../src/tefed/rendering.py) |
| Axial depth, unit conversion and fixed normalization | [depth.py](../src/tefed/depth.py), [Blender exporter](../src/tefed/blender_scripts/blender_render.py) |
| Client state and reuse | [state.py](../src/tefed/state.py), [core.py](../src/tefed/core.py), [client.py](../src/tefed/client.py) |
| ProxyFL | [proxyfl.py](../src/tefed/proxyfl.py), [implementation guide](proxyfl.md) |
| Calibration and evaluation | [calibration.py](../src/tefed/calibration.py), [evaluation_manifest.py](../src/tefed/evaluation_manifest.py), [metrics.py](../src/tefed/metrics.py) |
| Experiments, statistics and recovery | [experiments.py](../src/tefed/experiments.py), [analysis.py](../src/tefed/analysis.py), [resume.py](../src/tefed/resume.py) |
| Request-level P95 by paired block | [measurement.py](../src/tefed/measurement.py), [summarize_request_traces.py](../scripts/summarize_request_traces.py) |
| Input preview and single-sample inference | [preview.py](../src/tefed/preview.py), [inference.py](../src/tefed/inference.py) |
| Editable vehicle assets and environment | [create_vehicle_scene.py](../scripts/create_vehicle_scene.py), [refine_vehicle_assets.py](../scripts/refine_vehicle_assets.py), [refine_vehicle_environment.py](../scripts/refine_vehicle_environment.py) |
| Material-based infrared rendering | [infrared_model.py](../src/tefed/blender_scripts/infrared_model.py), [render_scene_ir.py](../scripts/render_scene_ir.py), [run_ir_example.py](../scripts/run_ir_example.py) |

## Method coverage

| Method | Registry key |
| --- | --- |
| Local-only | `local` |
| FedAvg | `fedavg` |
| FedProx | `fedprox` |
| FedBN | `fedbn` |
| Confidence self-training | `confidence` |
| Consistent-Teacher + FedAWA | `consistent_teacher_fedawa` |
| FedSTO | `fedsto` |
| FSSLOD | `fsslod` |
| ProxyFL | `proxyfl` |
| Server-MoE | `server_moe` |
| Generic HC | `generic_hc` |
| TE-Fed | `tefed` |

`python -m tefed methods` lists strategies. Method configuration tables define the training settings. The shipped configurations use FedSTO round stages 1–50, 51–150 and 151 onward. The separate local-epoch protocol is selected explicitly. Default cues use weak/strong reference augmentations. Temporal cues take a supplied previous observation.

See [training](training.md) for sensor preprocessing and rendering, and [experiments](experiments.md) for statistics and resource measurements.
