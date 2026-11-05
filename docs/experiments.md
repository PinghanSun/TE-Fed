# Experiments

The primary Tomato and DroneVehicle protocol uses 300 rounds and five paired blocks per setting. Each reported population and control requires its corresponding manifest and specific protocol configuration.

## Paired blocks

Run IDs are `0–4`. Methods within each block share its data partition and evaluation identities. Record one entry per block in `blocks.json`:

```json
{"schema_version":1,"blocks":[{"block_id":"0","seed":0,"partition_sha256":"<manifest SHA256>","evaluation_ids_sha256":"<evaluation IDs SHA256>"}]}
```

Use 64-character SHA256 values. The evaluation digest is `tefed.experiments.canonical_hash(sorted(ids))`, using the view IDs in `predictions/evaluation_manifest.json`. Analyze each setting, view, metric and checkpoint separately. `paired` supports different partitions across blocks; `summary` summarizes seeds sharing one partition.

## Comparisons and ablations

```bash
uv run --locked --extra cpu python scripts/run_suite.py --config local.toml --plan comparisons --output runs/comparisons --execute
```

The comparison plan runs the 12 registered methods, including [ProxyFL](proxyfl.md). Omit `--execute` to validate and save the plan; use `--resume` to continue it. Other plans cover `mechanisms`, `budgets`, `components` and frozen-pool `replay`.

The default qualification/correction comparison uses four effective A/E conditions (`tefed-conditions-v2`, backed by `tefed-factorial-v2`):

```bash
uv run --locked --extra cpu python scripts/run_suite.py --config local.toml --plan conditions --output runs/conditions --execute
```

| Condition | Qualification A | Correction E | Implementation bits |
| --- | --- | --- | --- |
| A0E0 | Fixed posterior threshold 0.75, unit weights | Off | A=0, B=C=0 |
| A1E0 | Reliability qualification | Off | A=1, B=C=0 |
| A0E1 | Fixed posterior threshold 0.75, unit weights | On | A=0, B=C=1 |
| A1E1 | Reliability qualification | On | A=1, B=C=1 |

Effective correction is `E=B AND C`. The four-condition plan sets `B=C=E`: E=1 enables the complete source-evidence rendering, stored-route verification, pool insertion, descriptor/case router probes and next-round state-matched reuse path; E=0 disables that path and uses Server-MoE.

At A=0, validity checks, routing, measured reliability and correction eligibility remain active. Qualification uses posterior confidence at least 0.75. Every qualified pseudo-target has weight one in each head of the image's existing active-head union, and unmatched background has weight one. A=1 uses the state-dependent reliability gate and routed reliability weights.

The eight-cell `factorial` plan retains the A/B/C encoding for protocol checks. At fixed A, B0C0, B0C1 and B1C0 all have E=0 and the same correction-disabled operations. They are structural duplicate controls. Generic HC is a separate registered method with its own coarse-prior correction procedure. Analyze the four A/E cells by default; retain the recorded version and operation-definition hashes when comparing runs.

```bash
uv run --locked --extra cpu python scripts/analyze_runs.py conditions --records runs/conditions/conditions_hard_mAP50.jsonl --block-manifest blocks.json --blocks 0 1 2 3 4 --scale 100 --out runs/condition_effects.json
```

This computes conditional effects and `(A1E1−A0E1)−(A1E0−A0E0)` within blocks, followed by mean and sample SD across blocks.

## Paired method results

Prepare one JSONL row per method and block with `block_id`, `seed`, `method`, numeric `value`, `partition_sha256` and `evaluation_ids_sha256`. Include `view`, `metric`, `checkpoint_round` and `setting_id` to identify the analysis.

```bash
uv run --locked --extra cpu python scripts/analyze_runs.py paired --records scores.jsonl --block-manifest blocks.json --methods tefed proxyfl server_moe --reference tefed --blocks 0 1 2 3 4 --scale 100 --out runs/paired_results.json
```

Method scores retain their input scale. Differences are reference minus comparator, computed within blocks before aggregation. `--scale 100` converts differences from 0–1 AP to percentage points. Each declared method and block must occur exactly once.

## Checkpoints and plots

The manuscript's primary AP uses fixed panels containing 512 annotated objects per dataset, block and evaluation view. Complete-test-pool AP evaluates the same checkpoint on a separate declared population. Supply the manifest and configuration for the intended population and retain its identity hashes. Historical NoCorrection controls also require their specific recorded configuration; the plan names alone do not recover those settings or the paper's scores.

Set completed-round counts in the configuration:

```toml
[experiment]
rounds = 1000
schedule_rounds = 300
save_rounds = [300, 500, 1000]
```

The primary 300-round configuration uses `save_rounds=[300]`. Checkpoints are written after aggregation, router and client-state updates, before the round commit. Evaluation checkpoints retain the model, router and client state summaries, plus Local-only models or FedBN statistics when applicable. `resume.pt` also retains training and aggregation state. Record the training schedule with each checkpoint.

`evaluation.monitor_every=1` saves per-round AP in `monitoring/round_*/metrics.json`. `report_runs.py plots` reads training loss from `rounds.jsonl` and final scores from `metrics.json`; it produces loss curves, metric comparisons and optional factorial-effect plots. Loss lines connect all recorded points, with sparse markers. `plot_values.csv` contains the plotted values. `report_runs.py qualitative` draws image-level detection comparisons.

The manuscript's checkpoint-based Fig. 8 includes calculated intermediate values between saved evaluations. Reproducing that figure requires its saved checkpoint inputs and plotting protocol in addition to the run-output plotting command above.

## Additional settings

Class-skew experiments use target label fractions of 10% and 25% with Dirichlet concentrations `0.1, 0.5, 1, 10`. Use the corresponding image-level manifests and configurations. The `components` plan defines loss/input ablations. Modality, robustness, client-scale and parameter-sensitivity experiments use separate manifests and TOML configurations.

## Timing and resources

| Analysis | Inputs and scope |
| --- | --- |
| Native inference time | `profile_inference.py`: synchronized image loading, preprocessing and routed inference; defaults are 20 warm-ups and 200 measured requests |
| Resource accounting | Recorded workload totals and round boundaries |
| Device energy | `record_power.py` plus `analyze_runs.py energy`: power samples and round boundaries |

`analyze_runs.py latency` computes node-trace quantiles and the worst-client summary. Record the timing scope with each result.

For block-level request traces, use a separate entry point:

```bash
uv run --locked python scripts/summarize_request_traces.py --csv requests.csv --out runs/request_p95.json
```

The CSV requires `scope`, `block_id`, `request_id`, `phase`, `status` and `latency_ms`. Set `phase` to `warmup` or `timed`, and `status` to `success`, `failure` or `timeout`. Successful timed requests require a finite nonnegative latency in milliseconds. Give distinct scopes, such as `forward` and `workflow`, separate names.

Within each block and scope, P95 is the nearest-rank value at `ceil(0.95*n)` among successful timed requests. Warm-ups are excluded and failures/timeouts are counted separately. The output reports attempt counts and summarizes block P95s using mean and sample SD. A block with no successful timed requests has an undefined P95 and remains listed in the output.

For the 50-warm-up/1000-attempt protocol, check the reported counts for every block. The summarizer accepts other trace lengths and does not infer missing requests. Keep this analysis separate from the existing 200-request node-registry quantile calculation. This command summarizes recorded traces. It does not collect timing measurements.
