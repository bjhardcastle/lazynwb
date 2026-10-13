# NeuroDataBench performance regression checks

The `NeuroDataBench performance regression` GitHub Actions workflow compares a
lazynwb checkout with a baseline checkout using the pinned implementation and
benchmark runner from [NeuroDataBench](https://github.com/bjhardcastle/neurodatabench).
Both checkouts run in fresh `uv` environments, so the package under test comes
from the selected commit rather than from PyPI.

Run it from the Actions tab with:

- `lazynwb_ref`: the commit, branch, or tag to evaluate;
- `baseline_ref`: the checkout used for comparison;
- `benchmark`: normally `dynamic_routing_nwb_hdf5_v0`, or
  `dynamic_routing_nwb_zarr_v0`;
- `backend`, `repeats`, and `max_regression_percent` for the benchmark policy.

Each repeat runs baseline and candidate in alternating order. Local lazynwb
caches are cold for every run, and the gate compares median total, setup, and
answer-submission time. A run is also required to complete successfully with
correct benchmark answers. Raw timing, validation, memory, network, logs, and
the generated summary are uploaded as the workflow artifact.

A timing regression must exceed both the configured percentage (15% by default)
and `--min-regression-seconds` (0.25 seconds by default). This prevents tiny
setup-phase changes from dominating the gate. Set the latter to zero for the
original percentage-only policy; every raw metric remains in the artifacts.

The workflow is reusable with `workflow_call`, so another workflow can evaluate
a specific commit without duplicating the checkout and artifact logic.

For a local comparison with identical dependencies, use an existing Python
environment containing NeuroDataBench and lazynwb's dependencies:

```powershell
python scripts/run_neurodatabench_regression.py `
  --neurodatabench-dir ../neurodatabench `
  --baseline-dir .scratch/perf-dev3 --candidate-dir . `
  --python-executable ../neurodatabench/.venv/Scripts/python.exe `
  --benchmark dynamic_routing_nwb_hdf5_v0 --repeats 3 `
  --output-dir .scratch/perf-results/local-hdf5
```

Repeat with `dynamic_routing_nwb_zarr_v0` and a new output directory. The
runner supports both the old `lazynwb_template.py` and the current
`implementations/lazynwb/dynamic_routing.py`. In shared-environment mode it
prints the imported lazynwb source path; installed distribution version
metadata can still describe the environment's editable install, so use that
source path and checkout revision to identify the tested code.

Run benchmarks sequentially, without other network benchmarks or test suites
competing for resources. Use a fresh output directory for each comparison.
Inspect raw seconds and per-question timings as well as the gate decision.

Measure `get_metadata_df` separately using the same source list and interpreter:

```powershell
../neurodatabench/.venv/Scripts/python.exe benchmarks/metadata_benchmark.py `
  --checkout . `
  --benchmark ../neurodatabench/src/neurodatabench/benchmarks/dynamic_routing_nwb_hdf5_v0.json
```

Run again with the baseline checkout and the Zarr benchmark. The output records
cold/repeated timings, actual file-open counts, row counts, and a hash of the
metadata values, so a speedup cannot silently change the returned metadata.
