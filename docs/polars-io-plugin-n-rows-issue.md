# Polars predicate plus head boundary for remote scans

## Summary

Polars currently pushes predicates into Python IO plugin scans, and it pushes a
plain `head(n)` into the plugin as `n_rows=n`. The boundary is the combination:

```python
scan.filter(predicate).select(expensive_columns).head(n)
```

For that shape, Polars passes the predicate to the Python IO plugin but leaves
`n_rows=None`. The logical plan keeps the slice above the Python scan, so the
plugin has to produce all rows that match the predicate before Polars truncates
the result to `n` rows.

For lazynwb this matters because the scan path reads cheap predicate columns
first, filters them, and then fetches projected columns by source row index. If
`n_rows` is missing, lazynwb cannot know that only the first `n` matching rows
are needed, so expensive projected NWB/HDF5 columns may be fetched for every
matching row.

## Current Polars reproduction

Run:

```shell
uv run python benchmarks/polars_io_plugin_n_rows_repro.py
```

Observed locally with Python `3.11.14` and Polars `1.38.1`:

```text
=== head_only ===
PYTHON SCAN []
PROJECT */2 COLUMNS
SLICE: Positive { offset: 0, len: 3 }
io_source_call=with_columns=None predicate_received=False n_rows=3 batch_size=None batches_yielded=[0]

=== filter_head ===
SLICE[offset: 0, len: 3]
  PYTHON SCAN []
  PROJECT */2 COLUMNS
  SELECTION: [(col("x")) >= (0)]
io_source_call=with_columns=None predicate_received=True n_rows=None batch_size=100000 batches_yielded=[0, 1, 2, 3, 4]

=== filter_select_head ===
SLICE[offset: 0, len: 3]
  PYTHON SCAN []
  PROJECT 1/2 COLUMNS
  SELECTION: [(col("x")) >= (0)]
io_source_call=with_columns=['x'] predicate_received=True n_rows=None batch_size=100000 batches_yielded=[0, 1, 2, 3, 4]
```

The first case proves the plugin can receive `n_rows` for a plain head. The
predicate cases prove the same slice remains outside the Python scan once a
predicate is pushed down.

This matches the current Polars optimizer guard on `main`, which still skips
Python-scan slice pushdown when `options.predicate` is not
`PythonPredicate::None`:

<https://github.com/pola-rs/polars/blob/main/crates/polars-plan/src/plans/optimizer/slice_pushdown_lp.rs#L348-L353>

The related upstream issue is still open:

<https://github.com/pola-rs/polars/issues/23026>

## Remote NWB/HDF5 impact

Natural query measured against a public remote HDF5 NWB file:

```python
import polars as pl

import lazynwb.lazyframe as lazyframe

source = "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/620263_2022-07-26.nwb"

query = (
    lazyframe.scan_nwb(source, table_path="units", disable_progress=True)
    .filter(pl.col("default_qc"))
    .select("unit_id", "spike_times")
    .head(5)
)
```

The plan shows the slice above the Python scan:

```text
simple pi 2/2 ["unit_id", "spike_times"]
  SLICE[offset: 0, len: 5]
    PYTHON SCAN []
    PROJECT 3/67 COLUMNS
    SELECTION: col("default_qc")
```

Observed lazynwb debug logs for one cold-cache run:

```text
Initial 'units' df filtered with predicate: 1865 rows reduced to 473
Fetching additional columns from 'units': ['spike_times', 'unit_id']
direct HDF5 indexed materialization for 'spike_times': rows=473 spans=330 requested_elements=6371571 spanned_elements=6371571 full_elements=16887206 range_requests=282 fetched_bytes=52235556
Created 'units' DataFrame (473 rows) from 1 NWB files in 12.36 s
```

The final result had only 5 rows:

```text
metric scan_s=1.411 collect_s=12.497 total_s=13.908 rows=5 columns=2 size_bytes=1010830
```

So `filter(...).select("unit_id", "spike_times").head(5)` still fetched
`spike_times` for all 473 matching rows. The remote indexed HDF5 materializer
made that reasonably efficient, but the cost was still 282 range requests and
about 52 MB of remote data for a 5-row result. This is the exact boundary users
can hit with expensive ragged, indexed, string, array, or fallback-projected NWB
columns.

For comparison only, the existing explicit benchmark workaround path:

```shell
uv run python benchmarks/filtered_units_cold_cache_benchmark.py \
  --row-mode filtered \
  --filter-column default_qc \
  --operator eq \
  --filter-value true \
  --select unit_id \
  --select spike_times \
  --limit 5 \
  --repeats 1
```

reported:

```text
summary repeats=1 total_median=2.479s total_best=2.479s scan_median=1.192s collect_median=1.287s
```

That benchmark path explicitly collects the predicate/internal rows, applies the
limit, and then fetches projected columns for the limited row set. It is not
normal `scan_nwb` behavior.

## What lazynwb can improve locally

- Keep documenting this optimizer boundary and keep the standalone Polars repro
  in `benchmarks/polars_io_plugin_n_rows_repro.py`.
- Continue improving row-indexed HDF5 materialization, range coalescing, and
  fallback column reads. Those improvements reduce the cost once lazynwb knows
  which rows to fetch.
- Add benchmark coverage for expensive projected columns so future HDF5
  materialization changes do not regress this boundary.
- Consider an explicit opt-in helper or benchmark workflow that performs a
  two-pass limited filtered read: fetch predicate and internal row identity
  columns, apply `head(n)`, then fetch expensive projections for those row
  identities.

## What is blocked outside lazynwb

- Transparent optimization of `scan_nwb(...).filter(...).head(n)` depends on
  Polars passing the post-predicate limit to Python IO plugins, either as
  `n_rows` after an accepted predicate or as a richer slice contract.
- lazynwb cannot infer the outer `head(n)` from inside the current Python IO
  plugin callback when Polars sends `n_rows=None`.
- Full offset/tail slice pushdown is also a Polars IO-plugin contract issue,
  tracked upstream by pola-rs/polars#23026.

## Follow-up scope for a workaround

Do not hide a workaround inside `scan_nwb` unless Polars exposes the slice to the
plugin. A local workaround changes execution shape and has edge cases around
ordering, unsupported predicates, joins, and user expectations for lazy plans.

If lazynwb wants the workaround, create a separate follow-up issue with an
explicit API/benchmark scope, for example:

> Add an opt-in limited filtered remote scan helper for expensive projections.

That follow-up should define the supported query shape, how row ordering is
preserved, which predicate expressions are accepted, whether private row
identity columns are exposed internally, and how it reports that it is not
general Polars lazy optimization.
