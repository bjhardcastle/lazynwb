# Polars dynamic TopK predicates and Python IO plugins

Investigated on 2026-05-13 with Python `3.11.14`, lazynwb's
`polars.io.plugins.register_io_source` scan path, and Polars `1.38.1`,
`1.39.0`, `1.39.3`, and `1.40.1`.

## Summary

Polars `1.39+` can optimize `sort(...).head(n)` / TopK plans by injecting a
dynamic predicate into the scan. In native Polars readers that can be a useful
early-pruning optimization. For lazynwb's Python IO plugin, the pushed predicate
currently arrives as an ordinary `pl.Expr` containing an optimizer-only dynamic
predicate expression:

```text
col("firing_rate").dynamic_predicate()
```

Applying that expression inside the plugin with `df.filter(predicate)` panics in
Polars `1.39+`:

```text
pyo3_runtime.PanicException: internal error: entered unreachable code
```

The crash does not require NWB, HDF5, or S3. A tiny Python IO plugin registered
with `polars.io.plugins.register_io_source` reproduces it.

## User query shape

The query shape that exposed this in lazynwb was:

```python
(
    lazynwb.scan_nwb(sources, "units", disable_progress=True)
    .filter(
        pl.col("structure") == "VISp",
        pl.col("firing_rate").is_not_null(),
    )
    .sort("firing_rate", descending=True)
    .head(1)
    .select("spike_times")
    .collect()
)
```

On Polars `1.38.1`, the optimized plan pushed the user predicate into the
Python scan and left the sort/head outside:

```text
SORT BY [slice: (0, 1), descending: [true]] [col("firing_rate")]
  PYTHON SCAN []
  PROJECT 3/3 COLUMNS
  SELECTION: [(col("firing_rate").is_not_null()) & ([(col("structure")) == ("VISp")])]
```

On Polars `1.40.1`, the optimized plan included a dynamic predicate in the
Python scan:

```text
SORT BY [slice: (0, 1, dynamic_pred: ...), descending: [true]] [col("firing_rate")]
  PYTHON SCAN []
  PROJECT 3/67 COLUMNS
  SELECTION: [([(col("firing_rate").is_not_null()) & (col("firing_rate").dynamic_predicate())]) & ([(col("structure")) == ("VISp")])]
```

The same behavior appeared in direct `top_k`, `sort(...).limit(...)`, and
`sort(...).slice(0, n)` plans.

## Why this could matter for lazynwb

If Polars eventually exposes this optimization to Python IO plugins in a usable
form, lazynwb could use it to reduce remote reads for sorted top-N queries over
expensive projected NWB columns.

The attractive future case is:

```python
scan_nwb(...).filter(cheap_predicate).sort(metric).head(n).select(expensive_column)
```

If the plugin knew the current TopK threshold or received a supported dynamic
predicate contract, it could avoid fetching expensive ragged/indexed columns for
rows that cannot enter the final top-N result. This is adjacent to, but distinct
from, the post-predicate `head(n)` boundary documented in
`docs/polars-io-plugin-n-rows-issue.md`.

## Why lazynwb should not raise the Polars minimum for this yet

The current Python IO plugin API documents `predicate` as a Polars expression
that the reader must apply. The dynamic TopK predicate is not currently
actionable from lazynwb because the callback receives only the expression, not
the dynamic TopK threshold/state needed to evaluate or exploit it safely.

Observed behavior:

- `predicate.meta.root_names()` sees the sorted column.
- `predicate.meta.serialize(format="json")` exposes the dynamic predicate as a
  `Display` expression with a `fmt_str` starting with `dynamic_pred`.
- `predicate.meta.tree_format()` and `df.filter(predicate)` can panic.
- Disabling `slice_pushdown` avoids the dynamic predicate but gives up the
  optimizer path for the whole collect call.

So increasing the minimum Polars version would opt users into an unstable
optimizer interaction without giving lazynwb enough information to implement
the optimization.

## Current lazynwb mitigation

lazynwb now strips only standalone Polars dynamic predicate conjuncts from
pushed scan predicates before applying the predicate in the Python callback.
Ordinary user predicates remain pushed down.

The mitigation lives in `src/lazynwb/lazyframe.py`:

```python
predicate, dynamic_predicate_count = _remove_polars_dynamic_predicates(predicate)
```

The regression test is:

```shell
uv run pytest tests/test_lazyframe.py::test_scan_nwb_sort_head_ignores_polars_dynamic_topk_predicate
uv run --with 'polars[pandas]==1.40.1' pytest tests/test_lazyframe.py::test_scan_nwb_sort_head_ignores_polars_dynamic_topk_predicate
```

The original remote scratch repro also succeeds under forced Polars `1.40.1`
after the mitigation:

```shell
uv run --with 'polars[pandas]==1.40.1' python .scratch/scan_test.py
```

## Revisit triggers

Revisit this if Polars adds one of the following:

- A Python IO plugin contract for dynamic TopK predicates.
- A callback argument that carries TopK threshold/state separately from
  ordinary predicates.
- A guarantee that optimizer-internal dynamic predicates will not be pushed into
  Python IO plugins.
- A richer scan slice contract for `filter(...).sort(...).head(n)` or
  `filter(...).select(...).head(n)`.

When revisiting, test at least:

- `filter(...).sort(metric).head(n).select(expensive_column)`
- `sort(metric).head(n).select(expensive_column)`
- `top_k(n, by=metric).select(expensive_column)`
- `sort(metric).slice(0, n).select(expensive_column)`
- the same shapes with and without user predicates
- remote HDF5 `units/spike_times` or another expensive indexed column

## Upstream references

- Polars IO plugin docs:
  <https://docs.pola.rs/user-guide/plugins/io_plugins/>
- `register_io_source` API docs:
  <https://docs.pola.rs/api/python/stable/reference/api/polars.io.plugins.register_io_source.html>
- Polars `1.39.0` release notes mention dynamic predicates for TopK:
  <https://github.com/pola-rs/polars/releases/tag/py-1.39.0>
- Dynamic TopK implementation PR:
  <https://github.com/pola-rs/polars/pull/26495>
- Related optimizer barrier fix for sort with baked-in slice:
  <https://github.com/pola-rs/polars/pull/26804>
- Related lazynwb note on post-predicate `head(n)`/`n_rows` pushdown:
  `docs/polars-io-plugin-n-rows-issue.md`
