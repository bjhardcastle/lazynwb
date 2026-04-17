---
name: lazynwb
description: How to efficiently read, filter, join, and explore NWB (Neurodata Without Borders) files with the lazynwb Python library. Use this skill whenever the user mentions NWB files, lazynwb, pynwb, DANDI, spike_times, trials/units/electrodes tables, session metadata across multiple recordings, or wants to load neuroscience data — even if they don't explicitly name lazynwb. Also use when translating pynwb code to lazynwb, exploring an unfamiliar NWB file, or scanning a DANDI dandiset.
---

# lazynwb

lazynwb is a **read-only**, cloud-friendly Python library for NWB (Neurodata
Without Borders) files. Its distinguishing feature is a Polars plugin —
`scan_nwb()` — that returns a `pl.LazyFrame` and pushes `.filter()` /
`.select()` down to the HDF5/Zarr read layer. For tables like `/units` that mix
cheap scalar columns (`presence_ratio`, `location`, …) with huge array columns
(`spike_times`, `waveform_mean`), this turns "load everything then filter" into
"load only the bytes the answer needs." Over cloud storage, that's the
difference between minutes and seconds.

## Mental model

- `lazynwb.scan_nwb(paths, table_path)` → `pl.LazyFrame` with pushdown.
  **Default choice** for almost every query.
- `lazynwb.get_df(paths, table_path)` → eager DataFrame (pandas, or polars
  with `as_polars=True`). Use when you know you want *all* rows with array columns skipped by default. Other columns can be included/excluded as needed.
- `lazynwb.get_metadata_df(paths)` → one row per file with session/subject
  fields (`session_id`, `subject_id`, `age`, `sex`, `species`, …). Cheap.
- `lazynwb.get_timeseries(path, "name", exact_path=True)` → a lightweight
  TimeSeries dataclass whose `.data` and `.timestamps` are lazy h5py/zarr
  arrays (not loaded until sliced).
- `lazynwb.get_internal_paths(path)` → dict of internal path → array accessor.
  For discovering what's in an unfamiliar file.
- `lazynwb.get_table_schema(paths, table_path)` → unified polars schema across
  files, resolving type drift (e.g. int vs float with NaN).
- `lazynwb.get_sql_context(paths)` → a `pl.SQLContext` with every common
  table registered. Nice for ad-hoc SQL.
- `lazynwb.scan_dandiset(dandiset_id, table_path)` → like `scan_nwb` but
  targets a DANDI dandiset's S3 URLs directly.

The full functional surface is the skill's API — there is no separate pynwb-
style object. If you need pynwb-style `nwb.trials.to_dataframe()`, translate
it to these functions (see [pynwb-migration.md](references/pynwb-migration.md)).

## The canonical recipe

Almost every good lazynwb query has this shape:

```python
import lazynwb
import polars as pl

df = (
    lazynwb.scan_nwb(nwb_paths, "/units")
    .filter(pl.col("presence_ratio") >= 0.95, pl.col("location") == "VISp")
    .select("unit_id", "location", "spike_times", "_nwb_path")
    .collect()
)
```

The order matters: **scan → filter → select → collect.**
Each step after `scan_nwb` is what enables pushdown; each step you skip
throws it away.

## Choosing the right primitive

| Intent | Use |
|---|---|
| Filter a big table, possibly across many files | `scan_nwb` |
| Read a whole small table eagerly | `get_df` |
| Session/subject metadata across files | `get_metadata_df` |
| Grab a TimeSeries (e.g. running_speed, LFP) | `get_timeseries` |
| "What's in this file?" | `get_internal_paths` + `get_table_schema` |
| Ad-hoc SQL across all tables | `get_sql_context` |
| Data from a DANDI dandiset | `scan_dandiset` / `get_dandiset_s3_urls` |
| Export tables to parquet/csv/…​ | `convert_nwb_tables` |
| Attach a large array column after filtering | `merge_array_column` |

For signatures and full option lists, see
[references/api-reference.md](references/api-reference.md).

## Multi-file and multi-table joins

Every row from `scan_nwb` / `get_df` carries three internal columns:

- `_nwb_path` (`lazynwb.NWB_PATH_COLUMN_NAME`) — source file
- `_table_path` (`lazynwb.TABLE_PATH_COLUMN_NAME`) — internal table path
- `_table_index` (`lazynwb.TABLE_INDEX_COLUMN_NAME`) — original row index

When joining tables from the same set of files, **always include `_nwb_path`
in the join key**. Otherwise rows from session A can silently pair with
rows from session B.

Runnable template: [scripts/multi_table_join.py](scripts/multi_table_join.py).

## Pitfalls (and why)

- **Eager-then-filter defeats pushdown.** `get_df(...).query(...)` loads
  everything, then throws most of it away. Use `scan_nwb(...).filter(...)
  .select(...).collect()` instead — pushdown means the filter columns are
  read, rows are eliminated, and only the remaining rows' other columns get
  fetched.
- **Use `exact_path=True` when you know the path.** `get_df(path, "trials",
  exact_path=False)` scans the file looking for partial matches — slow.
  `get_df(path, "/intervals/trials", exact_path=True)` jumps straight there.
  (`scan_nwb` always requires the exact path.)
- **Always `_nwb_path` as a join key for multi-file joins** (see above).
- **Schema drift across sessions.** If `brain_region_id` is int in one file
  and has NaN (so float) in another, `scan_nwb` warns and uses the most
  common dtype. If that's wrong, pass `schema_overrides={"brain_region_id":
  pl.Int64}` or limit inference with `infer_schema_length=5`.
- **lazynwb is read-only.** No writing, editing, or running processing
  modules. For those, the user needs pynwb.

## Task recipes

- **"I don't know what's in this file"** — see
  [references/exploring-files.md](references/exploring-files.md) and run
  [scripts/describe_nwb.py](scripts/describe_nwb.py) against the file.
- **"I have pynwb code, I want lazynwb"** — see
  [references/pynwb-migration.md](references/pynwb-migration.md) for a
  mapping table of common pynwb idioms.
- **"I want data from a DANDI dandiset"** — see
  [references/dandi-workflow.md](references/dandi-workflow.md); the entry
  point is `scan_dandiset(dandiset_id, table_path)`.
- **"I want a PSTH / spikes-by-trial"** — see
  [scripts/psth_from_units.py](scripts/psth_from_units.py).

## Output style when writing code for the user

- Import modules, not symbols: `import lazynwb`, `import polars as pl`,
  `import numpy as np`. Call as `lazynwb.scan_nwb(...)`. This matches the
  repo's own examples and is easier to read in notebooks.
- Favor Polars over pandas for any non-trivial query. lazynwb's pushdown is
  Polars-native, and Polars is faster on multi-file data.
- Keep examples runnable: include the `import` lines, spell out the
  `table_path` (e.g. `"/intervals/trials"`, not just `"trials"`), and use
  `exact_path=True` whenever the user has given you a path.
- If the user's data is on S3/GCS, remind them that the pushdown is the
  whole reason lazynwb is faster than pynwb there — and steer them to
  `scan_nwb` rather than `get_df`.
