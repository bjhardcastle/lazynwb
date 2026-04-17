# lazynwb API reference

The functions below are what users actually call. The `LazyNWB` class is
intentionally omitted — the maintainer may remove it, so always prefer the
functional API here.

Paths, file formats, and backends are abstracted: a path can be a local
`.nwb` (HDF5 or Zarr), an `s3://…` URL, `gs://…`, `az://…`, or `https://…`.
Iterables of paths are accepted wherever a single path is; every returned
row keeps its source in `_nwb_path`.

## `scan_nwb(source, table_path, **kwargs) -> pl.LazyFrame`

**Lazy read with predicate and projection pushdown.** The default choice
for querying any NWB table.

```python
scan_nwb(
    source,              # path, or iterable of paths
    table_path,          # exact internal path, e.g. "/intervals/trials"
    raise_on_missing=False,
    ignore_errors=False,
    infer_schema_length=None,   # inspect first N files for schema (speed)
    exclude_array_columns=False,
    low_memory=False,
    schema=None,               # full polars schema override
    schema_overrides=None,     # per-column overrides only
    disable_progress=False,
) -> pl.LazyFrame
```

- `table_path` must be exact. It is assumed to be the same in every file.
- When you chain `.filter()` / `.select()` on the returned LazyFrame, the
  filter columns are fetched, rows eliminated, then the remaining columns
  are fetched only for surviving rows. `.collect()` triggers execution.
- `schema_overrides` and `infer_schema_length` handle dtype drift across
  sessions (e.g. int in some files, float-with-NaN in others).

## `read_nwb(source, table_path, **kwargs) -> pl.DataFrame`

Equivalent to `scan_nwb(...).collect()`. Use when you want eager execution
with the same keyword arguments.

## `get_df(nwb_data_sources, search_term, **kwargs) -> pd.DataFrame | pl.DataFrame`

**Eager read.** Useful for small tables, or when you want pandas.

```python
get_df(
    nwb_data_sources,      # path, or iterable of paths
    search_term,           # "/intervals/trials" (set exact_path=True) or
                           # a partial match like "trials" (slower)
    exact_path=False,
    include_column_names=None,
    exclude_column_names=None,
    exclude_array_columns=True,    # note: True by default here
    parallel=True,
    use_process_pool=False,
    disable_progress=False,
    raise_on_missing=False,
    ignore_errors=False,
    low_memory=False,
    as_polars=False,               # pandas by default
)
```

- Returns a **pandas** DataFrame by default (`as_polars=False`). For Polars,
  use `scan_nwb` instead.
- `exclude_array_columns=True` by default. Use `merge_array_column` (pandas
  only) to re-attach an array column after filtering the result.
- Partial `search_term` ("units", "trials") triggers a full-file scan to
  disambiguate. If you know the path, pass `exact_path=True`.
- `TABLE_SHORTCUTS` ({"trials", "epochs", "electrodes", "optophysiology",
  "session"}) are recognized even without `exact_path=True`.

## `get_metadata_df(nwb_path_or_paths, **kwargs) -> pd.DataFrame | pl.DataFrame`

One row per file. Columns: `identifier`, `session_id`, `session_start_time`,
`session_description`, `experiment_description`, `subject_id`, `age`, `sex`,
`species`, `genotype`, `strain`, `date_of_birth`, …, `_nwb_path`.

```python
get_metadata_df(nwb_path_or_paths, disable_progress=False, as_polars=False)
```

Cheap — does not open any tables.

## `get_timeseries(nwb_path, search_term=None, exact_path=False, match_all=False)`

Returns a `TimeSeries` dataclass with lazy `.data` and `.timestamps` (or a
dict of them when `match_all=True`).

```python
ts = lazynwb.get_timeseries(path, "/acquisition/running_speed", exact_path=True)
ts.data          # h5py.Dataset / zarr.Array — slice to read
ts.timestamps    # likewise; synthesized from rate+starting_time if needed
ts.rate, ts.unit, ts.description
# For specialized TS, drop to the raw accessor:
ts.file[ts.path + "/data"]
```

Prefer `exact_path=True`; partial search scans the file.

## `get_internal_paths(nwb_path, **kwargs) -> dict[str, h5py.Dataset | zarr.Array]`

Maps internal paths (e.g. `/units/spike_times`, `/acquisition/running_speed/data`)
to their dataset accessors. For discovering file structure.

```python
get_internal_paths(
    nwb_path,
    include_arrays=True,
    include_table_columns=False,
    include_metadata=False,
    include_specifications=False,
    parents=False,
)
```

Typical call: `list(get_internal_paths(path))` for a quick tour.

## `get_table_schema(file_paths, table_path, **kwargs) -> pl.Schema`

Unified Polars schema across files. Warns on type mismatches and picks the
most common dtype.

```python
get_table_schema(
    file_paths,
    table_path,
    first_n_files_to_infer_schema=None,
    exclude_array_columns=False,
    exclude_internal_columns=False,
    raise_on_missing=False,
)
```

Use before `scan_nwb` if you want to preview what columns you're about to
query, or to seed `schema_overrides`.

## `get_sql_context(nwb_sources, **kwargs) -> pl.SQLContext`

Registers every common table found across `nwb_sources` as a SQL table.

```python
ctx = lazynwb.get_sql_context(paths)
df  = ctx.execute(
    "SELECT unit_id, location FROM units WHERE presence_ratio > 0.9"
).collect()
```

Key kwargs: `min_file_count` (table must appear in at least this many files),
`exclude_timeseries`, `exclude_array_columns`, `full_path` (use the full
internal path as the table name instead of just the last segment),
`table_names` (whitelist).

## `scan_dandiset(dandiset_id, table_path, **kwargs) -> pl.LazyFrame`

Resolves S3 URLs for every NWB asset in the dandiset and hands them to
`scan_nwb`. All `scan_nwb` kwargs pass through via `**scan_kwargs`.

```python
scan_dandiset(
    dandiset_id,                        # e.g. "000363"
    table_path,                         # "/units"
    version=None,                       # most recent published if None
    asset_filter=None,                  # optional: callable(asset_dict) -> bool
    max_assets=None,                    # for testing
    **scan_kwargs,                      # e.g. infer_schema_length=1
)
```

## `get_dandiset_s3_urls(dandiset_id, version=None, order="path") -> list[str]`

Just the list of S3 URLs. Use when you want to hand-pick a subset or do
exploration on one asset before committing to a `scan_dandiset` call.

## `convert_nwb_tables(nwb_sources, output_dir, output_format="parquet", **kwargs) -> dict[str, Path]`

Batch-export every common table to parquet (default), csv, json, excel,
feather, arrow, avro, or delta. One output file per table path.

```python
out = lazynwb.convert_nwb_tables(
    paths, "./out", output_format="parquet", compression="zstd",
)
```

## `merge_array_column(df, column_name, missing_ok=True) -> df`

Pandas-only companion to `get_df`. `get_df` excludes array columns by
default (`exclude_array_columns=True`); call this afterwards to attach the
array only for the rows that remain:

```python
df = lazynwb.get_df(paths, "/units", exact_path=True)
df = df[df["presence_ratio"] > 0.9].pipe(lazynwb.merge_array_column, "spike_times")
```

If you're working in Polars, use `scan_nwb` instead — include the array
column in `.select()` and the pushdown handles it:

```python
# Polars: use scan_nwb, not get_df + merge_array_column
df = (
    lazynwb.scan_nwb(paths, "/units")
    .filter(pl.col("presence_ratio") > 0.9)
    .select("unit_id", "spike_times")
    .collect()
)
```


## Internal column name constants

These are the columns automatically added to every row from `scan_nwb` /
`get_df`; use the constants (not string literals) when joining:

- `lazynwb.NWB_PATH_COLUMN_NAME` → `"_nwb_path"`
- `lazynwb.TABLE_PATH_COLUMN_NAME` → `"_table_path"`
- `lazynwb.TABLE_INDEX_COLUMN_NAME` → `"_table_index"`

For joins across tables from the same files, include `NWB_PATH_COLUMN_NAME`
in the join key so session A doesn't cross-contaminate session B.
