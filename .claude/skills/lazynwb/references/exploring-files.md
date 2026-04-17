# Exploring an unfamiliar NWB file

When a user hands you an NWB path (or a list of paths) and asks something
like "what's in this?" or "how do I get X from here?", work through these
three steps. They answer three different questions and they're all cheap.

## 1. One-shot summary: `scripts/describe_nwb.py`

For any workflow that starts with "I don't know what's here," run the
bundled script first:

```
python .claude/skills/lazynwb/scripts/describe_nwb.py <path-or-url> [<path-or-url> ...]
```

It prints session/subject metadata, the list of internal paths, and the
per-table schemas. That's almost always enough to know which `scan_nwb`
call to write next.

If you're working in a notebook and can't shell out, do the same thing
inline with the three lazynwb functions below.

## 2. Session/subject metadata — `get_metadata_df`

```python
import lazynwb

meta = lazynwb.get_metadata_df(paths)
# one row per file; columns include identifier, session_id,
# session_start_time, subject_id, age, sex, species, genotype, _nwb_path
```

Cheap — opens the file for attributes only; no tables touched. Use this to
confirm the user's file is what they think it is, or to preview a list of
sessions before committing to a query.

## 3. Internal structure — `get_internal_paths`

```python
paths_map = lazynwb.get_internal_paths(one_nwb_path)
for p in sorted(paths_map):
    print(p)
```

Returns a dict `{internal_path: h5py.Dataset | zarr.Array}`. By default
returns only leaf arrays/tables; flip `parents=True` to see intermediate
groups, `include_table_columns=True` to see each column of each table.

Typical next step: pick an interesting path (e.g. `/units`,
`/intervals/trials`, or `/acquisition/running_speed`) and go to step 4.

## 4. Column types — `get_table_schema`

Once you know the table path, preview the schema before querying:

```python
schema = lazynwb.get_table_schema(paths, "/units")
for name, dtype in schema.items():
    print(f"{name:32s}  {dtype}")
```

Across a multi-file dataset this also surfaces type drift (e.g.
`brain_region_id` stored as i64 in some files, f64 in others). If it
warns, you'll want `schema_overrides={"brain_region_id": pl.Int64}`
when you call `scan_nwb`.

## When to use which

| Question | Function |
|---|---|
| "Is this the right subject / session?" | `get_metadata_df` |
| "What tables/time series does this file contain?" | `get_internal_paths` |
| "What columns are in this table, and what types?" | `get_table_schema` |
| "Give me a one-shot overview" | `scripts/describe_nwb.py` |

For time-series objects specifically (e.g. LFP, running speed), once
`get_internal_paths` has shown you a path like
`/acquisition/running_speed`, reach for `get_timeseries(path,
"/acquisition/running_speed", exact_path=True)`.
