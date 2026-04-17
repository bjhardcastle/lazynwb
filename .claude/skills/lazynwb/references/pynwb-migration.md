# Migrating pynwb code to lazynwb

lazynwb is **read-only** and has no object-oriented `NWBFile` analogue — the
public API is a set of functions. Most pynwb read patterns translate
mechanically. Writing, editing, or running processing modules stays in pynwb.

## Idiom mapping

Below, `path` is an `.nwb` file (local HDF5, local Zarr, or a cloud URL).

### Opening a file

```python
# pynwb
with pynwb.NWBHDF5IO(path, "r") as io:
    nwb = io.read()
    ...

# lazynwb — no explicit open; each function takes the path directly.
# No context manager needed.
```

### Session / subject metadata

```python
# pynwb
nwb.session_id
nwb.session_start_time
nwb.subject.species
nwb.subject.age

# lazynwb — one row per file, all fields in one call
meta = lazynwb.get_metadata_df([path])
row = meta.iloc[0]
row["session_id"], row["session_start_time"], row["species"], row["age"]
```

### Trials / epochs / electrodes

```python
# pynwb
df = nwb.trials.to_dataframe()

# lazynwb — eager (pandas)
df = lazynwb.get_df(path, "/intervals/trials", exact_path=True)

# lazynwb — lazy (recommended, Polars)
lf = lazynwb.scan_nwb(path, "/intervals/trials")
df = lf.filter(pl.col("stim_name") == "Gabor").collect()
```

### Units (with large array columns)

```python
# pynwb — loads everything, including spike_times for every unit
df = nwb.units.to_dataframe()
good = df[df.presence_ratio > 0.9]

# lazynwb — scan_nwb pushdown fetches spike_times only for surviving rows
import polars as pl
df = (
    lazynwb.scan_nwb(path, "/units")
    .filter(pl.col("presence_ratio") > 0.9)
    .select("unit_id", "spike_times")
    .collect()
)
```

### TimeSeries (acquisition, processing)

```python
# pynwb
ts = nwb.acquisition["running_speed"]
data = ts.data[:]
t    = ts.timestamps[:] if ts.timestamps is not None else (
    ts.starting_time + np.arange(ts.data.shape[0]) / ts.rate
)

# lazynwb — same lazy arrays, with timestamps synthesized for you if the
# file stored rate + starting_time instead of a timestamps dataset
ts = lazynwb.get_timeseries(path, "/acquisition/running_speed", exact_path=True)
data = ts.data[:]
t    = ts.timestamps[:]
```

For specialised TimeSeries subclasses, drop to the raw accessor:

```python
ts.file[ts.path + "/custom_field"]
```

### Looping over multiple sessions

```python
# pynwb
dfs = []
for p in paths:
    with pynwb.NWBHDF5IO(p, "r") as io:
        nwb = io.read()
        df = nwb.units.to_dataframe()
        df["session_id"] = nwb.session_id
        dfs.append(df)
df = pd.concat(dfs)

# lazynwb — one call, rows auto-tagged with _nwb_path
df = lazynwb.get_df(paths, "/units", exact_path=True)
# or, with pushdown for filtered queries:
df = (
    lazynwb.scan_nwb(paths, "/units", exclude_array_columns=True)
    .filter(pl.col("presence_ratio") > 0.9)
    .collect()
)
```

### Discovering what's in a file

```python
# pynwb
print(nwb)                   # prints a nested summary
list(nwb.acquisition)
list(nwb.processing)

# lazynwb
list(lazynwb.get_internal_paths(path))
lazynwb.get_metadata_df([path])
```

See [exploring-files.md](exploring-files.md) for the full workflow.

## What lazynwb can't do

- **Write or edit NWB files** — strictly read-only. Use pynwb or `hdmf` for
  writes; convert to lazynwb for downstream analysis.
- **Run NWB processing modules** (spike sorting, etc.) — pynwb + extensions
  are the right tool.
- **Instantiate specialised neurodata types** (e.g. `Units`, `LFP`,
  `BehavioralTimeSeries`) as Python objects with methods. lazynwb gives you
  DataFrames and lazy arrays, not typed objects.

If the user's task requires any of the above, tell them so and keep the
pynwb code; lazynwb can still be the read layer for the parts that only
need data.
