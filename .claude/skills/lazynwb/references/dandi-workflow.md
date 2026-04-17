# DANDI dandiset workflow

[DANDI](https://dandiarchive.org/) hosts published NWB datasets on S3. lazynwb
talks to them directly — no download required. Pushdown still works, so
filtering `/units` across a 174-asset dandiset only pays I/O for the rows
and columns you actually want.

## The happy path

```python
import lazynwb
import polars as pl

lf = lazynwb.scan_dandiset(
    dandiset_id="000363",
    table_path="/units",
    version="0.231012.2129",      # leave None for most-recent published
)

df = (
    lf
    .filter(pl.col("presence_ratio") > 0.9)
    .select("unit_id", "location", "firing_rate")
    .collect()
)
```

`scan_dandiset` resolves S3 URLs for every asset and hands them to
`scan_nwb`. Any `scan_nwb` kwarg passes through (`infer_schema_length`,
`schema_overrides`, `exclude_array_columns`, …).

## "I don't know what tables this dandiset has"

Peek at one asset first, then commit to a full scan:

```python
urls = lazynwb.get_dandiset_s3_urls("000363", version="0.231012.2129")
print(len(urls), "assets")

# Explore one asset's structure
one = urls[0]
for p in sorted(lazynwb.get_internal_paths(one)):
    print(p)

# Pull the schema of the table you care about, across all files
schema = lazynwb.get_table_schema(urls, "/units", first_n_files_to_infer_schema=3)
```

## Scoping down

Large dandisets have hundreds of assets. Use `max_assets` while prototyping:

```python
lf = lazynwb.scan_dandiset(
    "000363", "/units",
    max_assets=2,
    infer_schema_length=1,   # don't probe every file for schema
)
```

For selecting assets by metadata (e.g. only mice of a given strain), pass
`asset_filter=callable`:

```python
def mouse_only(asset: dict) -> bool:
    return "mouse" in asset.get("path", "").lower()

lf = lazynwb.scan_dandiset("000363", "/units", asset_filter=mouse_only)
```

The `asset` dict is the raw DANDI asset metadata (path, size, created,
modified, encoding, etc.).

## Tips

- **First-time auth / network stalls** can look like hangs. Try `max_assets=1`
  and `infer_schema_length=1` to confirm connectivity.
- **Version pinning.** Always pass `version=` for reproducible analysis;
  otherwise DANDI will serve the latest published version, which may change.
- **Schema drift.** Multi-session dandisets often have slight column type
  differences. If you see a warning about inconsistent types, pass
  `schema_overrides` or `infer_schema_length=1` to force a single file's
  schema.
- **Array columns** (e.g. `spike_times`) work exactly as with `scan_nwb` —
  just include them in `.select()` and the pushdown fetches them only for
  rows that survive the filter.
