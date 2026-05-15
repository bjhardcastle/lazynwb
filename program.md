# Autoresearch Program

## Goal
Optimize `cold_zarr_schema_seconds` (lower is better) by modifying `src/lazynwb/_zarr/reader.py`.

## Ideas to Explore
1. Prefer direct object-store metadata reads for remote Zarr v2 stores instead of
   `UPath`/fsspec metadata probing.
2. Make consolidated `.zmetadata` a one-object cold metadata read when present.
3. For non-consolidated stores, list metadata keys once and batch targeted JSON
   reads instead of traversing groups through accessors.
4. Reuse one metadata catalog across same-source table schema snapshots.
5. Keep exact-array chunk planning and transfer separate from schema discovery
   so discovery never fetches raw chunks.
6. Add debug counters for metadata requests, chunk requests, fetched bytes, and
   elapsed time so benchmark regressions are visible.

## Constraints
- Time budget per experiment: 10m
- Only modify: src/lazynwb/_zarr/reader.py
- Eval command: `uv run python benchmarks/zarr_schema_benchmark.py --max-sources 1 --json-output /tmp/lazynwb-zarr-schema-benchmark.json`
- Preserve local Zarr behavior and fsspec compatibility fallback.
- Do not change public APIs unless the fast path needs a private selector.
- Keep store format support at Zarr v2.
