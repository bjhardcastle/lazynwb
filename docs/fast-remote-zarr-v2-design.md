# Fast Remote Zarr v2 Discovery And Large-Array Reads

Status: design draft

## Goal

Remote Zarr should support the same user journey as the fast HDF5 backend:

1. Get users to the valuable NWB objects quickly.
2. Let users select an exact table, column, or TimeSeries array.
3. Download the selected large chunks as fast as possible.

The target format remains Zarr v2. The implementation should improve how
lazynwb discovers metadata and reads selected arrays from remote Zarr stores,
especially object-store-backed NWB stores.

## Current State

lazynwb already has a private backend catalog layer. HDF5 has the most complete
fast path: it owns transport through obstore-backed range reads, builds catalog
snapshots, records source identity, uses the SQLite snapshot cache, and exposes
path summaries without accessor traversal.

Zarr already has a private catalog reader that can build table schema snapshots
and path summaries from Zarr metadata. It prefers consolidated `.zmetadata`
when available and falls back to targeted `.zarray`, `.zattrs`, and `.zgroup`
reads otherwise.

The remaining Zarr gaps are:

- Remote Zarr metadata access still leans on UPath/fsspec-shaped behavior.
- Multi-table schema warming exists for HDF5 but not for Zarr.
- Discovery and array reading are not cleanly separated for the fastest remote
  path.
- Large array reads should use object-store-native concurrency and chunk
  planning rather than broad high-level Zarr traversal.

## Definitions

**Zarr v2 format** means the on-disk/object-store layout with `.zgroup`,
`.zarray`, `.zattrs`, `.zmetadata`, and chunk objects. This remains the NWB
store format target.

**Zarr-Python v2/v3** means the installed Python package major version. This is
an implementation detail and should not change the store format contract.

**Catalog discovery** means schema, path, attrs, shape, dtype, chunking,
compression, and source-identity discovery from metadata. It should not fetch
raw chunks.

**Exact-array read** means the caller already knows the exact internal array
path, such as `/units/spike_times` or `/acquisition/ElectricalSeries/data`, and
the read engine only opens or fetches that array and the chunks needed for the
requested selection.

## Architecture

The core design is catalog-first and transport-aware.

Metadata discovery remains owned by lazynwb. Zarr or numcodecs can still be used
for array semantics and codec decoding where useful, but high-level Zarr group
traversal should not be the hot path for remote discovery.

### Discovery Path

Remote Zarr discovery should use a private object-store metadata client:

- Normalize storage options once.
- Resolve the store root and object prefix.
- Read `.zmetadata` directly when present and valid.
- If consolidated metadata is unavailable, list and read targeted metadata
  objects directly.
- Return the same catalog models used by HDF5 and the existing Zarr reader.

The Zarr metadata catalog should keep producing:

- Table schema snapshots.
- Path summary entries.
- Attr trees.
- Source identity with stable validators where available.
- Read capability hints for downstream planning.

### Array Read Path

Selected data reads should use an exact-array engine:

1. Start from catalog metadata for the exact array path.
2. Compute required chunk coordinates and chunk object keys.
3. Fetch chunk objects concurrently through obstore.
4. Decode chunks with numcodecs or a safe installed Zarr adapter.
5. Assemble the requested output selection.

This path should support large TimeSeries reads and exact table column reads.
It should avoid broad group traversal and should not re-fetch metadata that the
catalog already knows.

### Engine Selection

The public API should not expose a large new surface. Internally, select an
engine by task:

| Task | Preferred engine |
| --- | --- |
| Table schema discovery | Zarr catalog metadata engine |
| Internal path discovery | Zarr catalog metadata engine |
| Attr discovery | Zarr catalog metadata engine |
| SQL context schema warming | Batched Zarr catalog snapshots |
| Exact table column read | Exact-array engine or compatibility fallback |
| Exact TimeSeries slice | Exact-array engine or compatibility fallback |
| Public object traversal | Existing compatibility accessor path |

Configuration can stay private initially, but the likely shape is:

- Zarr transport: `auto`, `obstore`, or `fsspec`.
- Zarr array engine: `auto`, `zarr2`, `zarr3`, or `native-v2`.

The default should be conservative:

- Always use the catalog engine for metadata discovery when available.
- Use obstore for remote object-store sources when supported.
- Keep current compatibility behavior as fallback.
- Use Zarr-Python v3 only for exact-array reads if installed and a compatibility
  probe succeeds.

## Zarr-Python v3

Zarr-Python v3 can be useful, but it should not become the primary discovery
engine yet.

Smoke testing showed:

- Targeted v2 arrays can open with `zarr_format=2` and
  `use_consolidated=False`.
- Broad traversal or consolidated metadata parsing can hit HDMF-generated v2
  metadata edge cases.

Therefore:

- Do not use Zarr-Python v3 for schema discovery.
- Do not use Zarr-Python v3 for path summaries.
- Do not use Zarr-Python v3 for attrs traversal.
- Do use it experimentally for exact-array reads when the installed package is
  v3 and the exact array passes a compatibility probe.

Because Python imports only one `zarr` package at a time, lazynwb cannot switch
between Zarr-Python v2 and v3 inside one normal environment. It can only adapt
to the installed package major version, or use a separate worker environment.
The worker-environment approach is out of scope unless v3 proves dramatically
faster.

## Proposed Deep Modules

### Remote Zarr Object Client

A private async client around obstore for Zarr object access.

Responsibilities:

- Parse remote source URLs into store roots and prefixes.
- Normalize storage options.
- Cache object-store handles.
- Read one object.
- Read many objects concurrently.
- List objects by prefix.
- Read object metadata for source identity.
- Track request counts and fetched bytes for debug logs and benchmarks.

This module should not know NWB semantics.

### Zarr Metadata Access Layer

A private abstraction used by the catalog reader.

Responsibilities:

- Provide `.zmetadata`, `.zarray`, `.zattrs`, and `.zgroup` JSON lookups.
- Hide local-vs-remote access differences.
- Cache consolidated metadata in process.
- Support targeted metadata reads for non-consolidated stores.
- Support path existence and child-name discovery without high-level Zarr group
  traversal.

This module should know Zarr v2 metadata layout but not NWB table rules.

### Zarr Catalog Reader

The existing Zarr backend catalog reader remains the NWB-aware metadata engine.

Responsibilities:

- Build table schema snapshots.
- Build path summaries.
- Build attr trees.
- Apply NWB table, TimeSeries, metadata, and Zarr-only attr rules.
- Interact with the SQLite snapshot cache.
- Expose batched schema snapshot reads.

This module should preserve the existing catalog model contract shared with
HDF5.

### Zarr Chunk Planner

A private pure planning module for exact-array reads.

Responsibilities:

- Convert an array shape, chunk shape, dimension separator, and selection into
  chunk keys.
- Compute per-chunk input selections.
- Compute output placements.
- Account for missing chunks and fill values.
- Keep planning independent from transport and decoding.

This should be heavily unit tested because correctness is subtle and the public
read path depends on it.

### Zarr Chunk Transfer Engine

A private execution module for exact-array reads.

Responsibilities:

- Fetch planned chunk keys using the remote object client.
- Decode chunks using numcodecs or a safe Zarr adapter.
- Assemble the output array.
- Emit debug logs for requests, fetched bytes, chunk count, decoded bytes, and
  elapsed time.

This module should not perform schema discovery or broad traversal.

### Zarr Array Engine Selector

A small private policy module.

Responsibilities:

- Choose native v2 chunk transfer, installed Zarr-Python v2 compatibility,
  installed Zarr-Python v3 exact-array adapter, or fallback behavior.
- Keep engine decisions task-aware.
- Log why an engine was selected or rejected.
- Run and cache compatibility probes where needed.

## Implementation Plan

### Phase 0: Benchmarks And Instrumentation

- Add a remote Zarr discovery benchmark using representative public Zarr v2 NWB
  stores.
- Use the companion benchmark repository at
  `https://github.com/bjhardcastle/neurodatabench` for representative real-world
  workloads, source lists, and cross-format performance comparisons.
- Measure cold and warm schema discovery.
- Measure path summary discovery.
- Measure exact TimeSeries or large array slice reads.
- Record metadata request count, chunk request count, fetched bytes, cache hits,
  and elapsed time.

This gives the implementation a baseline and prevents optimizing blindly.

### Phase 1: Obstore Metadata Access

- Add the remote Zarr object client.
- Teach the Zarr metadata catalog to use obstore for supported remote sources.
- Preserve current local path behavior.
- Preserve fsspec fallback for unsupported or failing cases.
- Keep existing public schema/path/attrs behavior stable.

Success criteria:

- Remote Zarr stores with `.zmetadata` require one metadata object read for the
  catalog payload.
- Remote non-consolidated stores use targeted metadata reads and listing rather
  than accessor traversal.
- Debug logs clearly show obstore metadata use.

### Phase 2: Batched Zarr Schema Snapshots

- Add a batched schema snapshot method to the Zarr backend reader.
- Wire known table-path warming into schema inference and SQL context setup.
- Reuse the same SQLite snapshot cache model as HDF5.

Success criteria:

- Multi-table schema discovery avoids repeated metadata catalog setup.
- Warm schema reads reuse cache entries.
- Existing schema behavior and missing-table behavior are preserved.

### Phase 3: Exact-Array Planning

- Add Zarr v2 array metadata models needed for reading.
- Add the chunk planner.
- Support common selections first: full slices, bounded slices, one-dimensional
  arrays, two-dimensional arrays, and table columns.
- Add support for dimension separator differences.

Success criteria:

- Planner tests cover chunk boundary cases.
- Planner output is independent from obstore and decoding.

### Phase 4: Obstore Chunk Transfer

- Add the native v2 chunk transfer engine.
- Fetch planned chunks concurrently.
- Decode with numcodecs.
- Assemble output arrays.
- Use this path for selected large remote TimeSeries reads where safe.

Success criteria:

- Exact slices match the compatibility Zarr path.
- Large remote reads show improved throughput or lower latency than fsspec-based
  reads.
- Debug logs expose chunk request count and fetched bytes.

### Phase 5: Optional Zarr-Python v3 Exact-Array Adapter

- Detect installed Zarr major version.
- If v3 is installed, probe exact-array compatibility with v2 format and
  consolidated metadata disabled.
- Use the v3 adapter only for exact-array reads when the probe succeeds.
- Keep catalog discovery on lazynwb's metadata engine.

Success criteria:

- v3 is never selected for discovery tasks.
- v3 exact-array reads are version-gated and easy to disable.
- Known HDMF-generated v2 metadata edge cases fail closed to the safe path.

## Testing Strategy

Tests should focus on externally visible behavior and performance-relevant
contracts, with deep isolated tests for the private planning and transport
modules.

Catalog tests:

- Public schema from the fast Zarr catalog matches existing accessor behavior.
- Path summary discovery includes expected DynamicTable and TimeSeries paths.
- Attr discovery filters Zarr-only attrs consistently.
- Consolidated and targeted metadata produce equivalent public results.
- Remote-shaped stores without a `.zarr` suffix still use catalog summaries.

Cache tests:

- Cold catalog reads populate snapshots.
- Warm reads reuse snapshots without metadata re-fetching.
- Source identity mismatches do not reuse stale snapshots.

Remote object client tests:

- Object root and prefix parsing.
- Storage option normalization.
- Single-object reads.
- Batched reads.
- Prefix listing.
- Request and byte counters.
- Failure and fallback behavior.

Chunk planner tests:

- One-dimensional bounded slices.
- Two-dimensional bounded slices.
- Slices crossing chunk boundaries.
- Full-array reads.
- Missing chunk fill behavior.
- Dimension separator handling.
- Output placement correctness.

Chunk transfer tests:

- Compressed and uncompressed chunk decoding.
- Concurrent fetch behavior with fake object clients.
- Exact output equality against the compatibility Zarr path.
- Large selected reads that avoid fetching unrelated chunks.

Version-adapter tests:

- Zarr-Python v3 exact-array probe is optional and version-gated.
- v3 is not selected for schema, path, or attrs discovery.
- Failed v3 probes fall back cleanly.

Benchmark tests:

- Cold and warm remote Zarr schema discovery.
- Remote Zarr path discovery.
- Large exact-array slice reads.
- Request budgets that prevent accidental raw chunk reads during discovery.

## Risks

- Non-consolidated remote stores can require many metadata objects. Listing and
  targeted reads need careful batching.
- Zarr v2 string/object dtype behavior can differ across HDMF-generated stores.
- Chunk decoding correctness is easy to get subtly wrong around partial chunks,
  fill values, filters, and dimension separators.
- Optional Zarr-Python v3 behavior can change across releases.
- Object-store providers differ in listing, metadata, auth, and consistency
  details.

## Non-Goals

- Migrating NWB stores to Zarr format 3.
- Requiring Zarr-Python v3.
- Reimplementing the entire Zarr API.
- Reimplementing compression codecs.
- Supporting writes.
- Removing fsspec compatibility fallback.
- Changing public APIs beyond a small configuration surface if needed.

## Open Questions

- Should the native v2 chunk transfer engine become the default for all remote
  exact-array reads, or only for arrays above a size threshold?
- Which exact public calls should first route to the native chunk transfer path:
  TimeSeries data, table columns, or both?
- How much local chunk caching is useful without increasing memory surprise?
- Should chunk transfer concurrency be global, per-source, or per-read?
- What request and byte budgets should remote Zarr discovery benchmarks enforce?
- Should obstore transport be enabled by the existing global obstore setting, a
  new Zarr-specific setting, or auto-detected for remote object-store sources?
