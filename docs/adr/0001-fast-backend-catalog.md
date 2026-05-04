# ADR 0001: Fast backend catalog architecture

Status: Accepted

Date: 2026-05-03

Issue: #18

## Context

`lazynwb` currently answers public metadata questions through the
`FileAccessor`/HDMF-backed object model. That behavior is useful as a stable
reference path, but it can force broad object traversal or raw data access when
the source is a remote HDF5 NWB file. The fast backend work exists to make
metadata-heavy operations cheap enough for cloud-hosted NWB files while keeping
the public API behavior stable.

The first migrated public API is `get_table_schema`. It is a narrow,
performance-sensitive operation, and it can be implemented from file/catalog
metadata without reading large table columns. The work is therefore
metadata/catalog-first, but data-read-aware: catalog snapshots should preserve
the backend facts needed for later read planning rather than collapse
immediately into only a public `polars` schema.

## Decision

Introduce a private backend catalog layer behind an async-first, class-based
reader `Protocol`. Public APIs may remain synchronous and bridge into the async
backend internally, but backend reader methods should be async so range I/O,
cache I/O, and future remote reads can share one concurrency model.

Backend readers require exact, normalized internal NWB paths. Public API layers
remain responsible for user-facing path normalization, input validation,
multi-file behavior, and public error shaping. The backend layer should not
guess alternate paths or silently traverse the file to reinterpret a public
input.

The private readers have these roles:

- `_HDF5BackendReader` is the fast path for supported remote/single-object HDF5
  sources. It should detect HDF5 content by probing the file signature, parse
  HDF5 metadata needed for catalog snapshots, and avoid broad raw array reads.
- `_ZarrBackendReader` is the explicit Zarr catalog reader. It should prefer
  consolidated `.zmetadata` when available, use targeted Zarr metadata reads
  otherwise, and share catalog/cache models with the HDF5 backend.
- `_AccessorBackendReader` is a compatibility and reference adapter over the
  existing accessor behavior. It is useful for local HDF5/Zarr fixtures,
  baseline tests, and behavior that has not migrated yet, but it is not the
  fallback for migrated remote HDF5 schema paths.

Remote/single-object HDF5 range I/O should use a generic obstore-backed range
reader, not an S3-only reader. The range reader should be private, async, and
usable for obstore-openable URLs such as `s3://`, `https://`, and other schemes
supported by `obstore.store.from_url(...)`. It should expose source identity and
bounded range reads, coalesce aligned ranges where useful, and report enough
debug logging to understand backend selection, request counts, fetched bytes,
cache behavior, and parser coverage.

Catalog objects should be private, accessor-free dataclasses where practical.
They should capture source identity, neutral dtype facts, dataset facts, column
facts, table schema snapshots, and read capability hints. They must not store
`h5py.Dataset`, `h5py.Group`, `zarr.Array`, `zarr.Group`, or other live accessor
objects. Public `polars` schemas are derived from catalog snapshots rather than
being the only cached representation.

SQLite is the persistent snapshot cache substrate. The cache should store source
identity separately from per-kind snapshot tables, starting with table schema
snapshots and leaving room for path summaries, attrs, metadata, and read plans.
Snapshot payloads are JSON with explicit payload versions; pickle is not used.
Cache validation is strict and follows this storage identity order:

1. Object-store version ID.
2. Strong ETag.
3. Last-Modified plus Content-Length.
4. In-process-only behavior for sources without a reliable validator.

Cache hits require a matching source identity, snapshot kind, exact internal
path, and payload version. Identity or payload-version mismatches invalidate the
snapshot instead of trying to reuse it.

Once a remote HDF5 `get_table_schema` source/path is routed to the migrated fast
backend, unsupported parser features and backend read failures should fail fast
with structured parser/backend errors. Those errors should include useful
context such as source identity, exact table path, unsupported feature, and
parser object or offset information where available. They should not silently
fall back to accessor behavior, because doing so would hide parser coverage
gaps and reintroduce the slow remote traversal this architecture is designed to
avoid.

Public `get_internal_paths` now returns accessor-free path metadata dictionaries
and prefers catalog-backed path summaries where available. The accessor-backed
traversal remains as a fallback for sources that cannot use the catalog path.

## Public behavior to preserve

The first migration must preserve the public `get_table_schema` contract:

- `first_n_files_to_infer_schema` keeps limiting schema inference inputs.
- `raise_on_missing` keeps its current behavior.
- Multi-file schema merging keeps counting dtypes per column, warning on
  disagreements, and choosing the most common dtype.
- `exclude_array_columns` is applied after loading full per-file schema
  snapshots, not baked into cache entries.
- Public internal-column handling remains consistent with current behavior.

## Consequences

This architecture gives `get_table_schema` a fast path for supported remote HDF5
files while keeping local/accessor behavior available as a compatibility
reference. It also gives Zarr a first-class backend role instead of letting the
design become HDF5-only.

The main cost is that parser and cache coverage must be explicit. Unsupported
remote HDF5 schema cases will fail loudly once routed to the fast backend, so
follow-up issues need focused parser tests, opt-in remote request-budget tests,
and debug logging that makes failures easy to classify as parser coverage,
storage identity, range I/O, or cache behavior.
