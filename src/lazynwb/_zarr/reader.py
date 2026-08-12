from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import threading
import time
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import obstore
import obstore.store
import upath

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog._schema as catalog_schema
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._config
import lazynwb._storage_options
import lazynwb._zarr.chunk_planner as zarr_chunk_planner
import lazynwb._zarr.chunk_transfer as zarr_chunk_transfer
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)

_ZARR_ONLY_ATTR_NAMES = frozenset({"_ARRAY_DIMENSIONS", "zarr_dtype"})
_ZARR_METADATA_FILENAMES = (".zarray", ".zattrs", ".zgroup")
_ZARR_CONSOLIDATED_METADATA_CACHE_LOCK = threading.RLock()
_REMOTE_ZARR_SCHEMES = frozenset({"s3", "gs", "gcs", "az", "abfs", "http", "https"})
_OBSTORE_REMOTE_ZARR_SCHEMES = frozenset({"s3", "gs", "gcs", "az", "abfs"})
_ZARR_METADATA_PAYLOAD_VERSION = 1
_ZARR_METADATA_PAYLOAD_FORMAT = "zarr-v2-metadata-catalog"
_ZarrPath = pathlib.Path | upath.UPath
_ObstoreStoreCacheKey = tuple[str, tuple[tuple[str, str], ...]]


@dataclasses.dataclass(slots=True)
class _ZarrTableSchemaScanError(RuntimeError):
    """Structured per-table error from an internal multi-table schema scan."""

    source_url: str
    table_path: str
    feature: str
    detail: str

    def __str__(self) -> str:
        return (
            f"Zarr table schema scan failed for {self.source_url!r} at "
            f"{self.table_path!r}: {self.feature}: {self.detail}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrTableSchemaScanResult:
    """Per-table result from a same-source schema scan lifecycle."""

    table_path: str
    snapshot: catalog_models._TableSchemaSnapshot | None
    error: Exception | None
    _scan_metrics: catalog_backend._TableSchemaScanMetrics
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.snapshot is not None and self.error is None


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrConsolidatedMetadataCacheEntry:
    stat_key: tuple[int, int] | None
    metadata: Mapping[str, object] | None


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteZarrStoreContext:
    store_url: str
    root_prefix: str
    storage_options: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteZarrConsolidatedPayload:
    source_identity: catalog_models._SourceIdentity
    metadata: Mapping[str, object]
    fetched_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteZarrChildListing:
    child_names: tuple[str, ...]


_ZARR_CONSOLIDATED_METADATA_CACHE: dict[
    str,
    _ZarrConsolidatedMetadataCacheEntry,
] = {}
_REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE: dict[
    str,
    _RemoteZarrConsolidatedPayload | None,
] = {}
_OBSTORE_STORE_CACHE_LOCK = threading.RLock()
_OBSTORE_STORE_CACHE: dict[_ObstoreStoreCacheKey, obstore.store.ObjectStore] = {}


class _ZarrBackendReader:
    """Explicit Zarr catalog reader for table schema snapshots."""

    def __init__(
        self,
        source: lazynwb.types_.PathLike,
        cache: cache_sqlite._SQLiteSnapshotCache | None = None,
        *,
        chunk_transfer_config: zarr_chunk_transfer._ZarrChunkTransferConfig | None = None,
    ) -> None:
        self._source = source
        self._source_path = _source_path(source)
        self._cache = cache
        self._chunk_transfer_config = (
            chunk_transfer_config or zarr_chunk_transfer._ZarrChunkTransferConfig()
        )
        self._source_identity: catalog_models._SourceIdentity | None = None
        self._remote_metadata_client = _remote_metadata_client_if_available(source)
        self._remote_consolidated_payload: _RemoteZarrConsolidatedPayload | None = None
        self._remote_metadata_json_cache: dict[
            str,
            Mapping[str, object] | None,
        ] = {}
        self._remote_child_names_cache: dict[str, tuple[str, ...]] = {}
        self._metadata_catalog: _ZarrMetadataCatalog | None = None
        self._metadata_payload_lookup_keys: set[str] = set()
        self._metadata_payload_loaded_from_cache = False
        self._metadata_payload_options_key: str | None = None
        self._metadata_payload_origin: str | None = None
        self.metadata_read_count = 0
        self.metadata_bytes_fetched = 0
        self.chunk_read_count = 0
        self.chunk_bytes_fetched = 0
        self.used_consolidated_metadata = False

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        if self._source_identity is None:
            self._source_identity = await asyncio.to_thread(self._build_source_identity)
        return self._source_identity

    async def _read_table_schema_snapshots(
        self,
        exact_table_paths: tuple[str, ...],
    ) -> dict[str, _ZarrTableSchemaScanResult]:
        normalized_paths = tuple(dict.fromkeys(exact_table_paths))
        for exact_table_path in normalized_paths:
            catalog_backend._require_exact_normalized_path(exact_table_path)
        logger.debug(
            "explicit multi-table Zarr schema scan for %s: tables=%d",
            self._source,
            len(normalized_paths),
        )
        results: dict[str, _ZarrTableSchemaScanResult] = {}
        for exact_table_path in normalized_paths:
            metadata_reads_before = self.metadata_read_count
            metadata_bytes_before = self.metadata_bytes_fetched
            started = time.perf_counter()
            snapshot: catalog_models._TableSchemaSnapshot | None = None
            error: Exception | None = None
            try:
                snapshot = await self.read_table_schema_snapshot(exact_table_path)
            except KeyError as exc:
                error = _ZarrTableSchemaScanError(
                    source_url=self._schema_scan_source_url(),
                    table_path=exact_table_path,
                    feature="missing_table",
                    detail=repr(exc),
                )
            except Exception as exc:
                error = _ZarrTableSchemaScanError(
                    source_url=self._schema_scan_source_url(),
                    table_path=exact_table_path,
                    feature="unexpected_error",
                    detail=repr(exc),
                )
            metadata_read_count = self.metadata_read_count - metadata_reads_before
            metadata_bytes_fetched = (
                self.metadata_bytes_fetched - metadata_bytes_before
            )
            result = _ZarrTableSchemaScanResult(
                table_path=exact_table_path,
                snapshot=snapshot,
                error=error,
                _scan_metrics=catalog_backend._TableSchemaScanMetrics(
                    request_count=metadata_read_count,
                    fetched_bytes=metadata_bytes_fetched,
                ),
                elapsed_seconds=time.perf_counter() - started,
            )
            results[exact_table_path] = result
            if error is None:
                logger.debug(
                    "same-source Zarr schema scan built %s/%s "
                    "(requests=%d bytes=%d elapsed=%.3fs consolidated=%s)",
                    self._schema_scan_source_url(),
                    exact_table_path,
                    metadata_read_count,
                    metadata_bytes_fetched,
                    result.elapsed_seconds,
                    self.used_consolidated_metadata,
                )
            else:
                logger.debug(
                    "same-source Zarr schema scan error for %s/%s: %r",
                    self._schema_scan_source_url(),
                    exact_table_path,
                    error,
                )
        return results

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog_backend._require_exact_normalized_path(exact_table_path)
        logger.debug(
            "single-table Zarr schema scan for %s/%s",
            self._source,
            exact_table_path,
        )
        t0 = time.perf_counter()
        source_identity = await self.get_source_identity()
        if self._cache is not None:
            cached = await self._cache.get_table_schema_snapshot(
                source_identity,
                exact_table_path,
            )
            if cached.hit and cached.snapshot is not None:
                logger.debug(
                    "loaded Zarr table schema snapshot for %s/%s from cache",
                    source_identity.source_url,
                    exact_table_path,
                )
                return cached.snapshot
        metadata_options_key = _zarr_metadata_options_key(
            "table_schema",
            exact_table_path,
        )
        await self._load_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        snapshot = await asyncio.to_thread(
            self._read_table_schema_snapshot_sync,
            exact_table_path,
            source_identity,
        )
        if self._cache is not None:
            await self._cache.put_table_schema_snapshot(snapshot)
            await self._write_persistent_metadata_payload(
                source_identity,
                metadata_options_key,
            )
        logger.debug(
            "built Zarr table schema snapshot for %s/%s with %d columns in %.2f s "
            "(metadata_reads=%d consolidated=%s)",
            source_identity.source_url,
            exact_table_path,
            len(snapshot.columns),
            time.perf_counter() - t0,
            self.metadata_read_count,
            self.used_consolidated_metadata,
        )
        return snapshot

    async def read_path_summary(
        self,
    ) -> tuple[catalog_models._PathSummaryEntry, ...]:
        started = time.perf_counter()
        source_identity = await self.get_source_identity()
        metadata_options_key = _zarr_metadata_options_key("path_summary", "/")
        await self._load_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        summary = await asyncio.to_thread(self._read_path_summary_sync)
        await self._write_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        logger.debug(
            "built Zarr path summary for %s with %d entries in %.2f s "
            "(metadata_reads=%d consolidated=%s)",
            source_identity.source_url,
            len(summary),
            time.perf_counter() - started,
            self.metadata_read_count,
            self.used_consolidated_metadata,
        )
        return summary

    async def _read_attrs(
        self,
        internal_path: str,
    ) -> Mapping[str, object]:
        source_identity = await self.get_source_identity()
        normalized_path = _normalize_zarr_metadata_path(internal_path)
        metadata_options_key = _zarr_metadata_options_key("attrs", normalized_path)
        await self._load_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        attrs = await asyncio.to_thread(self._read_attrs_sync, internal_path)
        await self._write_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        logger.debug(
            "read Zarr attrs for %s/%s with %d keys "
            "(metadata_reads=%d consolidated=%s metadata_payload_cache=%s)",
            source_identity.source_url,
            internal_path,
            len(attrs),
            self.metadata_read_count,
            self.used_consolidated_metadata,
            self._metadata_payload_loaded_from_cache,
        )
        return attrs

    async def _read_attrs_tree(
        self,
        parent_path: str,
    ) -> dict[str, Mapping[str, object]]:
        source_identity = await self.get_source_identity()
        normalized_path = _normalize_zarr_metadata_path(parent_path)
        metadata_options_key = _zarr_metadata_options_key(
            "attrs_tree",
            normalized_path,
        )
        await self._load_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        attrs_tree = await asyncio.to_thread(self._read_attrs_tree_sync, parent_path)
        await self._write_persistent_metadata_payload(
            source_identity,
            metadata_options_key,
        )
        logger.debug(
            "read Zarr attrs tree for %s/%s with %d entries "
            "(metadata_reads=%d consolidated=%s metadata_payload_cache=%s)",
            source_identity.source_url,
            parent_path,
            len(attrs_tree),
            self.metadata_read_count,
            self.used_consolidated_metadata,
            self._metadata_payload_loaded_from_cache,
        )
        return attrs_tree

    async def read_array_selection(
        self,
        exact_array_path: str,
        selection: object = None,
    ) -> np.ndarray:
        results = await self._read_array_selections(exact_array_path, (selection,))
        return results[0]

    async def _read_array_selections(
        self,
        exact_array_path: str,
        selections: Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        normalized_path = _normalize_zarr_metadata_path(exact_array_path)
        if not selections:
            return ()
        client = self._remote_metadata_client
        if client is None:
            logger.debug(
                "rejecting native Zarr array selections for %s/%s: "
                "obstore transport unavailable",
                self._source,
                normalized_path,
            )
            raise RuntimeError("native Zarr array reads require obstore metadata client")
        zarray = await asyncio.to_thread(
            self._read_zarray_metadata_sync,
            normalized_path,
        )
        if zarray is None:
            logger.debug(
                "rejecting native Zarr array selections for %s/%s: missing .zarray",
                self._source,
                normalized_path,
            )
            raise KeyError(exact_array_path)
        planned_reads: list[
            tuple[
                object,
                zarr_chunk_transfer._ZarrV2ArraySpec,
                zarr_chunk_planner._ArrayChunkPlan,
            ]
        ] = []
        try:
            for selection in selections:
                spec, plan = zarr_chunk_transfer._plan_zarr_v2_array_selection(
                    array_path=normalized_path,
                    zarray=zarray,
                    selection=selection,
                )
                planned_reads.append((selection, spec, plan))
        except Exception as exc:
            logger.debug(
                "rejecting native Zarr array selections for %s/%s: %r",
                self._source,
                normalized_path,
                exc,
            )
            raise
        total_chunk_count = sum(plan.chunk_count for _, _, plan in planned_reads)
        spec = planned_reads[0][1]
        max_concurrency = zarr_chunk_transfer._max_concurrency(
            self._chunk_transfer_config
        )
        logger.debug(
            "selecting native Zarr v2 array engine for %s/%s with %d selection(s) "
            "(shape=%s chunks=%s dtype=%s total_chunk_count=%d "
            "max_concurrency=%d)",
            self._source,
            normalized_path,
            len(planned_reads),
            spec.shape,
            spec.chunks,
            spec.dtype,
            total_chunk_count,
            max_concurrency,
        )
        request_count_before = client.request_count
        fetched_bytes_before = client.bytes_fetched
        started = time.perf_counter()
        transfer_limiter = zarr_chunk_transfer._ZarrChunkTransferLimiter(
            self._chunk_transfer_config
        )
        try:
            results = await asyncio.gather(
                *(
                    zarr_chunk_transfer._read_zarr_v2_array_selection_from_plan(
                        client,
                        spec=selection_spec,
                        plan=plan,
                        config=self._chunk_transfer_config,
                        limiter=transfer_limiter,
                    )
                    for _, selection_spec, plan in planned_reads
                )
            )
        except Exception as exc:
            logger.debug(
                "native Zarr array selections failed for %s/%s after %.3f s; "
                "falling back is recommended: %r",
                self._source,
                normalized_path,
                time.perf_counter() - started,
                exc,
                exc_info=True,
            )
            raise
        chunk_requests = client.request_count - request_count_before
        chunk_bytes = client.bytes_fetched - fetched_bytes_before
        self.chunk_read_count += chunk_requests
        self.chunk_bytes_fetched += chunk_bytes
        logger.debug(
            "read native Zarr array selections for %s/%s in %.3f s "
            "(selection_count=%d shapes=%s total_chunk_count=%d "
            "chunk_requests=%d chunk_bytes=%d max_concurrency=%d)",
            self._source,
            normalized_path,
            time.perf_counter() - started,
            len(results),
            [result.shape for result in results],
            total_chunk_count,
            chunk_requests,
            chunk_bytes,
            max_concurrency,
        )
        return tuple(results)

    async def close(self) -> None:
        logger.debug(
            "closing Zarr backend reader for %s (metadata_reads=%d "
            "metadata_bytes=%d chunk_reads=%d chunk_bytes=%d consolidated=%s)",
            self._source,
            self.metadata_read_count,
            self.metadata_bytes_fetched,
            self.chunk_read_count,
            self.chunk_bytes_fetched,
            self.used_consolidated_metadata,
        )

    def _schema_scan_source_url(self) -> str:
        if self._source_identity is not None:
            return self._source_identity.source_url
        return str(self._source)

    def _metadata_catalog_for_reader(self) -> _ZarrMetadataCatalog:
        if self._metadata_catalog is None:
            self._metadata_catalog = _ZarrMetadataCatalog(self._source_path, self)
            logger.debug(
                "initialized reusable Zarr metadata catalog for %s",
                self._source,
            )
        return self._metadata_catalog

    def _read_zarray_metadata_sync(
        self,
        path: str,
    ) -> Mapping[str, object] | None:
        return self._metadata_catalog_for_reader().get_zarray(path)

    def _build_source_identity(self) -> catalog_models._SourceIdentity:
        remote_payload = _get_remote_consolidated_payload(self)
        if remote_payload is not None:
            logger.debug(
                "built Zarr source identity for %s from obstore .zmetadata "
                "(bytes=%d)",
                self._source,
                remote_payload.fetched_bytes,
            )
            return remote_payload.source_identity
        remote_identity = _get_remote_non_consolidated_source_identity(self)
        if remote_identity is not None:
            return remote_identity
        if not self._source_path.exists():
            return catalog_models._SourceIdentity(
                source_url=str(self._source),
                in_process_token=f"zarr-missing:{id(self)}",
            )
        metadata_path = self._source_path / ".zmetadata"
        if metadata_path.exists():
            stat = metadata_path.stat()
            stat_info = getattr(stat, "info", {}) or {}
            return catalog_models._SourceIdentity(
                source_url=self._source_path.as_posix(),
                content_length=stat.st_size,
                version_id=_stat_info_str(stat_info, "VersionId", "version"),
                etag=_stat_info_str(stat_info, "ETag", "e_tag"),
                last_modified=_mtime_iso(stat.st_mtime),
            )
        metadata_files = tuple(
            path
            for path in self._source_path.rglob("*")
            if path.name in {".zarray", ".zattrs", ".zgroup"}
        )
        if metadata_files:
            stats = [path.stat() for path in metadata_files]
            return catalog_models._SourceIdentity(
                source_url=self._source_path.as_posix(),
                content_length=sum(stat.st_size for stat in stats),
                last_modified=_mtime_iso(max(stat.st_mtime for stat in stats)),
            )
        return catalog_models._SourceIdentity(
            source_url=self._source_path.as_posix(),
            in_process_token=f"zarr-empty:{id(self)}",
        )

    def _read_table_schema_snapshot_sync(
        self,
        exact_table_path: str,
        source_identity: catalog_models._SourceIdentity,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog = self._metadata_catalog_for_reader()
        column_entries = _get_table_column_entries(catalog, exact_table_path)
        if not column_entries:
            raise KeyError(exact_table_path)
        column_facts = _column_facts_from_zarr_entries(column_entries)
        table_rules = catalog_schema._get_table_schema_rules(
            exact_table_path,
            column_facts.values(),
        )
        columns = tuple(
            _column_schema_from_zarr_entry(
                entry=entry,
                column_facts=column_facts[entry.name],
                source_identity=source_identity,
                table_path=exact_table_path,
                table_rules=table_rules,
            )
            for entry in column_entries
        )
        return catalog_models._TableSchemaSnapshot(
            source_identity=source_identity,
            table_path=exact_table_path,
            backend="zarr",
            columns=columns,
            table_length=catalog_schema._get_table_length(columns),
        )

    def _read_path_summary_sync(
        self,
    ) -> tuple[catalog_models._PathSummaryEntry, ...]:
        catalog = self._metadata_catalog_for_reader()
        return catalog.read_path_summary()

    def _read_attrs_sync(
        self,
        internal_path: str,
    ) -> Mapping[str, object]:
        normalized_path = _normalize_zarr_metadata_path(internal_path)
        catalog = self._metadata_catalog_for_reader()
        if not catalog.path_exists(normalized_path):
            raise KeyError(internal_path)
        return dict(catalog.get_zattrs(normalized_path))

    def _read_attrs_tree_sync(
        self,
        parent_path: str,
    ) -> dict[str, Mapping[str, object]]:
        catalog = self._metadata_catalog_for_reader()
        return {
            path: dict(attrs)
            for path, attrs in catalog.read_attrs_tree(parent_path).items()
        }

    async def _load_persistent_metadata_payload(
        self,
        source_identity: catalog_models._SourceIdentity,
        options_key: str,
    ) -> None:
        if self._cache is None or options_key in self._metadata_payload_lookup_keys:
            return
        self._metadata_payload_lookup_keys.add(options_key)
        lookup = await self._cache.get_metadata_payload(
            source_identity,
            backend="zarr",
            payload_version=_ZARR_METADATA_PAYLOAD_VERSION,
            options_key=options_key,
        )
        logger.debug(
            "Zarr metadata payload cache lookup for %s: %s",
            source_identity.source_url,
            lookup.reason,
        )
        if lookup.payload is None:
            if (
                self._metadata_payload_origin == "targeted"
                and self._metadata_payload_options_key != options_key
            ):
                logger.debug(
                    "discarding scoped targeted Zarr metadata catalog for %s: "
                    "cached scope %s cannot satisfy %s",
                    source_identity.source_url,
                    self._metadata_payload_options_key,
                    options_key,
                )
                self._metadata_catalog = None
                self._metadata_payload_options_key = None
                self._metadata_payload_origin = None
                self._metadata_payload_loaded_from_cache = False
            return
        try:
            metadata, origin = _metadata_payload_to_zarr_metadata(lookup.payload)
        except Exception:
            logger.debug(
                "discarding invalid Zarr metadata payload for %s",
                source_identity.source_url,
                exc_info=True,
            )
            return
        self._metadata_catalog = _ZarrMetadataCatalog(
            self._source_path,
            self,
            metadata=metadata,
            metadata_origin=origin,
        )
        self._metadata_payload_loaded_from_cache = True
        self._metadata_payload_options_key = options_key
        self._metadata_payload_origin = origin
        if origin == "consolidated":
            self.used_consolidated_metadata = True
            _store_shared_consolidated_metadata_payload(self._source_path, metadata)
        logger.debug(
            "loaded Zarr metadata payload for %s from persistent cache "
            "(entries=%d origin=%s)",
            source_identity.source_url,
            len(metadata),
            origin,
        )

    async def _write_persistent_metadata_payload(
        self,
        source_identity: catalog_models._SourceIdentity,
        options_key: str,
    ) -> None:
        if self._cache is None or self._metadata_catalog is None:
            return
        payload = _metadata_payload_from_catalog(self._metadata_catalog)
        if payload is None:
            logger.debug(
                "skipping Zarr metadata payload cache write for %s: "
                "metadata catalog has no payload",
                source_identity.source_url,
            )
            return
        await self._cache.put_metadata_payload(
            source_identity,
            backend="zarr",
            payload_version=_ZARR_METADATA_PAYLOAD_VERSION,
            options_key=options_key,
            payload=payload,
        )
        self._metadata_payload_options_key = options_key
        self._metadata_payload_origin = str(payload.get("origin", "targeted"))


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrEntry:
    name: str
    path: str
    zarray: Mapping[str, object] | None
    zattrs: Mapping[str, object]

    @property
    def is_group(self) -> bool:
        return self.zarray is None

    @property
    def dtype(self) -> np.dtype | None:
        if self.zarray is None:
            return None
        return np.dtype(str(self.zarray["dtype"]))

    @property
    def shape(self) -> tuple[int, ...] | None:
        if self.zarray is None:
            return None
        return tuple(int(item) for item in _as_sequence(self.zarray["shape"]))

    @property
    def chunks(self) -> tuple[int, ...] | None:
        if self.zarray is None:
            return None
        chunks = self.zarray.get("chunks")
        if chunks is None:
            return None
        return tuple(int(item) for item in _as_sequence(chunks))

    @property
    def ndim(self) -> int | None:
        if self.shape is None:
            return None
        return len(self.shape)


class _ZarrMetadataCatalog:
    def __init__(
        self,
        root: _ZarrPath,
        reader: _ZarrBackendReader | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
        metadata_origin: str = "targeted",
    ) -> None:
        self._root = root
        self._reader = reader
        if metadata is None:
            self._metadata = _get_shared_consolidated_metadata(root, reader)
            self._metadata_origin = (
                "consolidated" if self._metadata is not None else "targeted"
            )
        else:
            self._metadata = metadata
            self._metadata_origin = metadata_origin
        self._targeted_metadata_cache: dict[str, Mapping[str, object] | None] = {}
        self._consolidated_paths = (
            _consolidated_metadata_paths(self._metadata)
            if self._metadata is not None
            else None
        )
        self._consolidated_children = (
            _consolidated_child_index(self._consolidated_paths)
            if self._consolidated_paths is not None
            else None
        )

    @property
    def has_consolidated_metadata(self) -> bool:
        return self._metadata is not None and self._metadata_origin == "consolidated"

    def get_zarray(self, path: str) -> Mapping[str, object] | None:
        return self._get_json(_normalize_zarr_metadata_path(path), ".zarray")

    def get_zattrs(self, path: str) -> Mapping[str, object]:
        return self._get_json(_normalize_zarr_metadata_path(path), ".zattrs") or {}

    def path_exists(self, path: str) -> bool:
        normalized_path = _normalize_zarr_metadata_path(path)
        if self._metadata is not None:
            if normalized_path == "":
                root_children = (
                    ()
                    if self._consolidated_children is None
                    else self._consolidated_children.get("", ())
                )
                return self._has_consolidated_metadata("") or bool(root_children)
            return self._has_consolidated_metadata(normalized_path)
        if _catalog_has_remote_metadata_client(self):
            if _remote_path_exists(self._reader, normalized_path):
                return True
            try:
                return bool(_remote_child_names(self._reader, normalized_path))
            except KeyError:
                return False
        if normalized_path == "":
            exists = self._root.exists()
            if exists:
                self._read_local_metadata_path(normalized_path)
            return exists
        root = self._root / normalized_path
        exists = any(
            (root / filename).exists() for filename in _ZARR_METADATA_FILENAMES
        )
        if exists:
            self._read_local_metadata_path(normalized_path)
        return exists

    def read_path_summary(self) -> tuple[catalog_models._PathSummaryEntry, ...]:
        entries: list[catalog_models._PathSummaryEntry] = []
        self._collect_path_summary_entries(
            "",
            entries,
            include_self=False,
        )
        logger.debug(
            "built Zarr metadata catalog path summary for %s with %d entries",
            self._root,
            len(entries),
        )
        return tuple(entries)

    def read_attrs_tree(self, parent_path: str) -> dict[str, Mapping[str, object]]:
        normalized_parent_path = _normalize_zarr_metadata_path(parent_path)
        if not self.path_exists(normalized_parent_path):
            logger.debug(
                "Zarr metadata catalog attrs path %r not found under %s",
                parent_path,
                self._root,
            )
            return {}
        entries: dict[str, Mapping[str, object]] = {}
        self._collect_attrs_tree(normalized_parent_path, entries)
        logger.debug(
            "built Zarr metadata catalog attrs tree for %s/%s with %d entries "
            "(consolidated=%s)",
            self._root,
            normalized_parent_path or ".",
            len(entries),
            self.has_consolidated_metadata,
        )
        return entries

    def export_metadata_snapshot(self) -> tuple[dict[str, object], str] | None:
        metadata: dict[str, object] = {}
        if self._metadata is not None:
            metadata.update(
                {
                    str(key): value
                    for key, value in self._metadata.items()
                    if isinstance(key, str) and isinstance(value, Mapping)
                }
            )
            origin = self._metadata_origin
        else:
            metadata.update(
                {
                    key: value
                    for key, value in self._targeted_metadata_cache.items()
                    if isinstance(value, Mapping)
                }
            )
            if self._reader is not None:
                metadata.update(
                    {
                        key: value
                        for key, value in self._reader._remote_metadata_json_cache.items()
                        if isinstance(value, Mapping)
                    }
                )
            origin = "targeted"
        if not metadata:
            return None
        logger.debug(
            "exported Zarr metadata catalog snapshot for %s with %d entries "
            "(origin=%s)",
            self._root,
            len(metadata),
            origin,
        )
        return metadata, origin

    def child_names(self, parent_path: str) -> tuple[str, ...]:
        parent_path = _normalize_zarr_metadata_path(parent_path)
        if self._metadata is not None:
            names = self._consolidated_child_names(parent_path)
        elif _catalog_has_remote_metadata_client(self):
            names = _remote_child_names(self._reader, parent_path)
        else:
            parent = self._root / parent_path
            if not parent.exists():
                raise KeyError(parent_path)
            self._get_json(parent_path, ".zgroup")
            names = tuple(
                child.name
                for child in parent.iterdir()
                if child.is_dir() and not child.name.startswith(".")
            )
        return tuple(sorted(names))

    def _collect_path_summary_entries(
        self,
        path: str,
        entries: list[catalog_models._PathSummaryEntry],
        *,
        include_self: bool,
    ) -> None:
        entry = _entry_for_path(self, path, path.rsplit("/", 1)[-1])
        if include_self:
            entries.append(
                catalog_models._PathSummaryEntry(
                    path=_summary_path(path),
                    is_group=entry.is_group,
                    is_dataset=not entry.is_group,
                    shape=entry.shape,
                    attrs_json=catalog_models._attrs_to_tuple(
                        _filter_zarr_only_attrs(entry.zattrs)
                    ),
                )
            )
        if not entry.is_group:
            return
        for name in self.child_names(path):
            child_path = f"{path}/{name}" if path else name
            self._collect_path_summary_entries(
                child_path,
                entries,
                include_self=True,
            )

    def _collect_attrs_tree(
        self,
        path: str,
        entries: dict[str, Mapping[str, object]],
    ) -> None:
        entries[path] = self.get_zattrs(path)
        if self.get_zarray(path) is not None:
            return
        for name in self.child_names(path):
            child_path = f"{path}/{name}" if path else name
            self._collect_attrs_tree(child_path, entries)

    def _get_json(self, path: str, filename: str) -> Mapping[str, object] | None:
        path = _normalize_zarr_metadata_path(path)
        metadata_key = f"{path}/{filename}" if path else filename
        if self._metadata is not None:
            value = self._metadata.get(metadata_key)
            return value if isinstance(value, Mapping) else None
        if metadata_key in self._targeted_metadata_cache:
            return self._targeted_metadata_cache[metadata_key]
        if _catalog_has_remote_metadata_client(self):
            value = _remote_json_metadata(self._reader, path, filename)
            self._targeted_metadata_cache[metadata_key] = value
            return value
        metadata_path = self._root / path / filename
        if not metadata_path.exists():
            self._targeted_metadata_cache[metadata_key] = None
            return None
        if self._reader is not None:
            self._reader.metadata_read_count += 1
        logger.debug("read targeted Zarr metadata file %s", metadata_path)
        value = json.loads(metadata_path.read_text())
        cached_value = value if isinstance(value, Mapping) else None
        self._targeted_metadata_cache[metadata_key] = cached_value
        return cached_value

    def _consolidated_child_names(self, parent_path: str) -> tuple[str, ...]:
        if self._consolidated_children is None:
            return ()
        child_names = self._consolidated_children.get(parent_path, ())
        if not child_names and not self._has_consolidated_metadata(parent_path):
            raise KeyError(parent_path)
        return child_names

    def _has_consolidated_metadata(self, path: str) -> bool:
        path = _normalize_zarr_metadata_path(path)
        if self._consolidated_paths is None:
            return False
        return path in self._consolidated_paths

    def _read_local_metadata_path(self, path: str) -> None:
        if self._metadata is not None or _catalog_has_remote_metadata_client(self):
            return
        for filename in _ZARR_METADATA_FILENAMES:
            self._get_json(path, filename)


def _get_shared_metadata_catalog(
    source: lazynwb.types_.PathLike,
    reader: _ZarrBackendReader | None = None,
) -> _ZarrMetadataCatalog:
    """Build a metadata catalog using the shared consolidated metadata cache."""
    return _ZarrMetadataCatalog(_source_path(source), reader)


def _zarr_metadata_options_key(scope: str, path: str) -> str:
    return json.dumps(
        {
            "format": _ZARR_METADATA_PAYLOAD_FORMAT,
            "scope": scope,
            "path": _normalize_zarr_metadata_path(path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _metadata_payload_from_catalog(
    catalog: _ZarrMetadataCatalog,
) -> dict[str, object] | None:
    exported = catalog.export_metadata_snapshot()
    if exported is None:
        return None
    metadata, origin = exported
    return {
        "kind": "zarr_metadata_catalog",
        "metadata": metadata,
        "origin": origin,
    }


def _metadata_payload_to_zarr_metadata(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    if payload.get("kind") != "zarr_metadata_catalog":
        raise ValueError("unsupported Zarr metadata payload kind")
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("Zarr metadata payload missing metadata mapping")
    metadata = {
        str(key): dict(value)
        for key, value in raw_metadata.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    if not metadata:
        raise ValueError("Zarr metadata payload is empty")
    origin = str(payload.get("origin", "targeted"))
    if origin not in {"consolidated", "targeted"}:
        logger.debug("normalizing unsupported Zarr metadata payload origin %r", origin)
        origin = "targeted"
    return metadata, origin


def _store_shared_consolidated_metadata_payload(
    root: _ZarrPath,
    metadata: Mapping[str, object],
) -> None:
    stat_key = _zmetadata_stat_key(root)
    if stat_key is None:
        logger.debug(
            "skipping shared consolidated Zarr metadata warm for %s: "
            ".zmetadata missing",
            root,
        )
        return
    cache_key = _zarr_metadata_cache_key(root)
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        _ZARR_CONSOLIDATED_METADATA_CACHE[cache_key] = (
            _ZarrConsolidatedMetadataCacheEntry(
                stat_key=stat_key,
                metadata=metadata,
            )
        )
    logger.debug(
        "warmed shared consolidated Zarr metadata cache for %s from "
        "persistent payload with %d entries",
        root,
        len(metadata),
    )


def _clear_shared_metadata_catalog_cache(
    source: lazynwb.types_.PathLike | None = None,
) -> None:
    """Reset Zarr process metadata caches without evicting persistent SQLite."""
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        if source is None:
            metadata_entry_count = len(_ZARR_CONSOLIDATED_METADATA_CACHE)
            remote_payload_count = len(_REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE)
            _ZARR_CONSOLIDATED_METADATA_CACHE.clear()
            _REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE.clear()
        else:
            cache_key = _zarr_metadata_cache_key(_source_path(source))
            removed_metadata = _ZARR_CONSOLIDATED_METADATA_CACHE.pop(cache_key, None)
            remote_cache_key = _remote_consolidated_payload_cache_key(source)
            removed_remote_payload = None
            if remote_cache_key is not None:
                removed_remote_payload = _REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE.pop(
                    remote_cache_key,
                    None,
                )
    if source is None:
        _clear_obstore_store_cache()
        logger.debug(
            "cleared all shared Zarr metadata process cache entries "
            "(metadata=%d remote_payloads=%d); persistent SQLite catalog cache "
            "is unchanged",
            metadata_entry_count,
            remote_payload_count,
        )
        return
    logger.debug(
        "cleared shared Zarr metadata process cache entry for %s "
        "(metadata=%s remote_payload=%s)",
        cache_key,
        removed_metadata is not None,
        removed_remote_payload is not None,
    )


def _clear_obstore_store_cache() -> None:
    """Clear cached obstore stores under the same lock used by store reads/writes."""
    with _OBSTORE_STORE_CACHE_LOCK:
        logger.debug(
            "clearing obstore Zarr store cache with %d entries",
            len(_OBSTORE_STORE_CACHE),
        )
        _OBSTORE_STORE_CACHE.clear()


def _get_shared_consolidated_metadata(
    root: _ZarrPath,
    reader: _ZarrBackendReader | None = None,
) -> Mapping[str, object] | None:
    if reader is not None and reader._remote_metadata_client is not None:
        remote_payload = _get_remote_consolidated_payload(reader)
        if remote_payload is not None:
            reader.used_consolidated_metadata = True
            logger.debug(
                "using obstore Zarr consolidated metadata for %s with %d entries",
                reader._source,
                len(remote_payload.metadata),
            )
            return remote_payload.metadata
        logger.debug(
            "remote Zarr consolidated metadata unavailable for %s; using "
            "non-consolidated obstore metadata path",
            reader._source,
        )
        return None
    cache_key = _zarr_metadata_cache_key(root)
    stat_key = _zmetadata_stat_key(root)
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        cached = _ZARR_CONSOLIDATED_METADATA_CACHE.get(cache_key)
        if cached is not None and cached.stat_key == stat_key:
            if cached.metadata is None:
                logger.debug(
                    "shared Zarr metadata catalog cache hit without consolidated "
                    "metadata for %s",
                    root,
                )
                return None
            if reader is not None:
                reader.used_consolidated_metadata = True
            logger.debug(
                "shared Zarr metadata catalog cache hit for %s with %d entries",
                root,
                len(cached.metadata),
            )
            return cached.metadata

        metadata = _load_consolidated_metadata(root, reader, stat_key)
        _ZARR_CONSOLIDATED_METADATA_CACHE[cache_key] = (
            _ZarrConsolidatedMetadataCacheEntry(
                stat_key=stat_key,
                metadata=metadata,
            )
        )
        return metadata


def _load_consolidated_metadata(
    root: _ZarrPath,
    reader: _ZarrBackendReader | None,
    stat_key: tuple[int, int] | None,
) -> Mapping[str, object] | None:
    zmetadata_path = root / ".zmetadata"
    if stat_key is None:
        logger.debug("Zarr consolidated metadata not found for %s", root)
        return None
    try:
        payload = json.loads(zmetadata_path.read_text())
    except Exception:
        logger.debug(
            "failed to load Zarr consolidated metadata for %s",
            root,
            exc_info=True,
        )
        return None
    if payload.get("zarr_consolidated_format") != 1:
        logger.debug(
            "ignoring unsupported Zarr consolidated metadata format for %s: %r",
            root,
            payload.get("zarr_consolidated_format"),
        )
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        logger.debug("Zarr consolidated metadata for %s has no metadata mapping", root)
        return None
    if reader is not None:
        reader.metadata_read_count += 1
        reader.used_consolidated_metadata = True
    logger.debug(
        "loaded shared Zarr consolidated metadata for %s with %d entries",
        root,
        len(metadata),
    )
    return metadata


def _catalog_has_remote_metadata_client(catalog: _ZarrMetadataCatalog) -> bool:
    return (
        catalog._reader is not None
        and catalog._reader._remote_metadata_client is not None
    )


def _get_remote_non_consolidated_source_identity(
    reader: _ZarrBackendReader,
) -> catalog_models._SourceIdentity | None:
    if reader._remote_metadata_client is None:
        return None
    metadata_reads_before = reader.metadata_read_count
    metadata_bytes_before = reader.metadata_bytes_fetched
    _read_remote_metadata_path(reader, "")
    root_has_metadata = any(
        reader._remote_metadata_json_cache.get(
            _zarr_metadata_object_key("", filename)
        )
        is not None
        for filename in _ZARR_METADATA_FILENAMES
    )
    if not root_has_metadata:
        try:
            root_has_metadata = bool(_remote_child_names(reader, ""))
        except KeyError:
            root_has_metadata = False
    if not root_has_metadata:
        logger.debug(
            "remote non-consolidated Zarr metadata not found for %s",
            reader._source,
        )
        return catalog_models._SourceIdentity(
            source_url=_source_uri(reader._source),
            in_process_token=f"zarr-missing:{id(reader)}",
        )
    logger.debug(
        "built Zarr source identity for %s from remote non-consolidated "
        "metadata (metadata_reads=%d metadata_bytes=%d)",
        reader._source,
        reader.metadata_read_count - metadata_reads_before,
        reader.metadata_bytes_fetched - metadata_bytes_before,
    )
    return catalog_models._SourceIdentity(
        source_url=_source_uri(reader._source),
        in_process_token=f"zarr-non-consolidated:{id(reader)}",
    )


def _remote_path_exists(reader: _ZarrBackendReader | None, path: str) -> bool:
    if reader is None:
        return False
    normalized_path = _normalize_zarr_metadata_path(path)
    _read_remote_metadata_path(reader, normalized_path)
    return any(
        reader._remote_metadata_json_cache.get(
            _zarr_metadata_object_key(normalized_path, filename)
        )
        is not None
        for filename in _ZARR_METADATA_FILENAMES
    )


def _remote_json_metadata(
    reader: _ZarrBackendReader | None,
    path: str,
    filename: str,
) -> Mapping[str, object] | None:
    if reader is None:
        return None
    normalized_path = _normalize_zarr_metadata_path(path)
    _read_remote_metadata_path(reader, normalized_path)
    metadata_key = _zarr_metadata_object_key(normalized_path, filename)
    value = reader._remote_metadata_json_cache.get(metadata_key)
    return value if isinstance(value, Mapping) else None


def _remote_child_names(
    reader: _ZarrBackendReader | None,
    parent_path: str,
) -> tuple[str, ...]:
    if reader is None or reader._remote_metadata_client is None:
        return ()
    normalized_path = _normalize_zarr_metadata_path(parent_path)
    cached = reader._remote_child_names_cache.get(normalized_path)
    if cached is None:
        client = reader._remote_metadata_client
        request_count_before = client.request_count
        bytes_before = client.bytes_fetched
        listing = client.list_child_names_sync(normalized_path)
        reader.metadata_read_count += client.request_count - request_count_before
        reader.metadata_bytes_fetched += client.bytes_fetched - bytes_before
        cached = tuple(sorted(name for name in listing if not name.startswith(".")))
        reader._remote_child_names_cache[normalized_path] = cached
        logger.debug(
            "listed remote non-consolidated Zarr children for %s/%s: %d "
            "(metadata_reads=%d metadata_bytes=%d)",
            reader._source,
            normalized_path or ".",
            len(cached),
            client.request_count - request_count_before,
            client.bytes_fetched - bytes_before,
        )
    if not cached and not _remote_path_exists(reader, normalized_path):
        raise KeyError(parent_path)
    return cached


def _read_remote_metadata_path(reader: _ZarrBackendReader, path: str) -> None:
    client = reader._remote_metadata_client
    if client is None:
        return
    normalized_path = _normalize_zarr_metadata_path(path)
    missing_keys = tuple(
        key
        for key in (
            _zarr_metadata_object_key(normalized_path, filename)
            for filename in _ZARR_METADATA_FILENAMES
        )
        if key not in reader._remote_metadata_json_cache
    )
    if not missing_keys:
        return
    request_count_before = client.request_count
    bytes_before = client.bytes_fetched
    payloads = _run_remote_metadata_coroutine_sync(client.read_many(missing_keys))
    reader.metadata_read_count += client.request_count - request_count_before
    reader.metadata_bytes_fetched += client.bytes_fetched - bytes_before
    for key in missing_keys:
        payload = payloads.get(key)
        if payload is None:
            reader._remote_metadata_json_cache[key] = None
            continue
        try:
            value = json.loads(payload)
        except Exception:
            reader._remote_metadata_json_cache[key] = None
            logger.debug(
                "failed to parse remote non-consolidated Zarr metadata file "
                "%s/%s",
                reader._source,
                key,
                exc_info=True,
            )
            continue
        reader._remote_metadata_json_cache[key] = (
            value if isinstance(value, Mapping) else None
        )
        logger.debug(
            "read remote non-consolidated Zarr metadata file %s/%s (%d bytes)",
            reader._source,
            key,
            len(payload),
        )


def _run_remote_metadata_coroutine_sync(
    coroutine: object,
) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    future = lazynwb.utils.get_threadpool_executor().submit(asyncio.run, coroutine)
    return future.result()


def _zarr_metadata_object_key(path: str, filename: str) -> str:
    normalized_path = _normalize_zarr_metadata_path(path)
    return f"{normalized_path}/{filename}" if normalized_path else filename


class _ObstoreZarrMetadataClient:
    """Private obstore client for remote Zarr metadata objects."""

    def __init__(self, source: lazynwb.types_.PathLike) -> None:
        context = _remote_zarr_store_context(source)
        self._source_url = _source_uri(source)
        self._context = context
        self._store = _cached_obstore_store(context)
        self._chunk_transfer_config = zarr_chunk_transfer._ZarrChunkTransferConfig()
        self.request_count = 0
        self.bytes_fetched = 0
        max_concurrency = zarr_chunk_transfer._max_concurrency(
            self._chunk_transfer_config
        )
        logger.debug(
            "initialized obstore Zarr metadata client for %s as prefix %r "
            "(max_concurrency=%d)",
            self._source_url,
            self._context.root_prefix,
            max_concurrency,
        )

    @property
    def cache_key(self) -> str:
        return _remote_consolidated_payload_cache_key_from_context(self._context)

    def read_consolidated_payload_sync(
        self,
    ) -> _RemoteZarrConsolidatedPayload | None:
        return _run_remote_metadata_coroutine_sync(self._read_consolidated_payload())

    async def _read_consolidated_payload(
        self,
    ) -> _RemoteZarrConsolidatedPayload | None:
        object_key = _remote_zarr_object_key(self._context.root_prefix, ".zmetadata")
        result, payload_bytes = await self._read_obstore_object(object_key)
        payload = json.loads(payload_bytes)
        if not isinstance(payload, Mapping):
            logger.debug("Zarr consolidated metadata payload is not a mapping")
            return None
        if payload.get("zarr_consolidated_format") != 1:
            logger.debug(
                "ignoring unsupported remote Zarr consolidated metadata format "
                "for %s: %r",
                self._source_url,
                payload.get("zarr_consolidated_format"),
            )
            return None
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            logger.debug(
                "remote Zarr consolidated metadata for %s has no metadata mapping",
                self._source_url,
            )
            return None
        source_identity = _source_identity_from_obstore_metadata(
            self._source_url,
            getattr(result, "meta", None),
            content_length=len(payload_bytes),
        )
        return _RemoteZarrConsolidatedPayload(
            source_identity=source_identity,
            metadata=metadata,
            fetched_bytes=len(payload_bytes),
        )

    async def read_object(self, key: str) -> bytes:
        object_key = _remote_zarr_object_key(self._context.root_prefix, key)
        _, payload_bytes = await self._read_obstore_object(object_key)
        return payload_bytes

    async def read_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        if not keys:
            return {}
        started = time.perf_counter()
        max_concurrency = zarr_chunk_transfer._max_concurrency(
            self._chunk_transfer_config
        )
        logger.debug(
            "reading %d Zarr objects from obstore prefix %r with "
            "max_concurrency=%d",
            len(keys),
            self._context.root_prefix,
            max_concurrency,
        )
        payloads = await zarr_chunk_transfer._map_async_bounded(
            keys,
            read_one=self._read_object_or_none,
            max_concurrency=max_concurrency,
        )
        result = {
            key: payload
            for key, payload in zip(keys, payloads, strict=True)
            if payload is not None
        }
        logger.debug(
            "read %d/%d Zarr objects from obstore prefix %r in %.3f s "
            "(max_concurrency=%d)",
            len(result),
            len(keys),
            self._context.root_prefix,
            time.perf_counter() - started,
            max_concurrency,
        )
        return result

    def list_child_names_sync(self, parent_path: str) -> tuple[str, ...]:
        listing = _run_remote_metadata_coroutine_sync(
            self._list_child_names(parent_path)
        )
        return listing.child_names

    async def _list_child_names(self, parent_path: str) -> _RemoteZarrChildListing:
        normalized_path = _normalize_zarr_metadata_path(parent_path)
        prefix = f"{normalized_path}/" if normalized_path else ""
        object_prefix = _remote_zarr_object_key(self._context.root_prefix, prefix)
        started = time.perf_counter()
        result = await obstore.list_with_delimiter_async(
            self._store,
            prefix=object_prefix,
        )
        self.request_count += 1
        child_names = _child_names_from_obstore_delimited_listing(
            result,
            object_prefix=object_prefix,
        )
        logger.debug(
            "listed %d child prefixes from obstore Zarr prefix %s/%s in %.3f s",
            len(child_names),
            self._context.store_url,
            object_prefix,
            time.perf_counter() - started,
        )
        return _RemoteZarrChildListing(child_names=child_names)

    async def _read_object_or_none(self, key: str) -> bytes | None:
        try:
            return await self.read_object(key)
        except (FileNotFoundError, KeyError, obstore.exceptions.NotFoundError):
            logger.debug("remote Zarr object missing: %s/%s", self._source_url, key)
            return None

    async def _read_obstore_object(self, object_key: str) -> tuple[object, bytes]:
        started = time.perf_counter()
        result = await obstore.get_async(self._store, object_key)
        payload_bytes = _bytes_from_obstore_get_result(result)
        self.request_count += 1
        self.bytes_fetched += len(payload_bytes)
        logger.debug(
            "read obstore Zarr object %s/%s in %.3f s (%d bytes)",
            self._context.store_url,
            object_key,
            time.perf_counter() - started,
            len(payload_bytes),
        )
        return result, payload_bytes


def _remote_metadata_client_if_available(
    source: lazynwb.types_.PathLike,
) -> _ObstoreZarrMetadataClient | None:
    try:
        parsed = urllib.parse.urlsplit(_source_uri(source))
    except (TypeError, ValueError):
        logger.debug("cannot parse Zarr source %r for obstore metadata", source)
        return None
    if parsed.scheme not in _OBSTORE_REMOTE_ZARR_SCHEMES:
        return None
    try:
        return _ObstoreZarrMetadataClient(source)
    except Exception:
        logger.debug(
            "failed to initialize obstore Zarr metadata client for %r",
            source,
            exc_info=True,
        )
        return None


def _get_remote_consolidated_payload(
    reader: _ZarrBackendReader,
) -> _RemoteZarrConsolidatedPayload | None:
    client = reader._remote_metadata_client
    if client is None:
        return None
    if reader._remote_consolidated_payload is not None:
        return reader._remote_consolidated_payload
    cache_key = client.cache_key
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        if cache_key in _REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE:
            cached = _REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE[cache_key]
            reader._remote_consolidated_payload = cached
            if cached is not None:
                reader.used_consolidated_metadata = True
                logger.debug(
                    "remote Zarr consolidated metadata cache hit for %s with %d "
                    "entries",
                    reader._source,
                    len(cached.metadata),
                )
            else:
                logger.debug(
                    "remote Zarr consolidated metadata cache hit without metadata "
                    "for %s",
                    reader._source,
                )
            return cached

    request_count_before = client.request_count
    bytes_before = client.bytes_fetched
    try:
        payload = client.read_consolidated_payload_sync()
    except Exception:
        logger.debug(
            "obstore Zarr consolidated metadata read failed for %s; falling back",
            reader._source,
            exc_info=True,
        )
        payload = None
    reader.metadata_read_count += client.request_count - request_count_before
    reader.metadata_bytes_fetched += client.bytes_fetched - bytes_before
    if payload is not None:
        reader.used_consolidated_metadata = True
    reader._remote_consolidated_payload = payload
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        _REMOTE_ZARR_CONSOLIDATED_PAYLOAD_CACHE[cache_key] = payload
    return payload


def _remote_zarr_store_context(
    source: lazynwb.types_.PathLike,
) -> _RemoteZarrStoreContext:
    source_url = _source_uri(source)
    parsed = urllib.parse.urlsplit(source_url)
    if parsed.scheme not in _OBSTORE_REMOTE_ZARR_SCHEMES:
        raise ValueError(f"unsupported remote Zarr obstore scheme: {source_url!r}")
    if not parsed.netloc:
        raise ValueError(f"remote Zarr URL must include a store authority: {source_url!r}")
    root_prefix = parsed.path.lstrip("/").rstrip("/")
    store_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    storage_options = lazynwb._storage_options._get_obstore_storage_options(
        lazynwb._config.config.fsspec_storage_options,
        anon=lazynwb._config.config.anon,
    )
    if parsed.scheme == "s3":
        storage_options = lazynwb._storage_options._add_discovered_s3_region(
            bucket=parsed.netloc,
            storage_options=storage_options,
        )
    return _RemoteZarrStoreContext(
        store_url=store_url,
        root_prefix=root_prefix,
        storage_options=storage_options,
    )


def _cached_obstore_store(
    context: _RemoteZarrStoreContext,
) -> obstore.store.ObjectStore:
    cache_key = _obstore_store_cache_key(context)
    with _OBSTORE_STORE_CACHE_LOCK:
        cached = _OBSTORE_STORE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("obstore Zarr store cache hit for %s", context.store_url)
            return cached
        logger.debug(
            "obstore Zarr store cache miss for %s (storage_options=%s)",
            context.store_url,
            sorted(str(key) for key in context.storage_options),
        )
        store = obstore.store.from_url(context.store_url, **context.storage_options)
        _OBSTORE_STORE_CACHE[cache_key] = store
        return store


def _obstore_store_cache_key(
    context: _RemoteZarrStoreContext,
) -> _ObstoreStoreCacheKey:
    return (
        context.store_url,
        tuple(
            sorted(
                (str(key), repr(value))
                for key, value in context.storage_options.items()
            )
        ),
    )


def _remote_consolidated_payload_cache_key(
    source: lazynwb.types_.PathLike,
) -> str | None:
    try:
        context = _remote_zarr_store_context(source)
    except Exception:
        logger.debug("cannot build remote Zarr cache key for %r", source, exc_info=True)
        return None
    return _remote_consolidated_payload_cache_key_from_context(context)


def _remote_consolidated_payload_cache_key_from_context(
    context: _RemoteZarrStoreContext,
) -> str:
    options_key = tuple(
        sorted((str(key), repr(value)) for key, value in context.storage_options.items())
    )
    return repr((context.store_url, context.root_prefix, options_key))


def _remote_zarr_object_key(root_prefix: str, key: str) -> str:
    clean_key = key.lstrip("/")
    if not root_prefix:
        return clean_key
    return f"{root_prefix}/{clean_key}"


def _child_names_from_obstore_delimited_listing(
    listing: object,
    *,
    object_prefix: str,
) -> tuple[str, ...]:
    listing_mapping = listing if isinstance(listing, Mapping) else {}
    names: set[str] = set()
    for value in listing_mapping.get("common_prefixes", ()):
        child_name = _child_name_from_common_prefix(
            str(value),
            object_prefix=object_prefix,
        )
        if child_name is not None:
            names.add(child_name)
    return tuple(sorted(names))


def _child_name_from_common_prefix(
    common_prefix: str,
    *,
    object_prefix: str,
) -> str | None:
    prefix = object_prefix
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    if prefix and common_prefix.startswith(prefix):
        relative = common_prefix[len(prefix) :]
    else:
        relative = common_prefix.rsplit("/", maxsplit=1)[-1]
    relative = relative.strip("/")
    if not relative:
        return None
    return relative.split("/", maxsplit=1)[0]


def _bytes_from_obstore_get_result(result: object) -> bytes:
    bytes_method = getattr(result, "bytes", None)
    if callable(bytes_method):
        return bytes(bytes_method())
    to_bytes = getattr(result, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(result)


def _source_identity_from_obstore_metadata(
    source_url: str,
    metadata: object,
    *,
    content_length: int,
) -> catalog_models._SourceIdentity:
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    size = metadata_mapping.get("size", content_length)
    return catalog_models._SourceIdentity(
        source_url=source_url,
        content_length=int(size) if size is not None else content_length,
        version_id=_stat_info_str(metadata_mapping, "VersionId", "version"),
        etag=_stat_info_str(metadata_mapping, "ETag", "e_tag"),
        last_modified=_metadata_datetime_iso(metadata_mapping.get("last_modified")),
    )


def _zmetadata_stat_key(root: _ZarrPath) -> tuple[int, int] | None:
    try:
        stat = (root / ".zmetadata").stat()
    except (FileNotFoundError, OSError):
        logger.debug("Zarr consolidated metadata stat unavailable for %s", root)
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _zarr_metadata_cache_key(root: _ZarrPath) -> str:
    try:
        return root.resolve().as_posix()
    except OSError:
        logger.debug("failed to resolve Zarr metadata cache key for %s", root)
        return root.as_posix()


def _normalize_zarr_metadata_path(path: str) -> str:
    normalized = lazynwb.utils.normalize_internal_file_path(path)
    return "" if normalized == "/" else normalized.removeprefix("/")


def _filter_zarr_only_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in attrs.items()
        if str(key) not in _ZARR_ONLY_ATTR_NAMES
    }


def _consolidated_metadata_paths(metadata: Mapping[str, object]) -> frozenset[str]:
    paths: set[str] = set()
    for key in metadata:
        if not isinstance(key, str):
            continue
        path = _zarr_metadata_path_from_key(key)
        if path is not None:
            paths.add(path)
    return frozenset(paths)


def _consolidated_child_index(
    paths: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    children: dict[str, set[str]] = {}
    for path in paths:
        if path == "":
            continue
        parts = path.split("/")
        for index, name in enumerate(parts):
            parent = "/".join(parts[:index])
            children.setdefault(parent, set()).add(name)
    return {parent: tuple(sorted(names)) for parent, names in children.items()}


def _zarr_metadata_path_from_key(key: str) -> str | None:
    for filename in (".zarray", ".zattrs", ".zgroup"):
        if key == filename:
            return ""
        suffix = f"/{filename}"
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _get_table_column_entries(
    catalog: _ZarrMetadataCatalog,
    table_path: str,
) -> tuple[_ZarrEntry, ...]:
    names_to_entries = {
        name: _entry_for_path(catalog, f"{table_path}/{name}", name)
        for name in catalog.child_names(table_path)
    }
    if lazynwb.utils.normalize_internal_file_path(table_path) == "general":
        names_to_entries.update(_get_general_metadata_entries(catalog))
        names_to_entries = {
            name: entry
            for name, entry in names_to_entries.items()
            if not entry.is_group
        }
    _drop_known_reference_entries(names_to_entries)
    return tuple(names_to_entries.values())


def _get_general_metadata_entries(
    catalog: _ZarrMetadataCatalog,
) -> dict[str, _ZarrEntry]:
    names_to_entries: dict[str, _ZarrEntry] = {}
    for path in (
        "session_start_time",
        "session_description",
        "identifier",
        "timestamps_reference_time",
        "file_create_date",
    ):
        entry = _entry_for_path(catalog, path, path)
        if not entry.is_group:
            names_to_entries[path] = entry
    try:
        metadata_children = catalog.child_names("general/metadata")
    except KeyError:
        metadata_children = ()
    for name in metadata_children:
        if name in names_to_entries:
            continue
        entry = _entry_for_path(catalog, f"general/metadata/{name}", name)
        if not entry.is_group:
            names_to_entries[name] = entry
    return names_to_entries


def _entry_for_path(
    catalog: _ZarrMetadataCatalog,
    path: str,
    name: str,
) -> _ZarrEntry:
    return _ZarrEntry(
        name=name,
        path=path,
        zarray=catalog.get_zarray(path),
        zattrs=catalog.get_zattrs(path),
    )


def _drop_known_reference_entries(names_to_entries: dict[str, _ZarrEntry]) -> None:
    known_references = {
        "timeseries": "TimeSeriesReferenceVectorData",
    }
    for name, neurodata_type in known_references.items():
        entry = names_to_entries.get(name)
        if entry is None or entry.zattrs.get("neurodata_type") != neurodata_type:
            continue
        logger.debug(
            "skipping Zarr reference column %r with neurodata_type %r",
            name,
            neurodata_type,
        )
        names_to_entries.pop(name, None)
        names_to_entries.pop(f"{name}_index", None)


def _column_schema_from_zarr_entry(
    entry: _ZarrEntry,
    column_facts: catalog_schema._ColumnFacts,
    source_identity: catalog_models._SourceIdentity,
    table_path: str,
    table_rules: catalog_schema._TableSchemaRules,
) -> catalog_models._TableColumnSchema:
    column_rules = table_rules.column_rules(
        entry.name,
        shape=column_facts.shape,
        ndim=column_facts.ndim,
    )
    shape = column_facts.shape
    ndim = column_facts.ndim
    dataset = catalog_models._DatasetSchema(
        path=entry.path,
        dtype=_neutral_dtype_from_zarr_entry(entry),
        shape=shape,
        ndim=ndim,
        chunks=entry.chunks,
        storage_layout="group" if entry.is_group else "chunked",
        compression=_compression_name(entry),
        compression_opts=catalog_models._to_json_value(_compressor_config(entry)),
        filters_json=tuple(
            catalog_models._to_json_value(item) for item in _filters(entry)
        ),
        fill_value=catalog_models._to_json_value(
            None if entry.zarray is None else entry.zarray.get("fill_value")
        ),
        read_capabilities=_read_capabilities(entry),
        attrs_json=catalog_models._attrs_to_tuple(
            _filter_zarr_only_attrs(entry.zattrs)
        ),
        is_group=entry.is_group,
        is_dataset=not entry.is_group,
    )
    return catalog_models._TableColumnSchema(
        name=entry.name,
        table_path=table_path,
        source_path=source_identity.source_url,
        backend="zarr",
        dataset=dataset,
        is_metadata_table=column_rules.is_metadata_table,
        is_timeseries=column_rules.is_timeseries,
        is_timeseries_with_rate=column_rules.is_timeseries_with_rate,
        is_timeseries_length_aligned=column_rules.is_timeseries_length_aligned,
        is_nominally_indexed=column_rules.is_nominally_indexed,
        is_index_column=column_rules.is_index_column,
        is_multidimensional=column_rules.is_multidimensional,
        index_column_name=column_rules.index_column_name,
        data_column_name=column_rules.data_column_name,
        row_element_shape=column_rules.row_element_shape,
    )


def _column_facts_from_zarr_entries(
    entries: Iterable[_ZarrEntry],
) -> dict[str, catalog_schema._ColumnFacts]:
    return {
        entry.name: catalog_schema._ColumnFacts(
            name=entry.name,
            shape=entry.shape,
            ndim=entry.ndim,
            is_scalar_metadata=entry.ndim == 0 or entry.shape == (1,),
        )
        for entry in entries
    }


def _neutral_dtype_from_zarr_entry(entry: _ZarrEntry) -> catalog_models._NeutralDType:
    dtype = entry.dtype
    if dtype is None:
        return catalog_models._NeutralDType(kind="unknown")
    if dtype.kind == "O" and _is_zarr_string_like(entry):
        return catalog_models._NeutralDType(
            kind="vlen_string",
            numpy_dtype=dtype.str,
            byte_order=dtype.byteorder,
            itemsize=dtype.itemsize,
            detail="zarr-vlen-string",
        )
    return catalog_models._NeutralDType.from_backend_dtype(dtype)


def _is_zarr_string_like(entry: _ZarrEntry) -> bool:
    zarr_dtype = entry.zattrs.get("zarr_dtype")
    if zarr_dtype in {"str", "scalar"}:
        return True
    return any(
        isinstance(item, Mapping) and item.get("id") == "vlen-utf8"
        for item in _filters(entry)
    )


def _compression_name(entry: _ZarrEntry) -> str | None:
    compressor = _compressor_config(entry)
    if isinstance(compressor, Mapping):
        codec_id = compressor.get("id")
        return str(codec_id) if codec_id is not None else None
    return None


def _compressor_config(entry: _ZarrEntry) -> object | None:
    if entry.zarray is None:
        return None
    return entry.zarray.get("compressor")


def _filters(entry: _ZarrEntry) -> tuple[object, ...]:
    if entry.zarray is None:
        return ()
    filters = entry.zarray.get("filters")
    if filters is None:
        return ()
    return tuple(_as_sequence(filters))


def _read_capabilities(entry: _ZarrEntry) -> tuple[str, ...]:
    if entry.is_group:
        return ("metadata", "children")
    capabilities = ["metadata", "shape", "dtype", "slice"]
    if entry.chunks is not None:
        capabilities.append("chunked")
    if _compressor_config(entry) is not None or _filters(entry):
        capabilities.append("filtered")
    return tuple(capabilities)


def _as_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(f"expected sequence, got {type(value).__name__}")


def _mtime_iso(mtime: float) -> str:
    return datetime.datetime.fromtimestamp(
        mtime,
        tz=datetime.timezone.utc,
    ).isoformat()


def _metadata_datetime_iso(value: object) -> str | None:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _stat_info_str(info: object, *keys: str) -> str | None:
    if not isinstance(info, Mapping):
        return None
    for key in keys:
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _summary_path(path: str) -> str:
    if path.startswith("/"):
        return path
    return f"/{path}" if path else "/"


def _source_path(source: lazynwb.types_.PathLike) -> _ZarrPath:
    if isinstance(source, upath.UPath):
        return source
    if isinstance(source, pathlib.Path):
        return source
    raw_source = _source_uri(source)
    parsed = urllib.parse.urlsplit(raw_source)
    if parsed.scheme == "file":
        return pathlib.Path(urllib.parse.unquote(parsed.path))
    if parsed.scheme in _REMOTE_ZARR_SCHEMES:
        return upath.UPath(raw_source, **_get_zarr_storage_options())
    return pathlib.Path(os.fsdecode(source))


def _get_zarr_storage_options() -> dict[str, object]:
    import lazynwb.file_io

    return lazynwb.file_io._get_fsspec_storage_options()


def _source_uri(source: lazynwb.types_.PathLike) -> str:
    as_posix = getattr(source, "as_posix", None)
    if callable(as_posix):
        try:
            return str(as_posix())
        except Exception:
            logger.debug("failed to get as_posix() from source %r", source)
    try:
        return os.fsdecode(source)
    except TypeError:
        return str(source)


def _source_name_has_zarr_suffix(source: lazynwb.types_.PathLike) -> bool:
    return ".zarr" in _source_uri(source).lower()


def _source_supports_native_zarr_array_reads(
    source: lazynwb.types_.PathLike,
) -> bool:
    try:
        parsed = urllib.parse.urlsplit(_source_uri(source))
    except (TypeError, ValueError):
        return False
    return parsed.scheme in _OBSTORE_REMOTE_ZARR_SCHEMES


def _is_fast_zarr_candidate(source: lazynwb.types_.PathLike) -> bool:
    try:
        path = _source_path(source)
    except (TypeError, ValueError):
        logger.debug(
            "skipping fast Zarr catalog candidate check for unsupported path-like "
            "source %r",
            source,
        )
        return False
    try:
        has_metadata = _has_zarr_metadata(path)
        return (path.is_dir() or has_metadata) and (
            path.suffix == ".zarr" or has_metadata
        )
    except Exception:
        logger.debug(
            "fast Zarr catalog candidate check failed for source %r",
            source,
            exc_info=True,
        )
        return False


def _has_zarr_metadata(path: _ZarrPath) -> bool:
    return (path / ".zmetadata").exists() or (path / ".zgroup").exists()


def _default_zarr_backend_reader(
    source: lazynwb.types_.PathLike,
) -> _ZarrBackendReader:
    return _ZarrBackendReader(
        source,
        cache=cache_sqlite._SQLiteSnapshotCache(cache_sqlite._default_cache_path()),
    )
