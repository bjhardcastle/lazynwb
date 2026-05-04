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
from collections.abc import Iterable, Mapping

import numpy as np

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog._schema as catalog_schema
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)

_ZARR_ONLY_ATTR_NAMES = frozenset({"_ARRAY_DIMENSIONS", "zarr_dtype"})
_ZARR_CONSOLIDATED_METADATA_CACHE_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrConsolidatedMetadataCacheEntry:
    stat_key: tuple[int, int] | None
    metadata: Mapping[str, object] | None


_ZARR_CONSOLIDATED_METADATA_CACHE: dict[
    str,
    _ZarrConsolidatedMetadataCacheEntry,
] = {}


class _ZarrBackendReader:
    """Explicit Zarr catalog reader for table schema snapshots."""

    def __init__(
        self,
        source: lazynwb.types_.PathLike,
        cache: cache_sqlite._SQLiteSnapshotCache | None = None,
    ) -> None:
        self._source = source
        self._source_path = _local_source_path(source)
        self._cache = cache
        self._source_identity: catalog_models._SourceIdentity | None = None
        self.metadata_read_count = 0
        self.used_consolidated_metadata = False

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        if self._source_identity is None:
            self._source_identity = await asyncio.to_thread(self._build_source_identity)
        return self._source_identity

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog_backend._require_exact_normalized_path(exact_table_path)
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
        snapshot = await asyncio.to_thread(
            self._read_table_schema_snapshot_sync,
            exact_table_path,
            source_identity,
        )
        if self._cache is not None:
            await self._cache.put_table_schema_snapshot(snapshot)
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
        summary = await asyncio.to_thread(self._read_path_summary_sync)
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

    async def close(self) -> None:
        logger.debug("closing Zarr backend reader for %s", self._source)

    def _build_source_identity(self) -> catalog_models._SourceIdentity:
        if not self._source_path.exists():
            return catalog_models._SourceIdentity(
                source_url=str(self._source),
                in_process_token=f"zarr-missing:{id(self)}",
            )
        metadata_path = self._source_path / ".zmetadata"
        if metadata_path.exists():
            stat = metadata_path.stat()
            return catalog_models._SourceIdentity(
                source_url=self._source_path.as_posix(),
                content_length=stat.st_size,
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
        catalog = _ZarrMetadataCatalog(self._source_path, self)
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
        catalog = _ZarrMetadataCatalog(self._source_path, self)
        return catalog.read_path_summary()


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
        root: pathlib.Path,
        reader: _ZarrBackendReader | None = None,
    ) -> None:
        self._root = root
        self._reader = reader
        self._metadata = _get_shared_consolidated_metadata(root, reader)
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
        return self._metadata is not None

    def get_zarray(self, path: str) -> Mapping[str, object] | None:
        return self._get_json(_normalize_zarr_metadata_path(path), ".zarray")

    def get_zattrs(self, path: str) -> Mapping[str, object]:
        return self._get_json(_normalize_zarr_metadata_path(path), ".zattrs") or {}

    def path_exists(self, path: str) -> bool:
        normalized_path = _normalize_zarr_metadata_path(path)
        if normalized_path == "":
            return self._root.exists()
        if self._metadata is not None:
            return self._has_consolidated_metadata(normalized_path)
        root = self._root / normalized_path
        return any(
            (root / filename).exists() for filename in (".zarray", ".zattrs", ".zgroup")
        )

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

    def child_names(self, parent_path: str) -> tuple[str, ...]:
        parent_path = _normalize_zarr_metadata_path(parent_path)
        if self._metadata is not None:
            names = self._consolidated_child_names(parent_path)
        else:
            parent = self._root / parent_path
            if not parent.exists():
                raise KeyError(parent_path)
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
        metadata_path = self._root / path / filename
        if not metadata_path.exists():
            return None
        if self._reader is not None:
            self._reader.metadata_read_count += 1
        logger.debug("read targeted Zarr metadata file %s", metadata_path)
        value = json.loads(metadata_path.read_text())
        return value if isinstance(value, Mapping) else None

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


def _get_shared_metadata_catalog(
    source: lazynwb.types_.PathLike,
    reader: _ZarrBackendReader | None = None,
) -> _ZarrMetadataCatalog:
    """Build a metadata catalog using the shared consolidated metadata cache."""
    return _ZarrMetadataCatalog(_local_source_path(source), reader)


def _clear_shared_metadata_catalog_cache(
    source: lazynwb.types_.PathLike | None = None,
) -> None:
    with _ZARR_CONSOLIDATED_METADATA_CACHE_LOCK:
        if source is None:
            _ZARR_CONSOLIDATED_METADATA_CACHE.clear()
            logger.debug("cleared all shared Zarr metadata catalog cache entries")
            return
        cache_key = _zarr_metadata_cache_key(_local_source_path(source))
        _ZARR_CONSOLIDATED_METADATA_CACHE.pop(cache_key, None)
        logger.debug(
            "cleared shared Zarr metadata catalog cache entry for %s", cache_key
        )


def _get_shared_consolidated_metadata(
    root: pathlib.Path,
    reader: _ZarrBackendReader | None = None,
) -> Mapping[str, object] | None:
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
    root: pathlib.Path,
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


def _zmetadata_stat_key(root: pathlib.Path) -> tuple[int, int] | None:
    try:
        stat = (root / ".zmetadata").stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _zarr_metadata_cache_key(root: pathlib.Path) -> str:
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


def _summary_path(path: str) -> str:
    if path.startswith("/"):
        return path
    return f"/{path}" if path else "/"


def _local_source_path(source: lazynwb.types_.PathLike) -> pathlib.Path:
    raw_source = _source_uri(source)
    parsed = urllib.parse.urlsplit(raw_source)
    if parsed.scheme == "file":
        return pathlib.Path(urllib.parse.unquote(parsed.path))
    return pathlib.Path(os.fsdecode(source))


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


def _is_fast_zarr_candidate(source: lazynwb.types_.PathLike) -> bool:
    protocol = getattr(source, "protocol", None)
    parsed = urllib.parse.urlsplit(_source_uri(source))
    if protocol not in (None, "", "file") and parsed.scheme != "file":
        logger.debug(
            "skipping fast Zarr catalog candidate check for non-local source %r "
            "(protocol=%r)",
            source,
            protocol,
        )
        return False
    try:
        path = _local_source_path(source)
    except TypeError:
        logger.debug(
            "skipping fast Zarr catalog candidate check for unsupported path-like "
            "source %r",
            source,
        )
        return False
    return path.is_dir() and (
        path.suffix == ".zarr"
        or (path / ".zmetadata").exists()
        or (path / ".zgroup").exists()
    )


def _default_zarr_backend_reader(
    source: lazynwb.types_.PathLike,
) -> _ZarrBackendReader:
    return _ZarrBackendReader(
        source,
        cache=cache_sqlite._SQLiteSnapshotCache(cache_sqlite._default_cache_path()),
    )
