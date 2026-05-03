from __future__ import annotations

import dataclasses
import logging
import time
import urllib.parse

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.parser as hdf5_parser
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.types_

logger = logging.getLogger(__name__)

_PARSED_METADATA_OPTIONS_KEY = '{"resolve_vlen_attributes":false}'


@dataclasses.dataclass(slots=True)
class _HDF5ParserError(RuntimeError):
    """Structured fail-fast error for migrated HDF5 catalog parsing."""

    source_url: str
    table_path: str
    feature: str
    detail: str
    offset: int | None = None

    def __str__(self) -> str:
        offset = f", offset={self.offset}" if self.offset is not None else ""
        return (
            f"HDF5 catalog parser failed for {self.source_url!r} at "
            f"{self.table_path!r}: {self.feature}: {self.detail}{offset}"
        )


class _NotHDF5Error(_HDF5ParserError):
    """Raised when signature probing rejects a source as non-HDF5."""


@dataclasses.dataclass(slots=True)
class _HDF5TableSchemaScanError(RuntimeError):
    """Structured per-table error from an internal multi-table schema scan."""

    source_url: str
    table_path: str
    feature: str
    detail: str

    def __str__(self) -> str:
        return (
            f"HDF5 table schema scan failed for {self.source_url!r} at "
            f"{self.table_path!r}: {self.feature}: {self.detail}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _HDF5TableSchemaScanResult:
    """Per-table result from a same-source schema scan lifecycle."""

    table_path: str
    snapshot: catalog_models._TableSchemaSnapshot | None
    error: Exception | None
    request_count: int
    fetched_bytes: int
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.snapshot is not None and self.error is None


class _HDF5BackendReader:
    """Fast HDF5 catalog reader backed by byte-range reads."""

    def __init__(
        self,
        source: lazynwb.types_.PathLike,
        range_reader: hdf5_range_reader._RangeReader | None = None,
        cache: cache_sqlite._SQLiteSnapshotCache | None = None,
    ) -> None:
        self._source_url = str(source)
        self._range_reader = range_reader or hdf5_range_reader._ObstoreRangeReader(
            self._source_url
        )
        self._cache = cache
        self._source_identity: catalog_models._SourceIdentity | None = None
        self._scanner: hdf5_parser._HDF5MetadataScanner | None = None
        self._parsed_metadata_loaded = False

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        if self._source_identity is None:
            self._source_identity = await self._range_reader.get_source_identity()
        return self._source_identity

    async def _read_table_schema_snapshots(
        self,
        exact_table_paths: tuple[str, ...],
    ) -> dict[str, _HDF5TableSchemaScanResult]:
        normalized_paths = tuple(dict.fromkeys(exact_table_paths))
        for exact_table_path in normalized_paths:
            catalog_backend._require_exact_normalized_path(exact_table_path)
        logger.debug(
            "reading %d HDF5 table schema snapshots for %s in one reader lifecycle",
            len(normalized_paths),
            self._source_url,
        )
        results: dict[str, _HDF5TableSchemaScanResult] = {}
        for exact_table_path in normalized_paths:
            request_count_before = _range_reader_request_count(self._range_reader)
            fetched_bytes_before = _range_reader_fetched_bytes(self._range_reader)
            started = time.perf_counter()
            snapshot: catalog_models._TableSchemaSnapshot | None = None
            error: Exception | None = None
            try:
                snapshot = await self.read_table_schema_snapshot(exact_table_path)
            except KeyError as exc:
                error = _HDF5TableSchemaScanError(
                    source_url=self._source_url,
                    table_path=exact_table_path,
                    feature="missing_table",
                    detail=repr(exc),
                )
            except _HDF5ParserError as exc:
                error = exc
            except Exception as exc:
                error = _HDF5TableSchemaScanError(
                    source_url=self._source_url,
                    table_path=exact_table_path,
                    feature="unexpected_error",
                    detail=repr(exc),
                )
            request_count = (
                _range_reader_request_count(self._range_reader) - request_count_before
            )
            fetched_bytes = (
                _range_reader_fetched_bytes(self._range_reader) - fetched_bytes_before
            )
            result = _HDF5TableSchemaScanResult(
                table_path=exact_table_path,
                snapshot=snapshot,
                error=error,
                request_count=request_count,
                fetched_bytes=fetched_bytes,
                elapsed_seconds=time.perf_counter() - started,
            )
            results[exact_table_path] = result
            if error is None:
                logger.debug(
                    "same-source HDF5 schema scan built %s/%s "
                    "(requests=%d bytes=%d)",
                    self._source_url,
                    exact_table_path,
                    request_count,
                    fetched_bytes,
                )
            else:
                logger.debug(
                    "same-source HDF5 schema scan error for %s/%s: %r",
                    self._source_url,
                    exact_table_path,
                    error,
                )
        return results

    async def read_table_schema_snapshot(  # noqa: C901
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog_backend._require_exact_normalized_path(exact_table_path)
        started = time.perf_counter()
        phase_started = started
        source_identity = await self.get_source_identity()
        identity_seconds = time.perf_counter() - phase_started
        if self._cache is not None:
            phase_started = time.perf_counter()
            cached = await self._cache.get_table_schema_snapshot(
                source_identity,
                exact_table_path,
            )
            if cached.hit and cached.snapshot is not None:
                logger.debug(
                    "loaded HDF5 table schema snapshot for %s/%s from cache",
                    source_identity.source_url,
                    exact_table_path,
                )
                return cached.snapshot
            schema_cache_seconds = time.perf_counter() - phase_started
        else:
            schema_cache_seconds = 0.0
        if source_identity.content_length is None:
            raise _HDF5ParserError(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="source_identity",
                detail="content length is required for range-backed HDF5 parsing",
            )
        scanner = self._get_scanner(int(source_identity.content_length))
        parsed_cache_seconds = 0.0
        if self._cache is not None and not self._parsed_metadata_loaded:
            phase_started = time.perf_counter()
            parsed_lookup = await self._cache.get_parsed_hdf5_metadata(
                source_identity,
                payload_version=hdf5_parser._PARSED_METADATA_PAYLOAD_VERSION,
                options_key=_PARSED_METADATA_OPTIONS_KEY,
            )
            parsed_cache_seconds = time.perf_counter() - phase_started
            if parsed_lookup.payload is not None:
                scanner.import_metadata(parsed_lookup.payload)
            logger.debug(
                "parsed HDF5 metadata cache lookup for %s: %s",
                source_identity.source_url,
                parsed_lookup.reason,
            )
            self._parsed_metadata_loaded = True
        phase_started = time.perf_counter()
        try:
            column_schemas = await scanner.read_table_column_schemas(
                exact_table_path,
                source_identity,
            )
        except hdf5_parser._TableNotFoundError as exc:
            raise KeyError(exact_table_path) from exc
        except ValueError as exc:
            if scanner.is_hdf5 is False or "no HDF5 superblock" in str(exc):
                raise _NotHDF5Error(
                    source_url=source_identity.source_url,
                    table_path=exact_table_path,
                    feature="hdf5_signature",
                    detail=str(exc),
                ) from exc
            raise _HDF5ParserError(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="hdf5_metadata_parser",
                detail=repr(exc),
            ) from exc
        except Exception as exc:
            raise _HDF5ParserError(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="hdf5_metadata_parser",
                detail=repr(exc),
            ) from exc
        await scanner.warm_table_metadata(_followup_table_paths(exact_table_path))
        parse_seconds = time.perf_counter() - phase_started
        table_length = _get_table_length_from_columns(column_schemas)
        snapshot = catalog_models._TableSchemaSnapshot(
            source_identity=source_identity,
            table_path=exact_table_path,
            backend="hdf5",
            columns=column_schemas,
            table_length=table_length,
        )
        cache_write_seconds = 0.0
        if self._cache is not None:
            phase_started = time.perf_counter()
            await self._cache.put_table_schema_snapshot(snapshot)
            await self._cache.put_parsed_hdf5_metadata(
                source_identity,
                payload_version=hdf5_parser._PARSED_METADATA_PAYLOAD_VERSION,
                options_key=_PARSED_METADATA_OPTIONS_KEY,
                payload=scanner.export_metadata(),
            )
            cache_write_seconds = time.perf_counter() - phase_started
        logger.debug(
            "built HDF5 table schema snapshot for %s/%s with %d columns in %.2f s "
            "(identity=%.3fs schema_cache=%.3fs parsed_cache=%.3fs parse=%.3fs "
            "cache_write=%.3fs requests=%s bytes=%s)",
            source_identity.source_url,
            exact_table_path,
            len(snapshot.columns),
            time.perf_counter() - started,
            identity_seconds,
            schema_cache_seconds,
            parsed_cache_seconds,
            parse_seconds,
            cache_write_seconds,
            getattr(self._range_reader, "request_count", "?"),
            getattr(self._range_reader, "bytes_fetched", "?"),
        )
        return snapshot

    async def close(self) -> None:
        logger.debug("closing HDF5 backend reader for %s", self._source_url)

    def _get_scanner(
        self,
        content_length: int,
    ) -> hdf5_parser._HDF5MetadataScanner:
        if self._scanner is None:
            self._scanner = hdf5_parser._HDF5MetadataScanner(
                self._source_url,
                self._range_reader,
                content_length=content_length,
            )
        return self._scanner


def _get_table_length_from_columns(
    columns: tuple[catalog_models._TableColumnSchema, ...],
) -> int:
    columns_by_name = {column.name: column for column in columns}
    if columns and all(column.is_metadata_table for column in columns):
        logger.debug(
            "table length for %s/%s resolved as one metadata row",
            columns[0].source_path,
            columns[0].table_path,
        )
        return 1
    for column in columns:
        if column.is_nominally_indexed:
            index_column_name = column.index_column_name or column.name
            index_column = columns_by_name.get(index_column_name)
            if index_column is not None and index_column.shape is not None:
                logger.debug(
                    "table length for %s/%s resolved from indexed column %r: %d",
                    column.source_path,
                    column.table_path,
                    index_column.name,
                    index_column.shape[0],
                )
                return index_column.shape[0]
        if column.ndim == 1 and column.shape is not None:
            logger.debug(
                "table length for %s/%s resolved from regular column %r: %d",
                column.source_path,
                column.table_path,
                column.name,
                column.shape[0],
            )
            return column.shape[0]
        if column.ndim == 0:
            logger.debug(
                "table length for %s/%s resolved as metadata row",
                column.source_path,
                column.table_path,
            )
            return 1
    for column in columns:
        if column.shape:
            logger.debug(
                "table length for %s/%s resolved from multidimensional column %r: %d",
                column.source_path,
                column.table_path,
                column.name,
                column.shape[0],
            )
            return column.shape[0]
    raise lazynwb.exceptions.InternalPathError("Could not determine table length")


def _range_reader_request_count(
    range_reader: hdf5_range_reader._RangeReader,
) -> int:
    return int(getattr(range_reader, "request_count", 0))


def _range_reader_fetched_bytes(
    range_reader: hdf5_range_reader._RangeReader,
) -> int:
    return int(getattr(range_reader, "bytes_fetched", 0))


def _followup_table_paths(exact_table_path: str) -> tuple[str, ...]:
    if exact_table_path == "intervals/trials":
        return ("units",)
    if exact_table_path == "units":
        return ("intervals/trials",)
    return ()


def _is_fast_hdf5_candidate(source: lazynwb.types_.PathLike) -> bool:
    url = str(source)
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme in {"file", "http", "https", "s3", "gs", "gcs", "az", "abfs"}


def _default_hdf5_backend_reader(
    source: lazynwb.types_.PathLike,
) -> _HDF5BackendReader:
    return _HDF5BackendReader(
        source,
        range_reader=hdf5_range_reader._ObstoreRangeReader(
            str(source),
            config=hdf5_range_reader._RangeReaderConfig(
                storage_options=lazynwb.file_io._get_obstore_storage_options()
            ),
        ),
        cache=cache_sqlite._SQLiteSnapshotCache(cache_sqlite._default_cache_path()),
    )
