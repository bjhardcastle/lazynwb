from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import typing
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence

import lazynwb._catalog.models as catalog_models
import lazynwb.types_

logger = logging.getLogger(__name__)

_FastCatalogBackendName = typing.Literal["hdf5", "zarr"]


@typing.runtime_checkable
class _BackendReader(typing.Protocol):
    """Async private reader interface for exact-path catalog access."""

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        """Return storage identity facts for this reader's source."""

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        """Return catalog facts for one exact, normalized table path."""

    async def close(self) -> None:
        """Release backend resources, if any."""


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaScanMetrics:
    """Normalized private transfer counters for one table schema scan."""

    request_count: int = 0
    fetched_bytes: int = 0


class _TableSchemaScanResult(typing.Protocol):
    """Backend-neutral shape of one same-source schema scan result."""

    table_path: str
    snapshot: catalog_models._TableSchemaSnapshot | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        """Whether the scan produced a usable table snapshot."""

    @property
    def _scan_metrics(self) -> _TableSchemaScanMetrics:
        """Return normalized transfer counters for this scan result."""


@typing.runtime_checkable
class _PathSummaryBackendReader(_BackendReader, typing.Protocol):
    """Reader seam for accessor-free path discovery."""

    async def read_path_summary(
        self,
    ) -> tuple[catalog_models._PathSummaryEntry, ...]:
        """Return accessor-free facts for internal paths in this source."""


@typing.runtime_checkable
class _BatchTableSchemaBackendReader(_BackendReader, typing.Protocol):
    """Reader seam for same-source batched table schema snapshots."""

    async def _read_table_schema_snapshots(
        self,
        exact_table_paths: tuple[str, ...],
    ) -> Mapping[str, _TableSchemaScanResult]:
        """Return per-table scan results for exact, normalized paths."""


@typing.runtime_checkable
class _ArraySelectionBackendReader(_BackendReader, typing.Protocol):
    """Reader seam for native array selections."""

    async def read_array_selection(
        self,
        exact_array_path: str,
        selection: object = None,
    ) -> object:
        """Read one exact array path selection."""


@typing.runtime_checkable
class _BatchArraySelectionBackendReader(
    _ArraySelectionBackendReader,
    typing.Protocol,
):
    """Reader seam for native batched array selections from one array path."""

    async def _read_array_selections(
        self,
        exact_array_path: str,
        selections: Sequence[object],
    ) -> tuple[object, ...]:
        """Read several selections from one exact array path."""


@dataclasses.dataclass(frozen=True, slots=True)
class _FastBackendReaderBinding:
    """Concrete fast backend reader plus facts needed by callers."""

    backend_name: _FastCatalogBackendName
    source: lazynwb.types_.PathLike
    reader: _BackendReader

    @property
    def backend_label(self) -> str:
        if self.backend_name == "hdf5":
            return "HDF5"
        return "Zarr"


@dataclasses.dataclass(frozen=True, slots=True)
class _PathSummaryRead:
    """Completed path summary read through a fast backend reader."""

    backend_name: _FastCatalogBackendName
    source: lazynwb.types_.PathLike
    entries: tuple[catalog_models._PathSummaryEntry, ...]

    @property
    def backend_label(self) -> str:
        if self.backend_name == "hdf5":
            return "HDF5"
        return "Zarr"


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaBatchRead:
    """Completed same-source schema batch through a fast backend reader."""

    backend_name: _FastCatalogBackendName
    source: lazynwb.types_.PathLike
    results: Mapping[str, _TableSchemaScanResult]
    request_count_delta: int
    fetched_bytes_delta: int

    @property
    def backend_label(self) -> str:
        if self.backend_name == "hdf5":
            return "HDF5"
        return "Zarr"

    @property
    def request_count_label(self) -> str:
        return "requests"


def _require_exact_normalized_path(exact_table_path: str) -> None:
    if exact_table_path.startswith("/") or exact_table_path in {"", "."}:
        raise ValueError(
            "backend readers require exact normalized internal paths without a leading slash"
        )


def _fast_catalog_backend_order(
    source: lazynwb.types_.PathLike,
) -> tuple[_FastCatalogBackendName, _FastCatalogBackendName]:
    """Return the preferred fast catalog backend probe order for a source."""

    import lazynwb._zarr.reader as zarr_reader

    if zarr_reader._source_name_has_zarr_suffix(source):
        return ("zarr", "hdf5")
    return ("hdf5", "zarr")


def _fallback_backend_order_after_rejection(
    backend_name: _FastCatalogBackendName,
) -> tuple[_FastCatalogBackendName]:
    """Return the remaining fast backend order after one backend rejects a source."""

    if backend_name == "hdf5":
        return ("zarr",)
    return ("hdf5",)


async def _read_path_summary_if_available(
    source: lazynwb.types_.PathLike,
) -> _PathSummaryRead | None:
    """Try fast backends for an accessor-free path summary and close readers."""

    backend_order = _fast_catalog_backend_order(source)
    logger.debug(
        "using catalog path summary backend order for %r: %s",
        source,
        " -> ".join(backend_order),
    )
    for backend_name in backend_order:
        binding = _fast_backend_reader_if_available(
            source,
            backend_name,
            allow_local_hdf5_file=True,
        )
        if binding is None:
            continue
        try:
            if not isinstance(binding.reader, _PathSummaryBackendReader):
                logger.debug(
                    "%s backend reader for %r lacks path summary seam",
                    binding.backend_label,
                    binding.source,
                )
                continue
            entries = await binding.reader.read_path_summary()
            logger.debug(
                "read %s catalog path summary for %r (entries=%d)",
                binding.backend_label,
                source,
                len(entries),
            )
            return _PathSummaryRead(
                backend_name=binding.backend_name,
                source=binding.source,
                entries=entries,
            )
        except Exception as exc:
            if _is_hdf5_signature_rejection(binding.backend_name, exc):
                logger.debug("catalog path summary rejected non-HDF5 source %r", source)
                continue
            logger.debug(
                "%s catalog path summary unavailable for %r: %r",
                binding.backend_label,
                source,
                exc,
            )
        finally:
            await _close_reader_suppressing_errors(binding)
    return None


async def _read_table_schema_snapshots_if_available(
    source: lazynwb.types_.PathLike,
    exact_table_paths: Iterable[str],
    *,
    allow_local_hdf5_file: bool = False,
) -> _TableSchemaBatchRead | None:
    """Try fast backends for same-source schema snapshots and close readers."""

    normalized_paths = tuple(dict.fromkeys(exact_table_paths))
    for exact_table_path in normalized_paths:
        _require_exact_normalized_path(exact_table_path)
    if not normalized_paths:
        return None
    backend_order = _fast_catalog_backend_order(source)
    logger.debug(
        "using catalog schema batch backend order for %r: %s",
        source,
        " -> ".join(backend_order),
    )
    for backend_name in backend_order:
        binding = _fast_backend_reader_if_available(
            source,
            backend_name,
            allow_local_hdf5_file=allow_local_hdf5_file,
        )
        if binding is None:
            continue
        counters_before = _reader_transfer_counters(binding)
        try:
            if not isinstance(binding.reader, _BatchTableSchemaBackendReader):
                logger.debug(
                    "%s backend reader for %r lacks schema batch seam",
                    binding.backend_label,
                    binding.source,
                )
                continue
            results = await binding.reader._read_table_schema_snapshots(
                normalized_paths
            )
        except Exception as exc:
            if _is_hdf5_signature_rejection(binding.backend_name, exc):
                logger.debug(
                    "%s schema batch skipped non-HDF5 source %r",
                    binding.backend_label,
                    source,
                )
                continue
            logger.debug(
                "%s schema batch failed for %r: %r",
                binding.backend_label,
                source,
                exc,
            )
            continue
        finally:
            await _close_reader_suppressing_errors(binding)
        counters_after = _reader_transfer_counters(binding)
        return _TableSchemaBatchRead(
            backend_name=binding.backend_name,
            source=binding.source,
            results=results,
            request_count_delta=max(0, counters_after[0] - counters_before[0]),
            fetched_bytes_delta=max(0, counters_after[1] - counters_before[1]),
        )
    return None


async def _read_table_schema_snapshot_if_available(
    source: lazynwb.types_.PathLike,
    exact_table_path: str,
    *,
    backend_order: Sequence[_FastCatalogBackendName] | None = None,
    allow_local_hdf5_file: bool = False,
) -> catalog_models._TableSchemaSnapshot | None:
    """Try fast backends for one table schema snapshot and close readers."""

    _require_exact_normalized_path(exact_table_path)
    ordered_backends = tuple(backend_order or _fast_catalog_backend_order(source))
    logger.debug(
        "using catalog schema snapshot backend order for %r: %s",
        source,
        " -> ".join(ordered_backends),
    )
    for backend_name in ordered_backends:
        binding = _fast_backend_reader_if_available(
            source,
            backend_name,
            allow_local_hdf5_file=allow_local_hdf5_file,
        )
        if binding is None:
            continue
        try:
            return await binding.reader.read_table_schema_snapshot(exact_table_path)
        except Exception as exc:
            if _is_unavailable_hdf5_reader_error(binding.backend_name, exc):
                logger.debug(
                    "fast HDF5 backend rejected source %r: %r",
                    binding.source,
                    exc,
                )
                continue
            raise
        finally:
            await _close_reader_suppressing_errors(binding)
    return None


def _schema_scan_result_count_detail(
    backend_name: _FastCatalogBackendName,
    result: _TableSchemaScanResult,
) -> str:
    """Return compact normalized per-result counter text for debug logs."""

    del backend_name
    metrics = result._scan_metrics
    return f"requests={metrics.request_count} bytes={metrics.fetched_bytes}"


def _fast_backend_reader_if_available(
    source: lazynwb.types_.PathLike,
    backend_name: _FastCatalogBackendName,
    *,
    allow_local_hdf5_file: bool = False,
) -> _FastBackendReaderBinding | None:
    if backend_name == "hdf5":
        return _fast_hdf5_backend_reader_if_available(
            source,
            allow_local_file=allow_local_hdf5_file,
        )
    return _fast_zarr_backend_reader_if_available(source)


def _fast_hdf5_backend_reader_if_available(
    source: lazynwb.types_.PathLike,
    *,
    allow_local_file: bool,
) -> _FastBackendReaderBinding | None:
    import lazynwb._hdf5.reader as hdf5_reader

    hdf5_source: lazynwb.types_.PathLike | None
    hdf5_source = _fast_hdf5_source_if_available(
        source,
        allow_local_file=allow_local_file,
    )
    if hdf5_source is None or not hdf5_reader._is_fast_hdf5_candidate(hdf5_source):
        return None
    return _FastBackendReaderBinding(
        backend_name="hdf5",
        source=hdf5_source,
        reader=hdf5_reader._default_hdf5_backend_reader(hdf5_source),
    )


def _fast_backend_is_available(
    source: lazynwb.types_.PathLike,
    backend_name: _FastCatalogBackendName,
    *,
    allow_local_hdf5_file: bool = False,
) -> bool:
    if backend_name == "hdf5":
        return (
            _fast_hdf5_source_if_available(
                source,
                allow_local_file=allow_local_hdf5_file,
            )
            is not None
        )
    import lazynwb._zarr.reader as zarr_reader

    return zarr_reader._is_fast_zarr_candidate(source)


def _fast_hdf5_source_if_available(
    source: lazynwb.types_.PathLike,
    *,
    allow_local_file: bool,
) -> lazynwb.types_.PathLike | None:
    import lazynwb._hdf5.reader as hdf5_reader

    if hdf5_reader._is_fast_hdf5_candidate(source):
        return source
    if allow_local_file:
        return _local_hdf5_file_uri_if_available(source)
    return None


def _fast_zarr_backend_reader_if_available(
    source: lazynwb.types_.PathLike,
) -> _FastBackendReaderBinding | None:
    import lazynwb._zarr.reader as zarr_reader

    if not zarr_reader._is_fast_zarr_candidate(source):
        return None
    return _FastBackendReaderBinding(
        backend_name="zarr",
        source=source,
        reader=zarr_reader._default_zarr_backend_reader(source),
    )


def _local_hdf5_file_uri_if_available(
    source: lazynwb.types_.PathLike,
) -> str | None:
    raw_path = _pathlike_to_string(source)
    parsed = urllib.parse.urlsplit(raw_path)
    if parsed.scheme in {"file", "http", "https", "s3", "gs", "gcs", "az", "abfs"}:
        return raw_path
    if parsed.scheme and not _has_windows_drive_scheme(raw_path, parsed):
        logger.debug(
            "HDF5 catalog reader skipped unsupported source scheme %r for %r",
            parsed.scheme,
            source,
        )
        return None
    try:
        local_path = pathlib.Path(os.fsdecode(source)).expanduser()
    except TypeError:
        logger.debug(
            "HDF5 catalog reader skipped unsupported path-like source %r",
            source,
        )
        return None
    if not local_path.is_file():
        return None
    return local_path.resolve().as_uri()


def _has_windows_drive_scheme(
    raw_path: str,
    parsed: urllib.parse.SplitResult,
) -> bool:
    return len(parsed.scheme) == 1 and len(raw_path) >= 2 and raw_path[1] == ":"


def _pathlike_to_string(path: lazynwb.types_.PathLike) -> str:
    as_posix = getattr(path, "as_posix", None)
    if callable(as_posix):
        try:
            return str(as_posix())
        except Exception:
            logger.debug("failed to get as_posix() from path %r", path)
    try:
        return os.fsdecode(path)
    except TypeError:
        return str(path)


def _reader_transfer_counters(binding: _FastBackendReaderBinding) -> tuple[int, int]:
    reader = binding.reader
    if binding.backend_name == "hdf5":
        range_reader = getattr(reader, "_range_reader", None)
        return (
            int(getattr(range_reader, "request_count", 0)),
            int(getattr(range_reader, "bytes_fetched", 0)),
        )
    return (
        int(getattr(reader, "metadata_read_count", 0)),
        int(getattr(reader, "metadata_bytes_fetched", 0)),
    )


def _is_hdf5_signature_rejection(
    backend_name: _FastCatalogBackendName,
    exc: Exception,
) -> bool:
    if backend_name != "hdf5":
        return False
    import lazynwb._hdf5.reader as hdf5_reader

    return isinstance(exc, hdf5_reader._NotHDF5Error)


def _is_unavailable_hdf5_reader_error(
    backend_name: _FastCatalogBackendName,
    exc: Exception,
) -> bool:
    return _is_hdf5_signature_rejection(backend_name, exc) or (
        backend_name == "hdf5" and isinstance(exc, FileNotFoundError)
    )


async def _close_reader_suppressing_errors(binding: _FastBackendReaderBinding) -> None:
    try:
        await binding.reader.close()
    except Exception as exc:
        logger.debug(
            "error closing %s backend reader for %r: %r",
            binding.backend_label,
            binding.source,
            exc,
        )
