from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import time
import typing
import urllib.parse

import h5py

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb.file_io
import lazynwb.table_metadata
import lazynwb.types_

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
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

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        if self._source_identity is None:
            self._source_identity = await self._range_reader.get_source_identity()
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
                    "loaded HDF5 table schema snapshot for %s/%s from cache",
                    source_identity.source_url,
                    exact_table_path,
                )
                return cached.snapshot
        probe = await hdf5_range_reader._probe_hdf5_signature(self._range_reader)
        if not probe.is_hdf5:
            raise _NotHDF5Error(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="hdf5_signature",
                detail=f"checked offsets {probe.checked_offsets}",
            )
        if source_identity.content_length is None:
            raise _HDF5ParserError(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="source_identity",
                detail="content length is required for range-backed HDF5 parsing",
                offset=probe.signature_offset,
            )
        snapshot = await asyncio.to_thread(
            self._read_table_schema_snapshot_sync,
            exact_table_path,
            source_identity,
            int(source_identity.content_length),
        )
        if self._cache is not None:
            await self._cache.put_table_schema_snapshot(snapshot)
        logger.debug(
            "built HDF5 table schema snapshot for %s/%s with %d columns in %.2f s "
            "(requests=%s bytes=%s)",
            source_identity.source_url,
            exact_table_path,
            len(snapshot.columns),
            time.perf_counter() - t0,
            getattr(self._range_reader, "request_count", "?"),
            getattr(self._range_reader, "bytes_fetched", "?"),
        )
        return snapshot

    async def close(self) -> None:
        logger.debug("closing HDF5 backend reader for %s", self._source_url)

    def _read_table_schema_snapshot_sync(
        self,
        exact_table_path: str,
        source_identity: catalog_models._SourceIdentity,
        content_length: int,
    ) -> catalog_models._TableSchemaSnapshot:
        file_obj = _RangeReaderFile(self._range_reader, size=content_length)
        try:
            with h5py.File(file_obj, mode="r") as h5_file:
                adapter = _HDF5AccessorAdapter(
                    h5_file=h5_file,
                    source_url=source_identity.source_url,
                )
                columns = lazynwb.table_metadata.get_table_column_metadata(
                    adapter,
                    exact_table_path,
                )
                column_schemas = tuple(
                    catalog_models._column_from_raw_metadata(column)
                    for column in columns
                )
                table_length = lazynwb.table_metadata.get_table_length_from_metadata(
                    columns
                )
        except KeyError:
            raise
        except Exception as exc:
            raise _HDF5ParserError(
                source_url=source_identity.source_url,
                table_path=exact_table_path,
                feature="hdf5_metadata_parser",
                detail=repr(exc),
            ) from exc
        return catalog_models._TableSchemaSnapshot(
            source_identity=source_identity,
            table_path=exact_table_path,
            backend="hdf5",
            columns=column_schemas,
            table_length=table_length,
        )


class _RangeReaderFile(io.RawIOBase):
    """Synchronous file-like adapter over an async range reader."""

    def __init__(
        self,
        range_reader: hdf5_range_reader._RangeReader,
        size: int,
        block_size: int = 64 * 1024,
    ) -> None:
        self._range_reader = range_reader
        self._size = size
        self._position = 0
        self._block_size = block_size
        self._cache: dict[hdf5_range_reader._ByteRange, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return self._position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._position
        if size == 0 or self._position >= self._size:
            return b""
        end = min(self._position + size, self._size)
        payload = self._read_window(self._position, end)
        self._position += len(payload)
        return payload

    def readinto(self, buffer: bytearray | memoryview) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def _read_window(self, start: int, end: int) -> bytes:
        chunks = []
        position = start
        while position < end:
            block_start = (position // self._block_size) * self._block_size
            block_end = min(block_start + self._block_size, self._size)
            block_range = hdf5_range_reader._ByteRange(block_start, block_end)
            block = self._cache.get(block_range)
            if block is None:
                block = asyncio.run(
                    self._range_reader.read_range(block_start, end=block_end)
                )
                self._cache[block_range] = block
            slice_start = position - block_start
            slice_end = min(end, block_end) - block_start
            chunks.append(block[slice_start:slice_end])
            position = block_start + slice_end
        return b"".join(chunks)


class _HDF5AccessorAdapter:
    """Small FileAccessor-like adapter for h5py files opened via range I/O."""

    _hdmf_backend = lazynwb.file_io.FileAccessor.HDMFBackend.HDF5

    def __init__(self, h5_file: h5py.File, source_url: str) -> None:
        self._accessor = h5_file
        self._path = _DisplayPath(source_url)

    def get(self, name: str, default: object = None) -> object:
        return self._accessor.get(name, default)

    def __getitem__(self, name: str) -> object:
        return self._accessor[name]

    def __contains__(self, name: str) -> bool:
        return name in self._accessor

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._accessor)


@dataclasses.dataclass(frozen=True, slots=True)
class _DisplayPath:
    value: str

    def as_posix(self) -> str:
        return self.value


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
