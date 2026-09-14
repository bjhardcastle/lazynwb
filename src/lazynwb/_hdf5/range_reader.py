from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime
import logging
import math
import pathlib
import threading
import time
import typing
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping

import obstore
import obstore.store

import lazynwb._catalog.models as catalog_models
import lazynwb._storage_options

logger = logging.getLogger(__name__)

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_ObstoreStoreCacheKey = tuple[str, tuple[tuple[str, str], ...], str, str]
_SourceIdentityCacheKey = tuple[
    str,
    str | None,
    tuple[tuple[str, str], ...],
    str,
    str,
]
_RangeWindowCacheKey = tuple[
    str,
    tuple[tuple[str, str], ...],
    str,
    str,
    str,
    int,
]
_OBSTORE_STORE_CACHE_LOCK = threading.RLock()
_OBSTORE_STORE_CACHE: dict[_ObstoreStoreCacheKey, obstore.store.ObjectStore] = {}
_SOURCE_IDENTITY_CACHE_LOCK = threading.RLock()
_SOURCE_IDENTITY_CACHE: dict[
    _SourceIdentityCacheKey,
    catalog_models._SourceIdentity,
] = {}
_RANGE_WINDOW_CACHES_LOCK = threading.RLock()
_RANGE_WINDOW_CACHES: dict[_RangeWindowCacheKey, _SharedRangeWindowCache] = {}


class _RangeReadError(OSError):
    """Raised when a byte-range read cannot satisfy the requested window."""


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class _ByteRange:
    """Half-open byte range."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True, slots=True)
class _RangeReaderConfig:
    """Configuration for private range readers."""

    range_alignment: int = 4096
    coalesce_gap_bytes: int = 0
    max_concurrency: int = 8
    max_range_cache_bytes: int = 64 * 1024 * 1024
    storage_options: Mapping[str, object] | None = None
    client_options: object | None = None
    retry_config: object | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ObstoreUrlContext:
    """Parsed obstore URL context after storage option normalization."""

    store_url: str
    object_path: str
    storage_options: dict[str, object]


class _SharedRangeWindowCache:
    """Bounded source-level cache of byte windows shared by reader lifetimes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, max_bytes)
        self._windows: collections.OrderedDict[_ByteRange, bytes] = (
            collections.OrderedDict()
        )
        self._total_bytes = 0
        self._lock = threading.RLock()

    def get(self, byte_range: _ByteRange) -> bytes | None:
        with self._lock:
            matching = sorted(
                (cached_range, payload)
                for cached_range, payload in self._windows.items()
                if cached_range.start < byte_range.end
                and byte_range.start < cached_range.end
            )
            cursor = byte_range.start
            pieces: list[bytes] = []
            used_ranges: list[_ByteRange] = []
            for cached_range, payload in matching:
                if cached_range.end <= cursor:
                    continue
                if cached_range.start > cursor:
                    return None
                piece_end = min(byte_range.end, cached_range.end)
                pieces.append(
                    payload[cursor - cached_range.start : piece_end - cached_range.start]
                )
                used_ranges.append(cached_range)
                cursor = piece_end
                if cursor == byte_range.end:
                    for used_range in used_ranges:
                        self._windows.move_to_end(used_range)
                    return b"".join(pieces)
            return None

    def covered_ranges(self, byte_range: _ByteRange) -> tuple[_ByteRange, ...]:
        with self._lock:
            return tuple(
                _ByteRange(
                    max(byte_range.start, cached_range.start),
                    min(byte_range.end, cached_range.end),
                )
                for cached_range in self._windows
                if cached_range.start < byte_range.end
                and byte_range.start < cached_range.end
            )

    def put(self, byte_range: _ByteRange, payload: bytes) -> None:
        if self.max_bytes <= 0 or len(payload) > self.max_bytes:
            return
        with self._lock:
            previous = self._windows.pop(byte_range, None)
            if previous is not None:
                self._total_bytes -= len(previous)
            self._windows[byte_range] = payload
            self._total_bytes += len(payload)
            while self._total_bytes > self.max_bytes:
                _, evicted = self._windows.popitem(last=False)
                self._total_bytes -= len(evicted)

    @property
    def window_count(self) -> int:
        with self._lock:
            return len(self._windows)


@typing.runtime_checkable
class _RangeReader(typing.Protocol):
    """Async byte-range reader Protocol for single-object sources."""

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        """Return source identity derived from object metadata."""

    async def read_range(
        self,
        start: int,
        length: int | None = None,
        end: int | None = None,
    ) -> bytes:
        """Read one half-open byte range."""

    async def read_ranges(
        self,
        ranges: Iterable[_ByteRange],
    ) -> dict[_ByteRange, bytes]:
        """Read several half-open byte ranges."""


@dataclasses.dataclass(frozen=True, slots=True)
class _HDF5SignatureProbeResult:
    is_hdf5: bool
    signature_offset: int | None
    checked_offsets: tuple[int, ...]
    fetched_bytes: int


class _ObstoreRangeReader:
    """Async range reader for obstore-openable single-object URLs."""

    def __init__(
        self,
        url: str,
        config: _RangeReaderConfig | None = None,
    ) -> None:
        self._url = url
        self._config = config or _RangeReaderConfig()
        context = _obstore_url_context(url, self._config)
        self._store_url = context.store_url
        self._object_path = context.object_path
        self._storage_options = context.storage_options
        self._store = _cached_store_from_url(
            self._store_url,
            client_options=self._config.client_options,
            retry_config=self._config.retry_config,
            **self._storage_options,
        )
        self._window_cache = _shared_range_window_cache(
            _range_window_cache_key(
                self._store_url,
                self._object_path,
                self._config,
                self._storage_options,
            ),
            max_bytes=self._config.max_range_cache_bytes,
        )
        self._semaphore = asyncio.Semaphore(max(1, self._config.max_concurrency))
        self._inflight_windows: dict[_ByteRange, asyncio.Task[bytes]] = {}
        self._content_length: int | None = None
        self.request_count = 0
        self.bytes_fetched = 0
        self.cache_hit_count = 0
        self.cache_hit_bytes = 0
        self.inflight_hit_count = 0
        logger.debug(
            "initialized obstore range reader for %s as path %r",
            self._url,
            self._object_path,
        )

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        cached_identity = _get_cached_source_identity(
            self._source_identity_cache_key(resolved_url=None)
        )
        if cached_identity is not None:
            self._content_length = cached_identity.content_length
            logger.debug(
                "source identity cache hit for %s (resolved_url=%r, validator=%s)",
                self._url,
                cached_identity.resolved_url,
                cached_identity.validator_kind,
            )
            return cached_identity
        logger.debug("source identity cache miss for %s", self._url)
        metadata = await obstore.head_async(self._store, self._object_path)
        source_identity = _source_identity_from_metadata(self._url, metadata)
        self._content_length = source_identity.content_length
        _put_cached_source_identity(
            self._source_identity_cache_key(resolved_url=None),
            source_identity,
        )
        if source_identity.resolved_url is not None:
            _put_cached_source_identity(
                self._source_identity_cache_key(
                    resolved_url=source_identity.resolved_url
                ),
                source_identity,
            )
        logger.debug(
            "resolved source identity for %s: validator=%s",
            self._url,
            source_identity.validator_kind,
        )
        return source_identity

    def _source_identity_cache_key(
        self,
        *,
        resolved_url: str | None,
    ) -> _SourceIdentityCacheKey:
        return _source_identity_cache_key(
            self._url,
            resolved_url=resolved_url,
            client_options=self._config.client_options,
            retry_config=self._config.retry_config,
            storage_options=self._storage_options,
        )

    async def read_range(
        self,
        start: int,
        length: int | None = None,
        end: int | None = None,
    ) -> bytes:
        byte_range = _normalize_range(start=start, length=length, end=end)
        return await self._read_cached_range(byte_range)

    async def read_ranges(
        self,
        ranges: Iterable[_ByteRange],
    ) -> dict[_ByteRange, bytes]:
        requested_ranges = tuple(ranges)
        if not requested_ranges:
            return {}
        if self._content_length is None:
            identity = await self.get_source_identity()
            self._content_length = identity.content_length
        _require_ranges_within_bound(
            requested_ranges,
            upper_bound=self._content_length,
            source_url=self._url,
        )
        coalesced_ranges = _coalesce_ranges(
            requested_ranges,
            alignment=self._config.range_alignment,
            max_gap=self._config.coalesce_gap_bytes,
            upper_bound=self._content_length,
        )
        logger.debug(
            "planned %d requested ranges as %d coalesced windows for %s",
            len(requested_ranges),
            len(coalesced_ranges),
            self._url,
        )
        t0 = time.perf_counter()
        fetched = await asyncio.gather(
            *(self._read_cached_range(byte_range) for byte_range in coalesced_ranges)
        )
        by_coalesced_range = dict(zip(coalesced_ranges, fetched, strict=True))
        result: dict[_ByteRange, bytes] = {}
        for requested in requested_ranges:
            containing_range = next(
                byte_range
                for byte_range in coalesced_ranges
                if byte_range.start <= requested.start
                and requested.end <= byte_range.end
            )
            window = by_coalesced_range[containing_range]
            start = requested.start - containing_range.start
            end = requested.end - containing_range.start
            result[requested] = window[start:end]
        logger.debug(
            "read %d requested ranges from %s in %.3f s (%d total requests, "
            "%d bytes, %d cache hits/%d bytes, %d in-flight hits)",
            len(requested_ranges),
            self._url,
            time.perf_counter() - t0,
            self.request_count,
            self.bytes_fetched,
            self.cache_hit_count,
            self.cache_hit_bytes,
            self.inflight_hit_count,
        )
        return result

    async def _read_cached_range(self, byte_range: _ByteRange) -> bytes:
        if (
            self._window_cache.max_bytes <= 0
            or byte_range.length > self._window_cache.max_bytes
        ):
            return await self._fetch_range(byte_range)
        cached = self._window_cache.get(byte_range)
        if cached is not None:
            self.cache_hit_count += 1
            self.cache_hit_bytes += len(cached)
            logger.debug(
                "range cache hit for %s bytes %d:%d (%d shared windows)",
                self._url,
                byte_range.start,
                byte_range.end,
                self._window_cache.window_count,
            )
            return cached

        overlapping_inflight = {
            inflight_range: task
            for inflight_range, task in self._inflight_windows.items()
            if inflight_range.start < byte_range.end
            and byte_range.start < inflight_range.end
        }
        if overlapping_inflight:
            self.inflight_hit_count += len(overlapping_inflight)
            logger.debug(
                "range in-flight cache hit for %s bytes %d:%d across %d windows",
                self._url,
                byte_range.start,
                byte_range.end,
                len(overlapping_inflight),
            )
        covered_ranges = (
            *self._window_cache.covered_ranges(byte_range),
            *overlapping_inflight,
        )
        missing_ranges = _subtract_byte_ranges(byte_range, covered_ranges)
        new_tasks: list[asyncio.Task[bytes]] = []
        for missing_range in missing_ranges:
            task = asyncio.create_task(self._fetch_and_cache_range(missing_range))
            self._inflight_windows[missing_range] = task
            task.add_done_callback(
                lambda completed, planned_range=missing_range: self._remove_inflight_range(
                    planned_range,
                    completed,
                )
            )
            new_tasks.append(task)
        await asyncio.gather(
            *(
                asyncio.shield(task)
                for task in (*overlapping_inflight.values(), *new_tasks)
            )
        )
        cached = self._window_cache.get(byte_range)
        if cached is not None:
            return cached
        logger.debug(
            "range cache could not assemble %s bytes %d:%d after concurrent reads; "
            "fetching directly",
            self._url,
            byte_range.start,
            byte_range.end,
        )
        return await self._fetch_range(byte_range)

    async def _fetch_and_cache_range(self, byte_range: _ByteRange) -> bytes:
        payload = await self._fetch_range(byte_range)
        self._window_cache.put(byte_range, payload)
        return payload

    def _remove_inflight_range(
        self,
        byte_range: _ByteRange,
        completed: asyncio.Task[bytes],
    ) -> None:
        if self._inflight_windows.get(byte_range) is completed:
            self._inflight_windows.pop(byte_range, None)

    async def _fetch_range(self, byte_range: _ByteRange) -> bytes:
        async with self._semaphore:
            t0 = time.perf_counter()
            data = await obstore.get_range_async(
                self._store,
                self._object_path,
                start=byte_range.start,
                end=byte_range.end,
            )
            payload = data.to_bytes()
            self.request_count += 1
            self.bytes_fetched += len(payload)
            if len(payload) < byte_range.length:
                raise _RangeReadError(
                    f"short range response for {self._url}: requested "
                    f"{byte_range.start}:{byte_range.end}, got {len(payload)} bytes"
                )
            logger.debug(
                "read range %d:%d from %s in %.3f s (%d bytes)",
                byte_range.start,
                byte_range.end,
                self._url,
                time.perf_counter() - t0,
                len(payload),
            )
            return payload


class _BufferRangeReader:
    """In-memory range reader used by parser/probe tests."""

    def __init__(
        self,
        data: bytes,
        source_identity: catalog_models._SourceIdentity | None = None,
        config: _RangeReaderConfig | None = None,
    ) -> None:
        self._data = data
        self._source_identity = source_identity or catalog_models._SourceIdentity(
            source_url="memory://buffer",
            content_length=len(data),
            in_process_token=f"buffer:{id(data)}",
        )
        self._config = config or _RangeReaderConfig()
        self.request_count = 0
        self.bytes_fetched = 0

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        return self._source_identity

    async def read_range(
        self,
        start: int,
        length: int | None = None,
        end: int | None = None,
    ) -> bytes:
        byte_range = _normalize_range(start=start, length=length, end=end)
        payload = self._data[byte_range.start : byte_range.end]
        self.request_count += 1
        self.bytes_fetched += len(payload)
        if len(payload) < byte_range.length:
            raise _RangeReadError(
                f"short range response for in-memory buffer: requested "
                f"{byte_range.start}:{byte_range.end}, got {len(payload)} bytes"
            )
        return payload

    async def read_ranges(
        self,
        ranges: Iterable[_ByteRange],
    ) -> dict[_ByteRange, bytes]:
        requested_ranges = tuple(ranges)
        _require_ranges_within_bound(
            requested_ranges,
            upper_bound=len(self._data),
            source_url="memory://buffer",
        )
        coalesced_ranges = _coalesce_ranges(
            requested_ranges,
            alignment=self._config.range_alignment,
            max_gap=self._config.coalesce_gap_bytes,
            upper_bound=len(self._data),
        )
        coalesced_payloads = {
            byte_range: await self.read_range(byte_range.start, end=byte_range.end)
            for byte_range in coalesced_ranges
        }
        result: dict[_ByteRange, bytes] = {}
        for requested in requested_ranges:
            containing_range = next(
                byte_range
                for byte_range in coalesced_ranges
                if byte_range.start <= requested.start
                and requested.end <= byte_range.end
            )
            payload = coalesced_payloads[containing_range]
            start = requested.start - containing_range.start
            end = requested.end - containing_range.start
            result[requested] = payload[start:end]
        return result


async def _probe_hdf5_signature(
    reader: _RangeReader,
    max_probe_offset: int = 65536,
) -> _HDF5SignatureProbeResult:
    checked_offsets: list[int] = []
    fetched_bytes = 0
    for offset in _hdf5_signature_offsets(max_probe_offset=max_probe_offset):
        checked_offsets.append(offset)
        try:
            chunk = await reader.read_range(offset, length=len(_HDF5_SIGNATURE))
        except _RangeReadError:
            logger.debug("short response while probing HDF5 signature at %d", offset)
            break
        fetched_bytes += len(chunk)
        if chunk == _HDF5_SIGNATURE:
            logger.debug("found HDF5 signature at offset %d", offset)
            return _HDF5SignatureProbeResult(
                is_hdf5=True,
                signature_offset=offset,
                checked_offsets=tuple(checked_offsets),
                fetched_bytes=fetched_bytes,
            )
    logger.debug("no HDF5 signature found after checking offsets %s", checked_offsets)
    return _HDF5SignatureProbeResult(
        is_hdf5=False,
        signature_offset=None,
        checked_offsets=tuple(checked_offsets),
        fetched_bytes=fetched_bytes,
    )


def _hdf5_signature_offsets(max_probe_offset: int = 65536) -> tuple[int, ...]:
    offsets = [0]
    offset = 512
    while offset <= max_probe_offset:
        offsets.append(offset)
        offset *= 2
    return tuple(offsets)


def _normalize_range(
    start: int,
    length: int | None = None,
    end: int | None = None,
) -> _ByteRange:
    if start < 0:
        raise ValueError("range start must be non-negative")
    if (length is None) == (end is None):
        raise ValueError("provide exactly one of length or end")
    resolved_end = start + length if length is not None else end
    if resolved_end is None or resolved_end < start:
        raise ValueError("range end must be greater than or equal to start")
    return _ByteRange(start=start, end=resolved_end)


def _coalesce_ranges(
    ranges: Iterable[_ByteRange],
    alignment: int,
    max_gap: int,
    upper_bound: int | None = None,
) -> tuple[_ByteRange, ...]:
    aligned_ranges = sorted(
        _align_range(
            byte_range,
            alignment=alignment,
            upper_bound=upper_bound,
        )
        for byte_range in ranges
    )
    if not aligned_ranges:
        return ()
    coalesced = [aligned_ranges[0]]
    for byte_range in aligned_ranges[1:]:
        current = coalesced[-1]
        if byte_range.start <= current.end + max_gap:
            coalesced[-1] = _ByteRange(
                start=current.start,
                end=max(current.end, byte_range.end),
            )
        else:
            coalesced.append(byte_range)
    return tuple(coalesced)


def _subtract_byte_ranges(
    byte_range: _ByteRange,
    covered_ranges: Iterable[_ByteRange],
) -> tuple[_ByteRange, ...]:
    clipped_coverage = sorted(
        _ByteRange(
            max(byte_range.start, covered.start),
            min(byte_range.end, covered.end),
        )
        for covered in covered_ranges
        if covered.start < byte_range.end and byte_range.start < covered.end
    )
    missing: list[_ByteRange] = []
    cursor = byte_range.start
    for covered in clipped_coverage:
        if cursor < covered.start:
            missing.append(_ByteRange(cursor, covered.start))
        cursor = max(cursor, covered.end)
    if cursor < byte_range.end:
        missing.append(_ByteRange(cursor, byte_range.end))
    return tuple(missing)


def _require_ranges_within_bound(
    ranges: Iterable[_ByteRange],
    *,
    upper_bound: int | None,
    source_url: str,
) -> None:
    if upper_bound is None:
        return
    for byte_range in ranges:
        if byte_range.end > upper_bound:
            raise _RangeReadError(
                f"range {byte_range.start}:{byte_range.end} exceeds "
                f"{source_url} content length {upper_bound}"
            )


def _align_range(
    byte_range: _ByteRange,
    alignment: int,
    upper_bound: int | None = None,
) -> _ByteRange:
    if alignment <= 1:
        return byte_range
    start = (byte_range.start // alignment) * alignment
    end = math.ceil(byte_range.end / alignment) * alignment
    if upper_bound is not None:
        end = min(end, upper_bound)
    return _ByteRange(start=start, end=end)


def _source_identity_from_metadata(
    source_url: str,
    metadata: Mapping[str, object],
) -> catalog_models._SourceIdentity:
    last_modified = metadata.get("last_modified")
    if isinstance(last_modified, datetime.datetime):
        last_modified_str = last_modified.isoformat()
    elif last_modified is None:
        last_modified_str = None
    else:
        last_modified_str = str(last_modified)
    return catalog_models._SourceIdentity(
        source_url=source_url,
        resolved_url=_optional_str(metadata.get("location")) or source_url,
        content_length=_optional_int(metadata.get("size")),
        version_id=_optional_str(metadata.get("version")),
        etag=_optional_str(metadata.get("e_tag")),
        last_modified=last_modified_str,
    )


def _store_and_path_from_url(
    url: str,
    config: _RangeReaderConfig,
) -> tuple[obstore.store.ObjectStore, str]:
    context = _obstore_url_context(url, config)
    store = _cached_store_from_url(
        context.store_url,
        client_options=config.client_options,
        retry_config=config.retry_config,
        **context.storage_options,
    )
    return store, context.object_path


def _obstore_url_context(
    url: str,
    config: _RangeReaderConfig,
) -> _ObstoreUrlContext:
    parsed = urllib.parse.urlsplit(url)
    s3_bucket = _s3_bucket_from_parsed_url(parsed)
    if s3_bucket is not None and parsed.scheme in {"http", "https"}:
        logger.debug(
            "identified S3 bucket %s from %s URL host %s for range reader",
            s3_bucket,
            parsed.scheme,
            parsed.netloc,
        )
    storage_options = (
        lazynwb._storage_options._get_obstore_range_reader_storage_options(
            config.storage_options,
            s3_bucket=s3_bucket,
        )
    )
    if parsed.scheme == "file":
        path = _path_from_file_url(parsed)
        store_url = path.parent.as_uri()
        object_path = path.name
    elif parsed.scheme == "s3":
        store_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        object_path = parsed.path.lstrip("/")
    elif parsed.scheme in {"http", "https", "gs", "gcs", "az", "abfs"}:
        store_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        object_path = parsed.path.lstrip("/")
    else:
        raise ValueError(f"unsupported obstore URL scheme for range reader: {url!r}")
    if not object_path:
        raise ValueError(f"range reader URL must identify one object: {url!r}")
    return _ObstoreUrlContext(
        store_url=store_url,
        object_path=object_path,
        storage_options=storage_options,
    )


def _path_from_file_url(parsed: urllib.parse.SplitResult) -> pathlib.Path:
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{parsed.path}"
    else:
        path = parsed.path
    return pathlib.Path(urllib.request.url2pathname(path))


def _s3_bucket_from_parsed_url(parsed: urllib.parse.SplitResult) -> str | None:
    if parsed.scheme == "s3":
        return parsed.netloc
    if parsed.scheme not in {"http", "https"}:
        return None
    netloc = parsed.netloc
    host = (parsed.hostname or netloc.split("@")[-1].split(":")[0]).lower()
    suffixes = (".s3.amazonaws.com", ".s3.dualstack.amazonaws.com")
    for suffix in suffixes:
        if host.endswith(suffix):
            return host[: -len(suffix)]
    host_parts = host.split(".")
    if (
        len(host_parts) >= 5
        and host_parts[1] == "s3"
        and host_parts[-2:]
        == [
            "amazonaws",
            "com",
        ]
    ):
        return host_parts[0]
    return None


def _cached_store_from_url(
    store_url: str,
    *,
    client_options: object | None = None,
    retry_config: object | None = None,
    **storage_options: object,
) -> obstore.store.ObjectStore:
    cache_key = _obstore_store_cache_key(
        store_url,
        client_options=client_options,
        retry_config=retry_config,
        storage_options=storage_options,
    )
    with _OBSTORE_STORE_CACHE_LOCK:
        cached = _OBSTORE_STORE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug(
                "obstore store cache hit for %s (storage_options=%s, "
                "client_options=%s, retry_config=%s)",
                store_url,
                sorted(str(key) for key in storage_options),
                client_options is not None,
                retry_config is not None,
            )
            return cached
        logger.debug(
            "obstore store cache miss for %s (storage_options=%s, "
            "client_options=%s, retry_config=%s)",
            store_url,
            sorted(str(key) for key in storage_options),
            client_options is not None,
            retry_config is not None,
        )
        store = obstore.store.from_url(
            store_url,
            client_options=client_options,
            retry_config=retry_config,
            **storage_options,
        )
        _OBSTORE_STORE_CACHE[cache_key] = store
        logger.debug("created cached obstore store for %s", store_url)
        return store


def _range_window_cache_key(
    store_url: str,
    object_path: str,
    config: _RangeReaderConfig,
    storage_options: Mapping[str, object],
) -> _RangeWindowCacheKey:
    return (
        *_obstore_store_cache_key(
            store_url,
            client_options=config.client_options,
            retry_config=config.retry_config,
            storage_options=storage_options,
        ),
        object_path,
        config.max_range_cache_bytes,
    )


def _shared_range_window_cache(
    cache_key: _RangeWindowCacheKey,
    *,
    max_bytes: int,
) -> _SharedRangeWindowCache:
    with _RANGE_WINDOW_CACHES_LOCK:
        cache = _RANGE_WINDOW_CACHES.get(cache_key)
        if cache is None:
            cache = _SharedRangeWindowCache(max_bytes)
            _RANGE_WINDOW_CACHES[cache_key] = cache
            logger.debug(
                "created shared range window cache for %s/%s with %d byte budget",
                cache_key[0],
                cache_key[4],
                max_bytes,
            )
        return cache


def _clear_range_window_caches() -> None:
    with _RANGE_WINDOW_CACHES_LOCK:
        logger.debug(
            "clearing %d shared range window caches",
            len(_RANGE_WINDOW_CACHES),
        )
        _RANGE_WINDOW_CACHES.clear()


def _obstore_store_cache_key(
    store_url: str,
    *,
    client_options: object | None,
    retry_config: object | None,
    storage_options: Mapping[str, object],
) -> _ObstoreStoreCacheKey:
    return (
        store_url,
        tuple(
            sorted((str(key), repr(value)) for key, value in storage_options.items())
        ),
        repr(client_options),
        repr(retry_config),
    )


def _clear_obstore_store_cache() -> None:
    with _OBSTORE_STORE_CACHE_LOCK:
        logger.debug(
            "clearing obstore store cache with %d entries",
            len(_OBSTORE_STORE_CACHE),
        )
        _OBSTORE_STORE_CACHE.clear()


def _source_identity_cache_key(
    source_url: str,
    *,
    resolved_url: str | None,
    client_options: object | None,
    retry_config: object | None,
    storage_options: Mapping[str, object],
) -> _SourceIdentityCacheKey:
    return (
        source_url,
        resolved_url,
        tuple(
            sorted((str(key), repr(value)) for key, value in storage_options.items())
        ),
        repr(client_options),
        repr(retry_config),
    )


def _get_cached_source_identity(
    cache_key: _SourceIdentityCacheKey,
) -> catalog_models._SourceIdentity | None:
    with _SOURCE_IDENTITY_CACHE_LOCK:
        return _SOURCE_IDENTITY_CACHE.get(cache_key)


def _put_cached_source_identity(
    cache_key: _SourceIdentityCacheKey,
    source_identity: catalog_models._SourceIdentity,
) -> None:
    with _SOURCE_IDENTITY_CACHE_LOCK:
        _SOURCE_IDENTITY_CACHE[cache_key] = source_identity
        logger.debug(
            "stored source identity cache entry for %s (resolved_url=%r, "
            "validator=%s)",
            source_identity.source_url,
            cache_key[1],
            source_identity.validator_kind,
        )


def _clear_source_identity_cache() -> None:
    with _SOURCE_IDENTITY_CACHE_LOCK:
        logger.debug(
            "clearing source identity cache with %d entries",
            len(_SOURCE_IDENTITY_CACHE),
        )
        _SOURCE_IDENTITY_CACHE.clear()


def _clear_s3_region_cache() -> None:
    lazynwb._storage_options._clear_s3_region_cache()


def _clear_cache() -> None:
    """Reset process-lifetime range-reader caches.

    Source identities are reused for the life of the process to avoid repeated
    metadata HEAD requests. Callers that mutate or replace source objects in the
    same process must call ``lazynwb.clear_cache()`` before reading them again.
    Persistent SQLite caches still receive the original source identity and keep
    the existing validator priority: version ID, strong ETag, last-modified plus
    content length, then in-process token.
    """
    _clear_obstore_store_cache()
    _clear_source_identity_cache()
    _clear_range_window_caches()
    _clear_s3_region_cache()
    lazynwb._storage_options._clear_default_s3_credential_provider()


def _add_discovered_s3_region(
    bucket: str,
    storage_options: dict[str, object],
) -> dict[str, object]:
    return lazynwb._storage_options._add_discovered_s3_region(
        bucket=bucket,
        storage_options=storage_options,
        discover_bucket_region=_discover_s3_bucket_region,
    )


def _has_custom_s3_endpoint(storage_options: Mapping[str, object]) -> bool:
    return lazynwb._storage_options._has_custom_s3_endpoint(storage_options)


def _discover_s3_bucket_region(bucket: str) -> str | None:
    return lazynwb._storage_options._discover_s3_bucket_region(bucket)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
