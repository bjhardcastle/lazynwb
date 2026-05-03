from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import math
import pathlib
import threading
import time
import typing
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping

import obstore
import obstore.store

import lazynwb._catalog.models as catalog_models

logger = logging.getLogger(__name__)

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_S3_REGION_CACHE: dict[str, str] = {}
_ObstoreStoreCacheKey = tuple[str, tuple[tuple[str, str], ...], str, str]
_SourceIdentityCacheKey = tuple[
    str,
    str | None,
    tuple[tuple[str, str], ...],
    str,
    str,
]
_OBSTORE_STORE_CACHE_LOCK = threading.RLock()
_OBSTORE_STORE_CACHE: dict[_ObstoreStoreCacheKey, obstore.store.ObjectStore] = {}
_SOURCE_IDENTITY_CACHE_LOCK = threading.RLock()
_SOURCE_IDENTITY_CACHE: dict[
    _SourceIdentityCacheKey,
    catalog_models._SourceIdentity,
] = {}


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
    storage_options: Mapping[str, object] | None = None
    client_options: object | None = None
    retry_config: object | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ObstoreUrlContext:
    """Parsed obstore URL context after storage option normalization."""

    store_url: str
    object_path: str
    storage_options: dict[str, object]


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
        self._semaphore = asyncio.Semaphore(max(1, self._config.max_concurrency))
        self.request_count = 0
        self.bytes_fetched = 0
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
        return await self._read_range(byte_range)

    async def read_ranges(
        self,
        ranges: Iterable[_ByteRange],
    ) -> dict[_ByteRange, bytes]:
        requested_ranges = tuple(ranges)
        if not requested_ranges:
            return {}
        coalesced_ranges = _coalesce_ranges(
            requested_ranges,
            alignment=self._config.range_alignment,
            max_gap=self._config.coalesce_gap_bytes,
        )
        logger.debug(
            "planned %d requested ranges as %d coalesced windows for %s",
            len(requested_ranges),
            len(coalesced_ranges),
            self._url,
        )
        t0 = time.perf_counter()
        fetched = await asyncio.gather(
            *(self._read_range(byte_range) for byte_range in coalesced_ranges)
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
            "read %d requested ranges from %s in %.3f s (%d total requests, %d bytes)",
            len(requested_ranges),
            self._url,
            time.perf_counter() - t0,
            self.request_count,
            self.bytes_fetched,
        )
        return result

    async def _read_range(self, byte_range: _ByteRange) -> bytes:
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
        coalesced_ranges = _coalesce_ranges(
            requested_ranges,
            alignment=self._config.range_alignment,
            max_gap=self._config.coalesce_gap_bytes,
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
) -> tuple[_ByteRange, ...]:
    aligned_ranges = sorted(
        _align_range(byte_range, alignment=alignment) for byte_range in ranges
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


def _align_range(byte_range: _ByteRange, alignment: int) -> _ByteRange:
    if alignment <= 1:
        return byte_range
    start = (byte_range.start // alignment) * alignment
    end = math.ceil(byte_range.end / alignment) * alignment
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
    storage_options = dict(config.storage_options or {})
    if parsed.scheme == "file":
        path = pathlib.Path(urllib.parse.unquote(parsed.path))
        store_url = path.parent.as_uri()
        object_path = path.name
    elif parsed.scheme == "s3":
        store_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        object_path = parsed.path.lstrip("/")
        storage_options = _add_discovered_s3_region(
            bucket=parsed.netloc,
            storage_options=storage_options,
        )
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


def _add_discovered_s3_region(
    bucket: str,
    storage_options: dict[str, object],
) -> dict[str, object]:
    if "region" in storage_options or "aws_region" in storage_options or not bucket:
        return storage_options
    region = _discover_s3_bucket_region(bucket)
    if region is not None:
        storage_options["region"] = region
    return storage_options


def _discover_s3_bucket_region(bucket: str) -> str | None:
    cached = _S3_REGION_CACHE.get(bucket)
    if cached is not None:
        return cached
    request = urllib.request.Request(
        f"https://{bucket}.s3.amazonaws.com",
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            region = response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as exc:
        region = exc.headers.get("x-amz-bucket-region")
    except OSError as exc:
        logger.debug("could not discover S3 bucket region for %s: %r", bucket, exc)
        return None
    if region:
        _S3_REGION_CACHE[bucket] = region
        logger.debug("discovered S3 bucket region for %s: %s", bucket, region)
    return region


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
