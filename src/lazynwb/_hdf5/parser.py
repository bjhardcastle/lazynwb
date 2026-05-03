from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import struct
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb.exceptions
import lazynwb.table_metadata
import lazynwb.utils

logger = logging.getLogger(__name__)

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_TABLE_MARKER_ATTRIBUTE_NAME = "colnames"
_ATTRIBUTE_SKIPPED = object()
_PARSED_ATTRIBUTE_VALUE_NAMES = frozenset(
    {
        "conversion",
        "file_name",
        "neurodata_type",
        "nwb_version",
        "offset",
        "reference",
        "resolution",
    }
)
_DEFAULT_BOOTSTRAP_BYTES = int(os.getenv("LAZYNWB_HDF5_BOOTSTRAP_BYTES", 16 * 1024))
_DEFAULT_MAX_BOOTSTRAP_BYTES = int(
    os.getenv("LAZYNWB_HDF5_MAX_BOOTSTRAP_BYTES", 64 * 1024)
)
_DEFAULT_OBJECT_HEADER_BOOTSTRAP_BYTES = int(
    os.getenv("LAZYNWB_HDF5_OBJECT_HEADER_BOOTSTRAP_BYTES", 2 * 1024)
)
_DEFAULT_ALIGNMENT = int(os.getenv("LAZYNWB_HDF5_ALIGNMENT", 4 * 1024))
_DEFAULT_MERGE_GAP = int(os.getenv("LAZYNWB_HDF5_MERGE_GAP", 64 * 1024))
_PARSED_METADATA_PAYLOAD_VERSION = 1

_ByteBuffer = bytes


class _TableNotFoundError(KeyError):
    """Raised when an exact table path is absent from parsed HDF5 metadata."""


@dataclasses.dataclass(frozen=True, slots=True)
class _Window:
    start: int
    end: int


@dataclasses.dataclass(frozen=True, slots=True)
class _ContinuationSpan:
    address: int
    length: int


@dataclasses.dataclass(frozen=True, slots=True)
class _GlobalHeapReference:
    address: int
    index: int
    length: int


@dataclasses.dataclass(frozen=True, slots=True)
class _OldGroupPointers:
    btree_address: int
    heap_address: int


@dataclasses.dataclass(frozen=True, slots=True)
class _GroupEntry:
    name: str
    object_header_address: int
    old_group: _OldGroupPointers | None = None


@dataclasses.dataclass(slots=True)
class _GroupHandle:
    path: str
    object_header_address: int
    old_group: _OldGroupPointers | None = None


@dataclasses.dataclass(slots=True)
class _SuperblockInfo:
    version: int
    offset_size: int
    length_size: int
    base_address: int
    group_leaf_k: int
    group_internal_k: int
    root_group: _GroupHandle


@dataclasses.dataclass(slots=True)
class _H5Datatype:
    kind: str
    size: int
    signed: bool | None = None
    base: _H5Datatype | None = None
    shape: tuple[int, ...] = ()
    enum_names: tuple[str, ...] = ()

    @property
    def numpy_dtype(self) -> str | None:
        if self.kind == "int":
            prefix = "i" if self.signed else "u"
            endian = "|" if self.size == 1 else "<"
            return f"{endian}{prefix}{self.size}"
        if self.kind == "float":
            return f"<f{self.size}"
        if self.kind == "bool":
            return "|b1"
        if self.kind == "string":
            return f"|S{self.size}"
        if self.kind in {"vlen-string", "reference"}:
            return "|O"
        if self.kind == "array" and self.base is not None:
            return self.base.numpy_dtype
        return None

    @property
    def neutral_kind(self) -> str:
        if self.kind == "vlen-string":
            return "vlen_string"
        if self.kind in {"int", "float"}:
            return "numeric"
        if self.kind in {"string", "reference", "bool", "enum", "compound", "opaque"}:
            return self.kind
        if self.kind == "array":
            return "array"
        return "unknown"

    def to_neutral_dtype(self) -> catalog_models._NeutralDType:
        if self.kind == "array" and self.base is not None:
            base_dtype = self.base.to_neutral_dtype()
            return catalog_models._NeutralDType(
                kind="array",
                numpy_dtype=self.numpy_dtype,
                byte_order=_byte_order_from_numpy_dtype(self.numpy_dtype),
                itemsize=self.size,
                detail=f"array[{base_dtype.detail or base_dtype.numpy_dtype}]",
                element_numpy_dtype=base_dtype.numpy_dtype,
                element_shape=self.shape or None,
            )
        return catalog_models._NeutralDType(
            kind=self.neutral_kind,
            numpy_dtype=self.numpy_dtype,
            byte_order=_byte_order_from_numpy_dtype(self.numpy_dtype),
            itemsize=self.size if self.size else None,
            detail=self.kind,
        )


@dataclasses.dataclass(slots=True)
class _DatasetDescriptor:
    name: str
    path: str
    shape: tuple[int, ...]
    datatype: _H5Datatype
    attributes: dict[str, Any]
    layout: dict[str, Any] | None
    fill_value: dict[str, Any] | None
    filters: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclasses.dataclass(slots=True)
class _ObjectHeaderInfo:
    address: int
    datatype: _H5Datatype | None = None
    dataspace: tuple[int, ...] | None = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    old_group: _OldGroupPointers | None = None
    layout: dict[str, Any] | None = None
    fill_value: dict[str, Any] | None = None
    filters: tuple[int, ...] = ()
    has_colnames: bool = False

    @property
    def is_group(self) -> bool:
        return self.old_group is not None

    @property
    def is_dataset(self) -> bool:
        return self.datatype is not None and self.dataspace is not None


class _RangeWindowCache:
    """Small coalescing window cache for parser byte reads."""

    def __init__(
        self,
        reader: hdf5_range_reader._RangeReader,
        *,
        source_url: str,
        content_length: int | None,
        alignment: int = _DEFAULT_ALIGNMENT,
        merge_gap: int = _DEFAULT_MERGE_GAP,
    ) -> None:
        self._reader = reader
        self._source_url = source_url
        self._content_length = content_length
        self._alignment = alignment
        self._merge_gap = merge_gap
        self._windows: list[tuple[_Window, bytes]] = []

    @staticmethod
    def _merge_payloads(
        left_window: _Window,
        left_payload: bytes,
        right_window: _Window,
        right_payload: bytes,
    ) -> tuple[_Window, bytes]:
        start = min(left_window.start, right_window.start)
        end = max(left_window.end, right_window.end)
        merged = bytearray(end - start)
        left_start = left_window.start - start
        merged[left_start : left_start + len(left_payload)] = left_payload
        right_start = right_window.start - start
        merged[right_start : right_start + len(right_payload)] = right_payload
        return _Window(start=start, end=end), bytes(merged)

    def _covers(self, start: int, end: int) -> bool:
        cursor = start
        for window, _ in self._windows:
            if window.end <= cursor:
                continue
            if window.start > cursor:
                return False
            cursor = min(end, window.end)
            if cursor >= end:
                return True
        return cursor >= end

    def _lookup(self, start: int, end: int) -> bytes | None:
        parts: list[bytes] = []
        cursor = start
        for window, payload in self._windows:
            if window.end <= cursor:
                continue
            if window.start > cursor:
                break
            left = max(cursor, window.start) - window.start
            right = min(end, window.end) - window.start
            if right <= left:
                continue
            parts.append(payload[left:right])
            cursor += right - left
            if cursor >= end:
                break
        if cursor < end:
            return None
        if len(parts) == 1:
            return parts[0]
        merged = bytearray(end - start)
        position = 0
        for part in parts:
            merged[position : position + len(part)] = part
            position += len(part)
        return bytes(merged)

    def _insert(self, window: _Window, payload: bytes) -> None:
        actual_end = window.start + len(payload)
        if actual_end > window.end:
            payload = payload[: window.end - window.start]
            actual_end = window.end
        if actual_end <= window.start:
            return
        if actual_end < window.end:
            logger.debug(
                "range response for %s bytes=%s-%s was short; caching bytes=%s-%s",
                self._source_url,
                window.start,
                window.end - 1,
                window.start,
                actual_end - 1,
            )
            window = _Window(start=window.start, end=actual_end)
        merged_window = window
        merged_payload = payload
        merged_windows: list[tuple[_Window, bytes]] = []
        inserted = False
        for existing_window, existing_payload in self._windows:
            if existing_window.end < merged_window.start:
                merged_windows.append((existing_window, existing_payload))
                continue
            if merged_window.end < existing_window.start:
                if not inserted:
                    merged_windows.append((merged_window, merged_payload))
                    inserted = True
                merged_windows.append((existing_window, existing_payload))
                continue
            merged_window, merged_payload = self._merge_payloads(
                existing_window,
                existing_payload,
                merged_window,
                merged_payload,
            )
        if not inserted:
            merged_windows.append((merged_window, merged_payload))
        self._windows = merged_windows

    async def prefetch(self, spans: Iterable[tuple[int, int]]) -> None:
        normalized: list[tuple[int, int]] = []
        for start, end in spans:
            if end <= start:
                continue
            start = _align_down(max(0, start), self._alignment)
            end = _align_up(end, self._alignment)
            if self._content_length is not None:
                if start >= self._content_length:
                    continue
                end = min(end, self._content_length)
            if not self._covers(start, end):
                normalized.append((start, end))
        windows = _coalesce(normalized, self._merge_gap)
        if not windows:
            return
        logger.debug(
            "prefetching %d HDF5 parser windows for %d spans on %s",
            len(windows),
            len(normalized),
            self._source_url,
        )
        payloads = await asyncio.gather(
            *(
                self._reader.read_range(window.start, end=window.end)
                for window in windows
            )
        )
        for window, payload in zip(windows, payloads, strict=True):
            self._insert(window, payload)

    async def read(self, start: int, size: int) -> bytes:
        end = start + size
        if self._content_length is not None:
            end = min(end, self._content_length)
        if end <= start:
            return b""
        payload = self._lookup(start, end)
        if payload is not None:
            return payload
        await self.prefetch([(start, end)])
        payload = self._lookup(start, end)
        if payload is None:
            raise ValueError(f"failed to read cached HDF5 bytes {start}:{end}")
        return payload


class _HDF5MetadataScanner:
    """Parser-backed HDF5 metadata scanner adapted from nwbscan."""

    def __init__(
        self,
        source_url: str,
        reader: hdf5_range_reader._RangeReader,
        *,
        content_length: int | None = None,
        bootstrap_bytes: int = _DEFAULT_BOOTSTRAP_BYTES,
        max_bootstrap_bytes: int = _DEFAULT_MAX_BOOTSTRAP_BYTES,
        object_header_bootstrap_bytes: int = _DEFAULT_OBJECT_HEADER_BOOTSTRAP_BYTES,
        alignment: int = _DEFAULT_ALIGNMENT,
        merge_gap: int = _DEFAULT_MERGE_GAP,
        resolve_vlen_attributes: bool = False,
    ) -> None:
        self.source_url = source_url
        self.reader = reader
        self.content_length = content_length
        self.bootstrap_bytes = bootstrap_bytes
        self.max_bootstrap_bytes = max(max_bootstrap_bytes, bootstrap_bytes)
        self.object_header_bootstrap_bytes = object_header_bootstrap_bytes
        self.resolve_vlen_attributes = resolve_vlen_attributes
        self.window_cache = _RangeWindowCache(
            reader,
            source_url=source_url,
            content_length=content_length,
            alignment=alignment,
            merge_gap=merge_gap,
        )
        self.superblock: _SuperblockInfo | None = None
        self.group_cache: dict[str, dict[str, _GroupEntry]] = {}
        self.object_header_cache: dict[int, _ObjectHeaderInfo] = {}
        self.local_heap_cache: dict[int, bytes] = {}
        self.global_heap_cache: dict[int, dict[int, bytes]] = {}
        self.is_hdf5: bool | None = None

    def export_metadata(self) -> dict[str, Any]:
        return {
            "payload_version": _PARSED_METADATA_PAYLOAD_VERSION,
            "superblock": _encode_superblock(self.superblock),
            "is_hdf5": self.is_hdf5,
            "groups": {
                path: [_encode_group_entry(entry) for entry in members.values()]
                for path, members in self.group_cache.items()
            },
            "object_headers": [
                _encode_object_header(info)
                for info in self.object_header_cache.values()
            ],
        }

    def import_metadata(self, payload: Mapping[str, Any]) -> bool:
        try:
            if (
                int(payload.get("payload_version", 0))
                != _PARSED_METADATA_PAYLOAD_VERSION
            ):
                logger.debug(
                    "parsed HDF5 metadata version mismatch for %s: %r",
                    self.source_url,
                    payload.get("payload_version"),
                )
                return False
            superblock = _decode_superblock(payload.get("superblock"))
            groups_payload = payload.get("groups", {})
            object_headers_payload = payload.get("object_headers", ())
            if not isinstance(groups_payload, Mapping):
                raise ValueError("groups must be an object")
            if not isinstance(object_headers_payload, list):
                raise ValueError("object_headers must be a list")
            group_cache: dict[str, dict[str, _GroupEntry]] = {}
            for path, entries in groups_payload.items():
                if not isinstance(entries, list):
                    raise ValueError(f"group {path!r} entries must be a list")
                decoded_entries = [_decode_group_entry(entry) for entry in entries]
                group_cache[str(path)] = {
                    entry.name: entry for entry in decoded_entries
                }
            object_header_cache = {
                info.address: info
                for info in (
                    _decode_object_header(item) for item in object_headers_payload
                )
            }
        except Exception as exc:
            logger.debug(
                "ignoring invalid parsed HDF5 metadata for %s: %s", self.source_url, exc
            )
            return False

        self.superblock = superblock
        self.is_hdf5 = (
            bool(payload.get("is_hdf5"))
            if superblock is not None
            else payload.get("is_hdf5")
        )
        self.group_cache.update(group_cache)
        self.object_header_cache.update(object_header_cache)
        logger.debug(
            "loaded parsed HDF5 metadata for %s: groups=%d object_headers=%d",
            self.source_url,
            len(group_cache),
            len(object_header_cache),
        )
        return True

    async def read_table_column_schemas(
        self,
        exact_table_path: str,
        source_identity: catalog_models._SourceIdentity,
    ) -> tuple[catalog_models._TableColumnSchema, ...]:
        started = time.perf_counter()
        await self.bootstrap()
        descriptors = await self.describe_table(exact_table_path)
        column_names = tuple(descriptors)
        is_metadata_table = exact_table_path == "general" or _is_metadata(descriptors)
        is_timeseries = lazynwb.table_metadata._is_timeseries(column_names)
        is_timeseries_with_rate = lazynwb.table_metadata._is_timeseries_with_rate(
            column_names
        )
        timeseries_len = (
            descriptors["data"].shape[0]
            if is_timeseries and "data" in descriptors and descriptors["data"].shape
            else None
        )
        columns = tuple(
            _column_schema_from_descriptor(
                descriptor=descriptor,
                all_column_names=column_names,
                is_metadata_table=is_metadata_table,
                is_timeseries=is_timeseries,
                is_timeseries_with_rate=is_timeseries_with_rate,
                source_identity=source_identity,
                table_path=exact_table_path,
                timeseries_len=timeseries_len,
            )
            for descriptor in descriptors.values()
        )
        logger.debug(
            "built %d parser-derived HDF5 catalog columns for %s/%s in %.3f s",
            len(columns),
            source_identity.source_url,
            exact_table_path,
            time.perf_counter() - started,
        )
        return columns

    async def warm_table_metadata(self, exact_table_paths: Iterable[str]) -> None:
        for exact_table_path in dict.fromkeys(exact_table_paths):
            try:
                normalized_path = _normalize_exact_path(exact_table_path)
                if normalized_path == "general":
                    await self._describe_general_table()
                else:
                    group = await self.resolve_group(f"/{normalized_path}")
                    entries = await self.enumerate_group(group)
                    await self.describe_group_members(
                        parent_path=f"/{normalized_path}",
                        entries=entries,
                    )
            except Exception as exc:
                logger.debug(
                    "could not warm HDF5 parsed metadata for %s/%s: %r",
                    self.source_url,
                    exact_table_path,
                    exc,
                )
            else:
                logger.debug(
                    "warmed HDF5 parsed metadata for %s/%s",
                    self.source_url,
                    exact_table_path,
                )

    async def bootstrap(self) -> _SuperblockInfo:
        if self.superblock is not None:
            self.is_hdf5 = True
            return self.superblock
        boot, superblock_offset = await self._read_superblock_probe()
        if superblock_offset is None:
            raise ValueError(
                f"no HDF5 superblock found in first {self.max_bootstrap_bytes} bytes"
            )
        version = boot[superblock_offset + 8]
        if version not in (0, 1):
            raise NotImplementedError(
                f"only HDF5 superblock v0/v1 is supported, found v{version}"
            )
        offset_size = boot[superblock_offset + 13]
        length_size = boot[superblock_offset + 14]
        group_leaf_k = _u(boot, superblock_offset + 16, 2)
        group_internal_k = _u(boot, superblock_offset + 18, 2)
        prefix_size = 24 if version == 0 else 28
        base_address = _u(boot, superblock_offset + prefix_size, offset_size)
        root_entry_offset = superblock_offset + prefix_size + (4 * offset_size)
        root_entry = self._parse_symbol_table_entry(
            boot[root_entry_offset : root_entry_offset + (2 * offset_size + 24)],
            heap_data=None,
            offset_size=offset_size,
        )
        if root_entry.old_group is None:
            raise ValueError(
                "root symbol table entry did not include old-style group cache"
            )
        self.superblock = _SuperblockInfo(
            version=version,
            offset_size=offset_size,
            length_size=length_size,
            base_address=base_address,
            group_leaf_k=group_leaf_k,
            group_internal_k=group_internal_k,
            root_group=_GroupHandle(
                path="/",
                object_header_address=root_entry.object_header_address,
                old_group=root_entry.old_group,
            ),
        )
        logger.debug(
            "parsed HDF5 superblock for %s: version=%s offsets=%s lengths=%s base=%s",
            self.source_url,
            version,
            offset_size,
            length_size,
            base_address,
        )
        return self.superblock

    async def describe_table(
        self,
        exact_table_path: str,
    ) -> dict[str, _DatasetDescriptor]:
        normalized_path = _normalize_exact_path(exact_table_path)
        if normalized_path == "general":
            descriptors = await self._describe_general_table()
        else:
            group = await self.resolve_group(f"/{normalized_path}")
            entries = await self.enumerate_group(group)
            descriptors = await self.describe_group_members(
                parent_path=f"/{normalized_path}",
                entries=entries,
            )
        _drop_known_reference_columns(descriptors)
        if not descriptors:
            raise lazynwb.exceptions.InternalPathError(
                f"{exact_table_path!r} did not contain any supported dataset columns"
            )
        return descriptors

    async def resolve_group(self, path: str) -> _GroupHandle:
        superblock = await self.bootstrap()
        normalized = _normalize_internal_path(path)
        if normalized == "/":
            return superblock.root_group
        current = superblock.root_group
        current_path = ""
        for segment in normalized.strip("/").split("/"):
            members = await self.enumerate_group(current)
            try:
                entry = members[segment]
            except KeyError as exc:
                raise _TableNotFoundError(
                    f"missing {segment!r} under {current.path!r}"
                ) from exc
            current_path = f"{current_path}/{segment}"
            current = await self._entry_to_group_handle(current_path, entry)
        return current

    async def enumerate_group(self, group: _GroupHandle) -> dict[str, _GroupEntry]:
        if group.path in self.group_cache:
            logger.debug(
                "HDF5 parser group cache hit for %s/%s", self.source_url, group.path
            )
            return self.group_cache[group.path]
        pointers = group.old_group
        if pointers is None:
            header = await self.load_object_headers([group.object_header_address])
            pointers = header[group.object_header_address].old_group
        if pointers is None:
            raise NotImplementedError(
                f"only old-style HDF5 groups are supported: {group.path}"
            )
        heap_data = await self._read_local_heap_data(pointers.heap_address)
        symbol_node_addresses = await self._collect_group_symbol_nodes(
            pointers.btree_address
        )
        symbol_node_size = self._symbol_table_node_size()
        await self.window_cache.prefetch(
            [
                (self._absolute(address), self._absolute(address) + symbol_node_size)
                for address in symbol_node_addresses
            ]
        )
        members: dict[str, _GroupEntry] = {}
        for node_address in symbol_node_addresses:
            raw_node = await self.window_cache.read(
                self._absolute(node_address),
                symbol_node_size,
            )
            for entry in self._parse_symbol_table_node(raw_node, heap_data):
                members[entry.name] = entry
        self.group_cache[group.path] = members
        logger.debug(
            "enumerated %d HDF5 group members under %s/%s",
            len(members),
            self.source_url,
            group.path,
        )
        return members

    async def describe_group_members(
        self,
        *,
        parent_path: str,
        entries: Mapping[str, _GroupEntry],
    ) -> dict[str, _DatasetDescriptor]:
        object_addresses = [
            entry.object_header_address
            for entry in entries.values()
            if self._is_defined_address(entry.object_header_address)
        ]
        headers = await self.load_object_headers(object_addresses)
        descriptors: dict[str, _DatasetDescriptor] = {}
        for name, entry in entries.items():
            header = headers.get(entry.object_header_address)
            if header is None or header.is_group or not header.is_dataset:
                continue
            descriptors[name] = self._descriptor_from_header(
                name=name,
                path=f"{parent_path.rstrip('/')}/{name}".lstrip("/"),
                header=header,
            )
        return descriptors

    async def load_object_headers(
        self,
        addresses: Sequence[int],
    ) -> dict[int, _ObjectHeaderInfo]:
        pending_addresses = [
            address
            for address in dict.fromkeys(addresses)
            if address not in self.object_header_cache
            and self._is_defined_address(address)
        ]
        if pending_addresses:
            await self.window_cache.prefetch(
                [
                    (
                        self._absolute(address),
                        self._absolute(address) + self.object_header_bootstrap_bytes,
                    )
                    for address in pending_addresses
                ]
            )
        states: dict[
            int,
            tuple[_ObjectHeaderInfo, set[_ContinuationSpan], list[_ContinuationSpan]],
        ] = {}
        for address in pending_addresses:
            payload = await self.window_cache.read(
                self._absolute(address),
                self.object_header_bootstrap_bytes,
            )
            info, seen, pending = self._parse_v1_object_header_initial(address, payload)
            states[address] = (info, seen, pending)
        pending_spans = sorted(
            {span for _, _, spans in states.values() for span in spans},
            key=lambda span: (span.address, span.length),
        )
        while pending_spans:
            await self.window_cache.prefetch(
                [
                    (
                        self._absolute(span.address),
                        self._absolute(span.address) + span.length,
                    )
                    for span in pending_spans
                ]
            )
            next_pending: set[_ContinuationSpan] = set()
            for address, (info, seen, spans) in list(states.items()):
                new_spans: list[_ContinuationSpan] = []
                for span in spans:
                    block = await self.window_cache.read(
                        self._absolute(span.address),
                        span.length,
                    )
                    discovered = self._apply_v1_message_block(info, block)
                    for discovered_span in discovered:
                        if discovered_span not in seen:
                            seen.add(discovered_span)
                            new_spans.append(discovered_span)
                states[address] = (info, seen, new_spans)
                next_pending.update(new_spans)
            pending_spans = sorted(
                next_pending,
                key=lambda span: (span.address, span.length),
            )
        for address, (info, _, _) in states.items():
            if self.resolve_vlen_attributes:
                await self._resolve_vlen_string_attributes(info)
            else:
                _replace_vlen_string_references(info)
            self.object_header_cache[address] = info
            logger.debug(
                "loaded HDF5 object header for %s at %d: dataset=%s group=%s attrs=%d",
                self.source_url,
                address,
                info.is_dataset,
                info.is_group,
                len(info.attributes),
            )
        return {
            address: self.object_header_cache[address]
            for address in addresses
            if address in self.object_header_cache
        }

    async def _describe_general_table(self) -> dict[str, _DatasetDescriptor]:
        descriptors: dict[str, _DatasetDescriptor] = {}
        try:
            general_group = await self.resolve_group("/general")
            general_entries = await self.enumerate_group(general_group)
        except _TableNotFoundError:
            raise
        descriptors.update(
            await self.describe_group_members(
                parent_path="/general", entries=general_entries
            )
        )
        root_paths = (
            "session_start_time",
            "session_description",
            "identifier",
            "timestamps_reference_time",
            "file_create_date",
        )
        descriptors.update(
            await self._describe_named_child_datasets(
                parent_path="/",
                names=root_paths,
            )
        )
        try:
            metadata_group = await self.resolve_group("/general/metadata")
        except _TableNotFoundError:
            metadata_group = None
        if metadata_group is not None:
            metadata_entries = await self.enumerate_group(metadata_group)
            metadata_descriptors = await self.describe_group_members(
                parent_path="/general/metadata",
                entries=metadata_entries,
            )
            for name, descriptor in metadata_descriptors.items():
                descriptors.setdefault(name, descriptor)
        return descriptors

    async def _describe_named_child_datasets(
        self,
        *,
        parent_path: str,
        names: Iterable[str],
    ) -> dict[str, _DatasetDescriptor]:
        parent = await self.resolve_group(parent_path)
        members = await self.enumerate_group(parent)
        selected = {
            name: members[name]
            for name in names
            if name in members
            and self._is_defined_address(members[name].object_header_address)
        }
        return await self.describe_group_members(
            parent_path=parent_path, entries=selected
        )

    def _descriptor_from_header(
        self,
        *,
        name: str,
        path: str,
        header: _ObjectHeaderInfo,
    ) -> _DatasetDescriptor:
        return _DatasetDescriptor(
            name=name,
            path=path,
            shape=header.dataspace or (),
            datatype=header.datatype or _H5Datatype(kind="opaque", size=0),
            attributes=dict(header.attributes),
            layout=header.layout,
            fill_value=header.fill_value,
            filters=header.filters,
        )

    def _is_defined_address(self, address: int) -> bool:
        offset_size = self.superblock.offset_size if self.superblock is not None else 8
        return address != (1 << (offset_size * 8)) - 1

    def _absolute(self, relative_address: int) -> int:
        if self.superblock is None:
            raise AssertionError(
                "bootstrap() must be called before address translation"
            )
        return self.superblock.base_address + relative_address

    def _available_size(self, requested_size: int) -> int:
        if self.content_length is None:
            return requested_size
        return min(requested_size, self.content_length)

    def _locate_superblock(self, data: bytes) -> int | None:
        for candidate in _candidate_superblock_offsets(len(data)):
            if data[candidate : candidate + len(_HDF5_SIGNATURE)] == _HDF5_SIGNATURE:
                return candidate
        return None

    async def _read_superblock_probe(self) -> tuple[bytes, int | None]:
        await self.window_cache.prefetch([(0, self.bootstrap_bytes)])
        bootstrap_bytes = self._available_size(self.bootstrap_bytes)
        boot = await self.window_cache.read(0, bootstrap_bytes)
        superblock_offset = self._locate_superblock(boot)
        if (
            superblock_offset is None
            and self.bootstrap_bytes < self.max_bootstrap_bytes
        ):
            logger.debug(
                "HDF5 superblock not found in first %d bytes for %s; extending to %d",
                self.bootstrap_bytes,
                self.source_url,
                self.max_bootstrap_bytes,
            )
            await self.window_cache.prefetch(
                [(self.bootstrap_bytes, self.max_bootstrap_bytes)]
            )
            max_bootstrap_bytes = self._available_size(self.max_bootstrap_bytes)
            boot = await self.window_cache.read(0, max_bootstrap_bytes)
            superblock_offset = self._locate_superblock(boot)
        self.is_hdf5 = superblock_offset is not None
        return boot, superblock_offset

    def _symbol_table_node_size(self) -> int:
        assert self.superblock is not None
        entry_size = (2 * self.superblock.offset_size) + 24
        return 8 + (2 * self.superblock.group_leaf_k * entry_size)

    def _btree_node_size(self) -> int:
        assert self.superblock is not None
        key_size = self.superblock.length_size
        child_size = self.superblock.offset_size
        return (
            24
            + ((2 * self.superblock.group_internal_k) * (key_size + child_size))
            + key_size
        )

    async def _entry_to_group_handle(
        self, path: str, entry: _GroupEntry
    ) -> _GroupHandle:
        if entry.old_group is not None:
            return _GroupHandle(
                path=path,
                object_header_address=entry.object_header_address,
                old_group=entry.old_group,
            )
        header = await self.load_object_headers([entry.object_header_address])
        info = header[entry.object_header_address]
        if info.old_group is None:
            raise _TableNotFoundError(f"{path!r} is not an old-style group")
        return _GroupHandle(
            path=path,
            object_header_address=entry.object_header_address,
            old_group=info.old_group,
        )

    def _parse_group_btree_node(
        self,
        raw_node: bytes,
        node_address: int,
    ) -> tuple[int, list[int]]:
        if raw_node[:4] != b"TREE":
            raise ValueError(f"expected TREE at {node_address}, found {raw_node[:4]!r}")
        node_type = raw_node[4]
        node_level = raw_node[5]
        entries_used = _u(raw_node, 6, 2)
        if node_type != 0:
            raise NotImplementedError(f"unsupported v1 B-tree node type {node_type}")
        assert self.superblock is not None
        offset = 24
        children: list[int] = []
        for _ in range(entries_used):
            offset += self.superblock.length_size
            child_address = _u(raw_node, offset, self.superblock.offset_size)
            offset += self.superblock.offset_size
            children.append(child_address)
        return node_level, children

    async def _collect_group_symbol_nodes(self, btree_address: int) -> list[int]:
        node_size = self._btree_node_size()
        pending_addresses = [btree_address]
        leaf_node_addresses: list[int] = []
        while pending_addresses:
            await self.window_cache.prefetch(
                [
                    (self._absolute(address), self._absolute(address) + node_size)
                    for address in pending_addresses
                ]
            )
            next_addresses: list[int] = []
            for node_address in pending_addresses:
                raw_node = await self.window_cache.read(
                    self._absolute(node_address),
                    node_size,
                )
                node_level, children = self._parse_group_btree_node(
                    raw_node,
                    node_address,
                )
                if node_level == 0:
                    leaf_node_addresses.extend(children)
                else:
                    next_addresses.extend(children)
            pending_addresses = next_addresses
        return leaf_node_addresses

    async def _read_local_heap_data(self, heap_address: int) -> bytes:
        if heap_address in self.local_heap_cache:
            return self.local_heap_cache[heap_address]
        assert self.superblock is not None
        header_size = (
            8 + (2 * self.superblock.length_size) + self.superblock.offset_size
        )
        header = await self.window_cache.read(self._absolute(heap_address), header_size)
        if header[:4] != b"HEAP":
            raise ValueError(f"expected HEAP at {heap_address}, found {header[:4]!r}")
        data_segment_size = _u(header, 8, self.superblock.length_size)
        data_segment_address = _u(
            header,
            8 + (2 * self.superblock.length_size),
            self.superblock.offset_size,
        )
        await self.window_cache.prefetch(
            [
                (
                    self._absolute(data_segment_address),
                    self._absolute(data_segment_address) + data_segment_size,
                )
            ]
        )
        payload = await self.window_cache.read(
            self._absolute(data_segment_address),
            data_segment_size,
        )
        self.local_heap_cache[heap_address] = payload
        return payload

    def _parse_symbol_table_node(
        self,
        data: bytes,
        heap_data: bytes,
    ) -> list[_GroupEntry]:
        if data[:4] != b"SNOD":
            raise ValueError(f"expected SNOD, found {data[:4]!r}")
        assert self.superblock is not None
        entries_used = _u(data, 6, 2)
        offset = 8
        entries: list[_GroupEntry] = []
        entry_size = (2 * self.superblock.offset_size) + 24
        for _ in range(entries_used):
            raw_entry = data[offset : offset + entry_size]
            entries.append(self._parse_symbol_table_entry(raw_entry, heap_data))
            offset += entry_size
        return entries

    def _parse_symbol_table_entry(
        self,
        data: bytes,
        heap_data: bytes | None,
        offset_size: int | None = None,
    ) -> _GroupEntry:
        if offset_size is None:
            assert self.superblock is not None
            offset_size = self.superblock.offset_size
        name_offset = _u(data, 0, offset_size)
        object_header_address = _u(data, offset_size, offset_size)
        cache_type = _u(data, 2 * offset_size, 4)
        scratch = data[(2 * offset_size) + 8 : (2 * offset_size) + 24]
        old_group = None
        if cache_type == 1:
            old_group = _OldGroupPointers(
                btree_address=_u(scratch, 0, offset_size),
                heap_address=_u(scratch, offset_size, offset_size),
            )
        name = "" if heap_data is None else _decode_heap_string(heap_data[name_offset:])
        return _GroupEntry(
            name=name,
            object_header_address=object_header_address,
            old_group=old_group,
        )

    def _parse_v1_object_header_initial(
        self,
        address: int,
        data: bytes,
    ) -> tuple[_ObjectHeaderInfo, set[_ContinuationSpan], list[_ContinuationSpan]]:
        if not data or data[0] != 1:
            raise NotImplementedError("only version 1 object headers are supported")
        header_size = _u(data, 8, 4)
        total_size = 16 + header_size
        if total_size > len(data):
            raise NotImplementedError(
                f"object header {address} exceeds bootstrap window "
                f"({total_size} > {len(data)})"
            )
        info = _ObjectHeaderInfo(address=address)
        pending = self._apply_v1_message_block(info, data[16:total_size])
        seen = set(pending)
        return info, seen, pending

    def _apply_v1_message_block(  # noqa: C901
        self,
        info: _ObjectHeaderInfo,
        block: bytes,
    ) -> list[_ContinuationSpan]:
        assert self.superblock is not None
        continuation_spans: list[_ContinuationSpan] = []
        offset = 0
        while offset + 8 <= len(block):
            message_type = _u(block, offset, 2)
            message_size = _u(block, offset + 2, 2)
            payload = block[offset + 8 : offset + 8 + message_size]
            if len(payload) < message_size:
                break
            if message_type == 0x0001:
                info.dataspace = _parse_dataspace(payload, self.superblock.length_size)
            elif message_type == 0x0003:
                datatype, _ = _parse_datatype(payload)
                info.datatype = datatype
            elif message_type == 0x0005:
                info.fill_value = _parse_fill_value(payload)
            elif message_type == 0x0008:
                info.layout = _parse_data_layout(
                    payload,
                    offset_size=self.superblock.offset_size,
                    length_size=self.superblock.length_size,
                )
            elif message_type == 0x000B:
                info.filters = _parse_filter_pipeline(payload)
            elif message_type == 0x000C:
                try:
                    name = _parse_attribute_name(payload)
                    if name == _TABLE_MARKER_ATTRIBUTE_NAME:
                        info.has_colnames = True
                    _, value = _parse_attribute(
                        payload,
                        self.superblock.length_size,
                        offset_size=self.superblock.offset_size,
                        wanted_names=_PARSED_ATTRIBUTE_VALUE_NAMES,
                    )
                    if value is not _ATTRIBUTE_SKIPPED:
                        info.attributes[name] = value
                except Exception as exc:
                    logger.debug(
                        "skipping undecodable HDF5 attribute at object header %s: %s",
                        info.address,
                        exc,
                    )
            elif message_type == 0x0010:
                continuation_spans.append(
                    _ContinuationSpan(
                        address=_u(payload, 0, self.superblock.offset_size),
                        length=_u(
                            payload,
                            self.superblock.offset_size,
                            self.superblock.length_size,
                        ),
                    )
                )
            elif message_type == 0x0011:
                info.old_group = _OldGroupPointers(
                    btree_address=_u(payload, 0, self.superblock.offset_size),
                    heap_address=_u(
                        payload,
                        self.superblock.offset_size,
                        self.superblock.offset_size,
                    ),
                )
            offset += 8 + _align8(message_size)
        return continuation_spans

    async def _resolve_vlen_string_attributes(self, info: _ObjectHeaderInfo) -> None:
        for name, value in tuple(info.attributes.items()):
            try:
                if isinstance(value, _GlobalHeapReference):
                    info.attributes[name] = await self._read_global_heap_string(value)
                elif _is_global_heap_reference_sequence(value):
                    info.attributes[name] = [
                        await self._read_global_heap_string(reference)
                        for reference in value
                    ]
            except Exception as exc:
                logger.debug(
                    "skipping vlen string dereference for attribute %s at object header %s: %s",
                    name,
                    info.address,
                    exc,
                )
                info.attributes[name] = "<vlen-string>"

    async def _read_global_heap_string(self, reference: _GlobalHeapReference) -> str:
        objects = await self._read_global_heap_collection(reference.address)
        payload = objects.get(reference.index)
        if payload is None:
            raise KeyError(
                f"global heap object {reference.index} not found at {reference.address}"
            )
        if reference.length:
            payload = payload[: reference.length]
        return _strip_nul_bytes(payload).decode("utf-8", errors="replace")

    async def _read_global_heap_collection(self, address: int) -> dict[int, bytes]:
        if address in self.global_heap_cache:
            return self.global_heap_cache[address]
        assert self.superblock is not None
        header_size = 8 + self.superblock.length_size
        header = await self.window_cache.read(self._absolute(address), header_size)
        if header[:4] != b"GCOL":
            raise ValueError(f"expected GCOL at {address}, found {header[:4]!r}")
        collection_size = _u(header, 8, self.superblock.length_size)
        await self.window_cache.prefetch(
            [(self._absolute(address), self._absolute(address) + collection_size)]
        )
        payload = await self.window_cache.read(
            self._absolute(address),
            collection_size,
        )
        objects = _parse_global_heap_collection(
            payload,
            length_size=self.superblock.length_size,
        )
        self.global_heap_cache[address] = objects
        return objects


def _column_schema_from_descriptor(
    *,
    descriptor: _DatasetDescriptor,
    all_column_names: tuple[str, ...],
    is_metadata_table: bool,
    is_timeseries: bool,
    is_timeseries_with_rate: bool,
    source_identity: catalog_models._SourceIdentity,
    table_path: str,
    timeseries_len: int | None,
) -> catalog_models._TableColumnSchema:
    storage_facts = _storage_facts_from_descriptor(descriptor)
    is_nominally_indexed = lazynwb.table_metadata._is_nominally_indexed_column(
        descriptor.name,
        all_column_names,
    )
    is_index_column = is_nominally_indexed and descriptor.name.endswith("_index")
    index_column_name = lazynwb.table_metadata._get_index_column_name(
        descriptor.name,
        all_column_names,
    )
    data_column_name = lazynwb.table_metadata._get_data_column_name(
        descriptor.name,
        all_column_names,
    )
    is_timeseries_length_aligned = lazynwb.table_metadata._is_timeseries_length_aligned(
        is_timeseries=is_timeseries,
        shape=descriptor.shape,
        timeseries_len=timeseries_len,
    )
    dataset = catalog_models._DatasetSchema(
        path=descriptor.path,
        dtype=descriptor.datatype.to_neutral_dtype(),
        shape=descriptor.shape,
        ndim=descriptor.ndim,
        maxshape=None,
        chunks=storage_facts["chunks"],
        storage_layout=storage_facts["storage_layout"],
        compression=storage_facts["compression"],
        compression_opts=storage_facts["compression_opts"],
        filters_json=storage_facts["filters"],
        fill_value=catalog_models._to_json_value(descriptor.fill_value),
        read_capabilities=storage_facts["read_capabilities"],
        attrs_json=catalog_models._attrs_to_tuple(descriptor.attributes),
        is_group=False,
        is_dataset=True,
    )
    if is_metadata_table and descriptor.ndim <= 1:
        logger.debug(
            "using metadata-only schema facts for tiny metadata column %r at %s/%s "
            "(shape=%s dtype=%s)",
            descriptor.name,
            source_identity.source_url,
            table_path,
            descriptor.shape,
            descriptor.datatype.kind,
        )
    if (
        descriptor.name == "starting_time"
        and is_timeseries_with_rate
        and "rate" in descriptor.attributes
    ):
        logger.debug(
            "resolved selected TimeSeries rate attr for %s/%s from %r: %r",
            source_identity.source_url,
            table_path,
            descriptor.name,
            descriptor.attributes["rate"],
        )
    return catalog_models._TableColumnSchema(
        name=descriptor.name,
        table_path=table_path,
        source_path=source_identity.source_url,
        backend="hdf5",
        dataset=dataset,
        is_metadata_table=is_metadata_table,
        is_timeseries=is_timeseries,
        is_timeseries_with_rate=is_timeseries_with_rate,
        is_timeseries_length_aligned=is_timeseries_length_aligned,
        is_nominally_indexed=is_nominally_indexed,
        is_index_column=is_index_column,
        is_multidimensional=descriptor.ndim > 1,
        index_column_name=index_column_name,
        data_column_name=data_column_name,
        row_element_shape=_get_row_element_shape(
            shape=descriptor.shape,
            ndim=descriptor.ndim,
            is_index_column=is_index_column,
        ),
    )


def _storage_facts_from_descriptor(descriptor: _DatasetDescriptor) -> dict[str, Any]:
    layout = descriptor.layout or {}
    storage_layout = layout.get("kind")
    chunks = layout.get("chunk_shape") if storage_layout == "chunked" else None
    if isinstance(chunks, list):
        chunks = tuple(int(item) for item in chunks)
    compression, compression_opts = _compression_facts_from_filter_ids(
        descriptor.filters
    )
    filters: list[Any] = []
    if compression is not None:
        filters.append(
            {
                "id": "compression",
                "name": compression,
                "options": compression_opts,
            }
        )
    for filter_id in descriptor.filters:
        if filter_id == 1:
            continue
        filters.append({"id": filter_id})
    read_capabilities = ["metadata", "shape", "dtype", "slice"]
    if descriptor.ndim == 0:
        read_capabilities.append("scalar")
    if chunks is not None:
        read_capabilities.append("chunked")
    if filters:
        read_capabilities.append("filtered")
    logger.debug(
        "HDF5 parser storage facts for %s: layout=%s chunks=%s filters=%s",
        descriptor.path,
        storage_layout,
        chunks,
        filters,
    )
    return {
        "chunks": chunks,
        "storage_layout": storage_layout,
        "compression": compression,
        "compression_opts": compression_opts,
        "filters": tuple(catalog_models._to_json_value(item) for item in filters),
        "read_capabilities": tuple(read_capabilities),
    }


def _compression_facts_from_filter_ids(
    filter_ids: tuple[int, ...],
) -> tuple[str | None, Any]:
    if 1 in filter_ids:
        return "gzip", None
    if 32001 in filter_ids:
        return "blosc", None
    if 32015 in filter_ids:
        return "zstd", None
    return None, None


def _parse_dataspace(data: bytes, length_size: int) -> tuple[int, ...]:
    version = data[0]
    if version == 1:
        rank = data[1]
        offset = 8
        return tuple(
            _u(data, offset + (index * length_size), length_size)
            for index in range(rank)
        )
    if version == 2:
        rank = data[1]
        type_code = data[3]
        if type_code in {0, 2}:
            return ()
        offset = 4
        return tuple(
            _u(data, offset + (index * length_size), length_size)
            for index in range(rank)
        )
    raise NotImplementedError(f"unsupported dataspace message version {version}")


def _parse_datatype(  # noqa: C901
    data: bytes, offset: int = 0
) -> tuple[_H5Datatype, int]:
    class_and_version = data[offset]
    version = class_and_version >> 4
    class_id = class_and_version & 0x0F
    class_bits = int.from_bytes(data[offset + 1 : offset + 4], "little")
    size = _u(data, offset + 4, 4)
    props = data[offset + 8 :]
    if class_id == 0:
        signed = bool(class_bits & (1 << 3))
        return _H5Datatype(kind="int", size=size, signed=signed), 16
    if class_id == 1:
        return _H5Datatype(kind="float", size=size), 24
    if class_id == 3:
        return _H5Datatype(kind="string", size=size), 8
    if class_id == 6:
        return _H5Datatype(kind="compound", size=size), len(data) - offset
    if class_id == 7:
        return _H5Datatype(kind="reference", size=size), 8
    if class_id == 8:
        member_count = class_bits & 0xFFFF
        base, consumed = _parse_datatype(props, 0)
        upper_props = props.upper()
        if b"FALSE" in upper_props and b"TRUE" in upper_props:
            return _H5Datatype(kind="bool", size=size, base=base), len(data) - offset
        cursor = consumed
        names: list[str] = []
        if version <= 2:
            for _ in range(member_count):
                end = _find_nul(props, cursor)
                name = bytes(props[cursor:end]).decode("utf-8", errors="replace")
                cursor = _align8(end + 1)
                names.append(name)
        else:
            for _ in range(member_count):
                end = _find_nul(props, cursor)
                name = bytes(props[cursor:end]).decode("utf-8", errors="replace")
                cursor = end + 1
                names.append(name)
        cursor += member_count * base.size
        lowered = {name.casefold() for name in names}
        if lowered == {"false", "true"}:
            return (
                _H5Datatype(
                    kind="bool",
                    size=size,
                    base=base,
                    enum_names=tuple(names),
                ),
                8 + cursor,
            )
        return (
            _H5Datatype(
                kind="enum",
                size=size,
                base=base,
                enum_names=tuple(names),
            ),
            8 + cursor,
        )
    if class_id == 9:
        subtype = class_bits & 0x0F
        if subtype == 1:
            return _H5Datatype(kind="vlen-string", size=size), len(data) - offset
        base, consumed = _parse_datatype(props, 0)
        return _H5Datatype(kind="array", size=size, base=base), 8 + consumed
    if class_id == 10:
        if version == 2:
            rank = props[0]
            cursor = 2
            dims = tuple(_u(props, cursor + (index * 4), 4) for index in range(rank))
            cursor += rank * 8
            base, consumed = _parse_datatype(props, cursor)
            return (
                _H5Datatype(
                    kind="array",
                    size=size,
                    base=base,
                    shape=dims,
                ),
                8 + cursor + consumed,
            )
        if version == 3:
            rank = props[0]
            cursor = 4
            dims = tuple(_u(props, cursor + (index * 4), 4) for index in range(rank))
            cursor += rank * 4
            base, consumed = _parse_datatype(props, cursor)
            return (
                _H5Datatype(
                    kind="array",
                    size=size,
                    base=base,
                    shape=dims,
                ),
                8 + cursor + consumed,
            )
    raise NotImplementedError(
        f"unsupported datatype class={class_id} version={version}"
    )


def _parse_attribute(
    data: bytes,
    length_size: int,
    *,
    offset_size: int,
    wanted_names: frozenset[str] | None = None,
) -> tuple[str, Any]:
    version = data[0]
    if version not in (1, 2):
        raise NotImplementedError(f"unsupported attribute message version {version}")
    name_size = _u(data, 2, 2)
    datatype_size = _u(data, 4, 2)
    dataspace_size = _u(data, 6, 2)
    cursor = 8
    name = _strip_nul_bytes(data[cursor : cursor + name_size]).decode(
        "utf-8",
        errors="replace",
    )
    if wanted_names is not None and name not in wanted_names:
        return name, _ATTRIBUTE_SKIPPED
    cursor += _align8(name_size) if version == 1 else name_size
    datatype, _ = _parse_datatype(data[cursor : cursor + datatype_size])
    cursor += _align8(datatype_size) if version == 1 else datatype_size
    dataspace = _parse_dataspace(data[cursor : cursor + dataspace_size], length_size)
    cursor += _align8(dataspace_size) if version == 1 else dataspace_size
    value = _decode_attribute_payload(
        datatype,
        dataspace,
        data[cursor:],
        offset_size=offset_size,
    )
    return name, value


def _parse_attribute_name(data: bytes) -> str:
    version = data[0]
    if version not in (1, 2):
        raise NotImplementedError(f"unsupported attribute message version {version}")
    name_size = _u(data, 2, 2)
    return _strip_nul_bytes(data[8 : 8 + name_size]).decode("utf-8", errors="replace")


def _parse_global_heap_collection(data: bytes, *, length_size: int) -> dict[int, bytes]:
    if data[:4] != b"GCOL":
        raise ValueError(f"expected GCOL, found {data[:4]!r}")
    collection_size = min(_u(data, 8, length_size), len(data))
    offset = 8 + length_size
    objects: dict[int, bytes] = {}
    while offset + 8 + length_size <= collection_size:
        object_index = _u(data, offset, 2)
        object_size = _u(data, offset + 8, length_size)
        offset += 8 + length_size
        if object_index == 0 and object_size == 0:
            break
        if object_index == 0:
            offset += _align8(object_size)
            continue
        if offset + object_size > collection_size:
            raise ValueError(
                f"global heap object {object_index} extends past collection"
            )
        objects[object_index] = bytes(data[offset : offset + object_size])
        offset += _align8(object_size)
    return objects


def _parse_fill_value(data: bytes) -> dict[str, Any]:
    version = data[0]
    if version in (1, 2):
        defined = len(data) > 3 and data[3] != 0
        size = _u(data, 4, 4) if defined and len(data) >= 8 else 0
        return {"version": version, "defined": defined, "size": size}
    if version == 3:
        flags = data[1] if len(data) > 1 else 0
        defined = bool(flags & 0b0010_0000)
        size = _u(data, 2, 4) if defined and len(data) >= 6 else 0
        return {"version": version, "defined": defined, "size": size}
    return {"version": version}


def _parse_data_layout(
    data: bytes,
    *,
    offset_size: int,
    length_size: int,
) -> dict[str, Any]:
    version = data[0]
    layout_class = data[1] if len(data) > 1 else -1
    kind = {0: "compact", 1: "contiguous", 2: "chunked", 3: "virtual"}.get(
        layout_class,
        f"unknown-{layout_class}",
    )
    payload: dict[str, Any] = {"version": version, "kind": kind}
    if kind == "contiguous" and len(data) >= 2 + offset_size + length_size:
        payload["address"] = _u(data, 2, offset_size)
        payload["storage_size"] = _u(data, 2 + offset_size, length_size)
    elif kind == "compact" and len(data) >= 4:
        storage_size = _u(data, 2, 2)
        payload["storage_size"] = storage_size
        payload["data"] = bytes(data[4 : 4 + storage_size])
    elif kind == "chunked" and version in (3, 4) and len(data) >= 4 + offset_size:
        rank = data[2]
        chunk_address_offset = 4
        payload["address"] = _u(data, chunk_address_offset, offset_size)
        dims_offset = chunk_address_offset + offset_size
        if len(data) >= dims_offset + (rank * 4):
            payload["chunk_shape"] = tuple(
                _u(data, dims_offset + (index * 4), 4) for index in range(rank)
            )
    return payload


def _parse_filter_pipeline(data: bytes) -> tuple[int, ...]:
    version = data[0]
    filter_count = data[1] if len(data) > 1 else 0
    offset = 8 if version == 1 else 2
    filter_ids: list[int] = []
    for _ in range(filter_count):
        if version == 1:
            if offset + 8 > len(data):
                break
            filter_id = _u(data, offset, 2)
            name_size = _u(data, offset + 2, 2)
            client_value_count = _u(data, offset + 6, 2)
            offset += 8
            if filter_id >= 256:
                offset += _align8(name_size)
        else:
            if offset + 6 > len(data):
                break
            filter_id = _u(data, offset, 2)
            client_value_count = _u(data, offset + 4, 2)
            offset += 6
        offset += client_value_count * 4
        offset = _align8(offset)
        filter_ids.append(filter_id)
    return tuple(filter_ids)


def _decode_attribute_payload(
    datatype: _H5Datatype,
    shape: tuple[int, ...],
    payload: bytes,
    *,
    offset_size: int,
) -> object:
    count = 1 if not shape else math.prod(shape)
    if datatype.kind == "string":
        if count == 1:
            return _strip_nul_bytes(payload[: datatype.size]).decode(
                "utf-8", errors="replace"
            )
        return [
            _strip_nul_bytes(
                payload[index * datatype.size : (index + 1) * datatype.size]
            ).decode("utf-8", errors="replace")
            for index in range(count)
        ]
    if datatype.kind == "vlen-string":
        references = _parse_vlen_string_references(
            payload,
            count=count,
            offset_size=offset_size,
        )
        return references[0] if count == 1 else references
    if datatype.kind == "bool":
        return bool(payload[0])
    if datatype.kind == "int":
        values = [
            int.from_bytes(
                payload[index * datatype.size : (index + 1) * datatype.size],
                "little",
                signed=bool(datatype.signed),
            )
            for index in range(count)
        ]
        return values[0] if count == 1 else values
    if datatype.kind == "float":
        if datatype.size == 4:
            fmt = "<f"
        elif datatype.size == 8:
            fmt = "<d"
        else:
            raise NotImplementedError(f"unsupported float size {datatype.size}")
        values = [
            struct.unpack(
                fmt,
                payload[index * datatype.size : (index + 1) * datatype.size],
            )[0]
            for index in range(count)
        ]
        return values[0] if count == 1 else values
    if datatype.kind == "reference":
        return bytes(payload[: datatype.size]).hex()
    return bytes(payload)


def _parse_vlen_string_references(
    payload: bytes,
    *,
    count: int,
    offset_size: int,
) -> tuple[_GlobalHeapReference, ...]:
    stride = 4 + offset_size + 4
    references: list[_GlobalHeapReference] = []
    for index in range(count):
        cursor = index * stride
        if cursor + stride > len(payload):
            raise ValueError(
                f"need {stride} bytes for vlen string reference {index}, "
                f"only {len(payload) - cursor} available"
            )
        length = _u(payload, cursor, 4)
        address = _u(payload, cursor + 4, offset_size)
        heap_index = _u(payload, cursor + 4 + offset_size, 4)
        references.append(
            _GlobalHeapReference(address=address, index=heap_index, length=length)
        )
    return tuple(references)


def _drop_known_reference_columns(
    descriptors: dict[str, _DatasetDescriptor],
) -> None:
    descriptor = descriptors.get("timeseries")
    if (
        descriptor is not None
        and descriptor.attributes.get("neurodata_type")
        == "TimeSeriesReferenceVectorData"
    ):
        logger.debug("skipping TimeSeriesReferenceVectorData column %r", "timeseries")
        descriptors.pop("timeseries", None)
        descriptors.pop("timeseries_index", None)


def _is_metadata(descriptors: Mapping[str, _DatasetDescriptor]) -> bool:
    no_multi_dim_columns = all(value.ndim <= 1 for value in descriptors.values())
    some_scalar_columns = any(value.ndim == 0 for value in descriptors.values())
    return (
        no_multi_dim_columns
        and some_scalar_columns
        and not lazynwb.table_metadata._is_timeseries_with_rate(descriptors.keys())
    )


def _get_row_element_shape(
    *,
    shape: tuple[int, ...],
    ndim: int,
    is_index_column: bool,
) -> tuple[int, ...] | None:
    if is_index_column:
        return None
    if ndim <= 1:
        return ()
    return shape[1:]


def _normalize_exact_path(path: str) -> str:
    normalized = lazynwb.utils.normalize_internal_file_path(path)
    if normalized in {"", "."}:
        raise ValueError("table path must not be empty")
    return normalized


def _normalize_internal_path(path: str) -> str:
    normalized = lazynwb.utils.normalize_internal_file_path(path)
    if normalized in {"", "."}:
        return "/"
    return f"/{normalized.strip('/')}"


def _candidate_superblock_offsets(limit: int) -> list[int]:
    offsets = [0]
    probe = 512
    while probe < limit:
        offsets.append(probe)
        probe *= 2
    return offsets


def _coalesce(
    offsets: Iterable[tuple[int, int]],
    merge_gap: int,
) -> list[_Window]:
    spans = sorted((start, end) for start, end in offsets if end > start)
    if not spans:
        return []
    out: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [_Window(start=start, end=end) for start, end in out]


def _align_up(value: int, alignment: int) -> int:
    if value % alignment == 0:
        return value
    return value + (alignment - (value % alignment))


def _align_down(value: int, alignment: int) -> int:
    return value - (value % alignment)


def _align8(value: int) -> int:
    return _align_up(value, 8)


def _u(data: bytes, offset: int, size: int) -> int:
    end = offset + size
    if end > len(data):
        raise ValueError(
            f"need {size} bytes at offset {offset}, only {len(data) - offset} available"
        )
    return int.from_bytes(data[offset:end], "little", signed=False)


def _find_nul(value: bytes, start: int = 0) -> int:
    index = value.find(b"\x00", start)
    return len(value) if index < 0 else index


def _strip_nul_bytes(value: bytes) -> bytes:
    return bytes(value).split(b"\x00", 1)[0]


def _decode_heap_string(value: bytes) -> str:
    return _strip_nul_bytes(value).decode("utf-8", errors="replace")


def _is_global_heap_reference_sequence(value: object) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, _GlobalHeapReference) for item in value
    )


def _replace_vlen_string_references(info: _ObjectHeaderInfo) -> None:
    for name, value in tuple(info.attributes.items()):
        if isinstance(
            value, _GlobalHeapReference
        ) or _is_global_heap_reference_sequence(value):
            info.attributes[name] = "<vlen-string>"


def _byte_order_from_numpy_dtype(numpy_dtype: str | None) -> str | None:
    if not numpy_dtype:
        return None
    return numpy_dtype[0]


def _encode_old_group(value: _OldGroupPointers | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "btree_address": value.btree_address,
        "heap_address": value.heap_address,
    }


def _decode_old_group(payload: object) -> _OldGroupPointers | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("old_group must be an object")
    return _OldGroupPointers(
        btree_address=int(payload["btree_address"]),
        heap_address=int(payload["heap_address"]),
    )


def _encode_group_handle(value: _GroupHandle) -> dict[str, Any]:
    return {
        "path": value.path,
        "object_header_address": value.object_header_address,
        "old_group": _encode_old_group(value.old_group),
    }


def _decode_group_handle(payload: object) -> _GroupHandle:
    if not isinstance(payload, Mapping):
        raise ValueError("group handle must be an object")
    return _GroupHandle(
        path=str(payload["path"]),
        object_header_address=int(payload["object_header_address"]),
        old_group=_decode_old_group(payload.get("old_group")),
    )


def _encode_group_entry(value: _GroupEntry) -> dict[str, Any]:
    return {
        "name": value.name,
        "object_header_address": value.object_header_address,
        "old_group": _encode_old_group(value.old_group),
    }


def _decode_group_entry(payload: object) -> _GroupEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("group entry must be an object")
    return _GroupEntry(
        name=str(payload["name"]),
        object_header_address=int(payload["object_header_address"]),
        old_group=_decode_old_group(payload.get("old_group")),
    )


def _encode_superblock(value: _SuperblockInfo | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "version": value.version,
        "offset_size": value.offset_size,
        "length_size": value.length_size,
        "base_address": value.base_address,
        "group_leaf_k": value.group_leaf_k,
        "group_internal_k": value.group_internal_k,
        "root_group": _encode_group_handle(value.root_group),
    }


def _decode_superblock(payload: object) -> _SuperblockInfo | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("superblock must be an object")
    return _SuperblockInfo(
        version=int(payload["version"]),
        offset_size=int(payload["offset_size"]),
        length_size=int(payload["length_size"]),
        base_address=int(payload["base_address"]),
        group_leaf_k=int(payload["group_leaf_k"]),
        group_internal_k=int(payload["group_internal_k"]),
        root_group=_decode_group_handle(payload["root_group"]),
    )


def _encode_datatype(value: _H5Datatype | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "kind": value.kind,
        "size": value.size,
        "signed": value.signed,
        "base": _encode_datatype(value.base),
        "shape": list(value.shape),
        "enum_names": list(value.enum_names),
    }


def _decode_datatype(payload: object) -> _H5Datatype | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("datatype must be an object")
    return _H5Datatype(
        kind=str(payload["kind"]),
        size=int(payload["size"]),
        signed=payload.get("signed"),
        base=_decode_datatype(payload.get("base")),
        shape=tuple(int(value) for value in payload.get("shape", ())),
        enum_names=tuple(str(value) for value in payload.get("enum_names", ())),
    )


def _encode_json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, _GlobalHeapReference):
        return {
            "__global_heap_reference__": {
                "address": value.address,
                "index": value.index,
                "length": value.length,
            }
        }
    if isinstance(value, tuple):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, list):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode_json_value(item) for key, item in value.items()}
    return value


def _decode_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        if set(value) == {"__bytes__"}:
            return bytes.fromhex(str(value["__bytes__"]))
        if set(value) == {"__global_heap_reference__"}:
            reference = value["__global_heap_reference__"]
            if not isinstance(reference, Mapping):
                raise ValueError("global heap reference must be an object")
            return _GlobalHeapReference(
                address=int(reference["address"]),
                index=int(reference["index"]),
                length=int(reference["length"]),
            )
        return {key: _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    return value


def _decode_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected integer sequence")
    return tuple(int(item) for item in value)


def _encode_object_header(value: _ObjectHeaderInfo) -> dict[str, Any]:
    return {
        "address": value.address,
        "datatype": _encode_datatype(value.datatype),
        "dataspace": None if value.dataspace is None else list(value.dataspace),
        "attributes": _encode_json_value(value.attributes),
        "old_group": _encode_old_group(value.old_group),
        "layout": _encode_json_value(value.layout),
        "fill_value": _encode_json_value(value.fill_value),
        "filters": list(value.filters),
        "has_colnames": value.has_colnames,
    }


def _decode_object_header(payload: object) -> _ObjectHeaderInfo:
    if not isinstance(payload, Mapping):
        raise ValueError("object header must be an object")
    layout = _decode_json_value(payload.get("layout"))
    if isinstance(layout, Mapping) and isinstance(layout.get("chunk_shape"), list):
        layout = dict(layout)
        layout["chunk_shape"] = tuple(int(value) for value in layout["chunk_shape"])
    return _ObjectHeaderInfo(
        address=int(payload["address"]),
        datatype=_decode_datatype(payload.get("datatype")),
        dataspace=_decode_int_tuple(payload.get("dataspace")),
        attributes=_decode_json_value(payload.get("attributes", {})),
        old_group=_decode_old_group(payload.get("old_group")),
        layout=layout,
        fill_value=_decode_json_value(payload.get("fill_value")),
        filters=tuple(int(value) for value in payload.get("filters", ())),
        has_colnames=bool(payload.get("has_colnames", False)),
    )
