from __future__ import annotations

import dataclasses
import datetime
import logging
import pathlib
import types
from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np

import lazynwb.table_metadata

logger = logging.getLogger(__name__)

_JsonValue = Any
_EMPTY_ATTRS: tuple[tuple[str, _JsonValue], ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceIdentity:
    """Stable identity facts for one NWB storage object."""

    source_url: str
    resolved_url: str | None = None
    content_length: int | None = None
    version_id: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    in_process_token: str | None = None

    @property
    def validator_kind(self) -> str:
        if self.version_id:
            return "version_id"
        if self.etag and not self.etag.lower().startswith("w/"):
            return "etag"
        if self.last_modified is not None and self.content_length is not None:
            return "last_modified_content_length"
        if self.in_process_token is not None:
            return "in_process"
        return "none"

    @property
    def validator_value(self) -> str | None:
        if self.validator_kind == "version_id":
            return self.version_id
        if self.validator_kind == "etag":
            return self.etag
        if self.validator_kind == "last_modified_content_length":
            return f"{self.last_modified}:{self.content_length}"
        if self.validator_kind == "in_process":
            return self.in_process_token
        return None

    @property
    def is_persistent(self) -> bool:
        return self.validator_kind not in {"in_process", "none"}

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "content_length": self.content_length,
            "version_id": self.version_id,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "in_process_token": self.in_process_token,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _SourceIdentity:
        return cls(
            source_url=str(data["source_url"]),
            resolved_url=_optional_str(data.get("resolved_url")),
            content_length=_optional_int(data.get("content_length")),
            version_id=_optional_str(data.get("version_id")),
            etag=_optional_str(data.get("etag")),
            last_modified=_optional_str(data.get("last_modified")),
            in_process_token=_optional_str(data.get("in_process_token")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _NeutralDType:
    """Backend-neutral dtype facts, intentionally smaller than NumPy/h5py dtypes."""

    kind: str
    numpy_dtype: str | None = None
    byte_order: str | None = None
    itemsize: int | None = None
    detail: str | None = None
    element_numpy_dtype: str | None = None
    element_shape: tuple[int, ...] | None = None

    @classmethod
    def from_backend_dtype(cls, dtype: object | None) -> _NeutralDType:
        if dtype is None:
            return cls(kind="unknown")
        with np.errstate(all="ignore"):
            np_dtype = np.dtype(dtype)
        string_info = h5py.check_string_dtype(np_dtype)
        if string_info is not None:
            kind = "vlen_string" if string_info.length is None else "string"
            return cls(
                kind=kind,
                numpy_dtype=np_dtype.str,
                byte_order=np_dtype.byteorder,
                itemsize=np_dtype.itemsize,
                detail=f"{string_info.encoding}[{string_info.length}]",
            )
        vlen_dtype = h5py.check_dtype(vlen=np_dtype)
        if vlen_dtype is not None:
            with np.errstate(all="ignore"):
                element_dtype = np.dtype(vlen_dtype)
            return cls(
                kind="array",
                numpy_dtype=np_dtype.str,
                byte_order=np_dtype.byteorder,
                itemsize=np_dtype.itemsize,
                detail=f"vlen[{element_dtype}]",
                element_numpy_dtype=element_dtype.str,
            )
        if np_dtype.subdtype is not None:
            element_dtype, element_shape = np_dtype.subdtype
            return cls(
                kind="array",
                numpy_dtype=np_dtype.str,
                byte_order=np_dtype.byteorder,
                itemsize=np_dtype.itemsize,
                detail=str(np_dtype),
                element_numpy_dtype=element_dtype.str,
                element_shape=tuple(int(item) for item in element_shape),
            )
        return cls(
            kind=_classify_numpy_dtype(np_dtype),
            numpy_dtype=np_dtype.str,
            byte_order=np_dtype.byteorder,
            itemsize=np_dtype.itemsize,
            detail=str(np_dtype),
        )

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "kind": self.kind,
            "numpy_dtype": self.numpy_dtype,
            "byte_order": self.byte_order,
            "itemsize": self.itemsize,
            "detail": self.detail,
            "element_numpy_dtype": self.element_numpy_dtype,
            "element_shape": (
                list(self.element_shape) if self.element_shape is not None else None
            ),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _NeutralDType:
        element_shape = data.get("element_shape")
        return cls(
            kind=str(data["kind"]),
            numpy_dtype=_optional_str(data.get("numpy_dtype")),
            byte_order=_optional_str(data.get("byte_order")),
            itemsize=_optional_int(data.get("itemsize")),
            detail=_optional_str(data.get("detail")),
            element_numpy_dtype=_optional_str(data.get("element_numpy_dtype")),
            element_shape=(
                tuple(int(item) for item in element_shape)
                if element_shape is not None
                else None
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _DatasetSchema:
    """Accessor-free facts for one backend dataset or group."""

    path: str
    dtype: _NeutralDType
    shape: tuple[int, ...] | None
    ndim: int | None
    maxshape: tuple[int | None, ...] | None = None
    chunks: tuple[int, ...] | None = None
    storage_layout: str | None = None
    compression: str | None = None
    compression_opts: _JsonValue | None = None
    filters_json: tuple[_JsonValue, ...] = ()
    fill_value: _JsonValue | None = None
    read_capabilities: tuple[str, ...] = ()
    attrs_json: tuple[tuple[str, _JsonValue], ...] = _EMPTY_ATTRS
    is_group: bool = False
    is_dataset: bool = False
    hdf5_data_offset: int | None = None
    hdf5_storage_size: int | None = None

    @property
    def attrs(self) -> types.MappingProxyType:
        return types.MappingProxyType(dict(self.attrs_json))

    @property
    def filters(self) -> tuple[_JsonValue, ...]:
        return self.filters_json

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "path": self.path,
            "dtype": self.dtype.to_json_dict(),
            "shape": list(self.shape) if self.shape is not None else None,
            "ndim": self.ndim,
            "maxshape": (
                [item if item is None else int(item) for item in self.maxshape]
                if self.maxshape is not None
                else None
            ),
            "chunks": list(self.chunks) if self.chunks is not None else None,
            "storage_layout": self.storage_layout,
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "filters": list(self.filters_json),
            "fill_value": self.fill_value,
            "read_capabilities": list(self.read_capabilities),
            "attrs": dict(self.attrs_json),
            "is_group": self.is_group,
            "is_dataset": self.is_dataset,
            "hdf5_data_offset": self.hdf5_data_offset,
            "hdf5_storage_size": self.hdf5_storage_size,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _DatasetSchema:
        attrs = _attrs_to_tuple(data.get("attrs", {}))
        shape = data.get("shape")
        maxshape = data.get("maxshape")
        chunks = data.get("chunks")
        return cls(
            path=str(data["path"]),
            dtype=_NeutralDType.from_json_dict(_as_mapping(data["dtype"])),
            shape=tuple(int(item) for item in shape) if shape is not None else None,
            ndim=_optional_int(data.get("ndim")),
            maxshape=(
                tuple(None if item is None else int(item) for item in maxshape)
                if maxshape is not None
                else None
            ),
            chunks=(
                tuple(int(item) for item in chunks) if chunks is not None else None
            ),
            storage_layout=_optional_str(data.get("storage_layout")),
            compression=_optional_str(data.get("compression")),
            compression_opts=_to_json_value(data.get("compression_opts")),
            filters_json=tuple(
                _to_json_value(item) for item in data.get("filters", ())
            ),
            fill_value=_to_json_value(data.get("fill_value")),
            read_capabilities=tuple(
                str(item) for item in data.get("read_capabilities", ())
            ),
            attrs_json=attrs,
            is_group=bool(data.get("is_group", False)),
            is_dataset=bool(data.get("is_dataset", False)),
            hdf5_data_offset=_optional_int(data.get("hdf5_data_offset")),
            hdf5_storage_size=_optional_int(data.get("hdf5_storage_size")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _TableColumnSchema:
    """Accessor-free catalog facts for one table planning column."""

    name: str
    table_path: str
    source_path: str
    backend: str
    dataset: _DatasetSchema
    is_metadata_table: bool = False
    is_timeseries: bool = False
    is_timeseries_with_rate: bool = False
    is_timeseries_length_aligned: bool = True
    is_nominally_indexed: bool = False
    is_index_column: bool = False
    is_multidimensional: bool = False
    index_column_name: str | None = None
    data_column_name: str | None = None
    row_element_shape: tuple[int, ...] | None = None

    @property
    def dtype(self) -> _NeutralDType:
        return self.dataset.dtype

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.dataset.shape

    @property
    def ndim(self) -> int | None:
        return self.dataset.ndim

    @property
    def attrs(self) -> types.MappingProxyType:
        return self.dataset.attrs

    @property
    def is_group(self) -> bool:
        return self.dataset.is_group

    @property
    def is_dataset(self) -> bool:
        return self.dataset.is_dataset

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "name": self.name,
            "table_path": self.table_path,
            "source_path": self.source_path,
            "backend": self.backend,
            "dataset": self.dataset.to_json_dict(),
            "is_metadata_table": self.is_metadata_table,
            "is_timeseries": self.is_timeseries,
            "is_timeseries_with_rate": self.is_timeseries_with_rate,
            "is_timeseries_length_aligned": self.is_timeseries_length_aligned,
            "is_nominally_indexed": self.is_nominally_indexed,
            "is_index_column": self.is_index_column,
            "is_multidimensional": self.is_multidimensional,
            "index_column_name": self.index_column_name,
            "data_column_name": self.data_column_name,
            "row_element_shape": (
                list(self.row_element_shape)
                if self.row_element_shape is not None
                else None
            ),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _TableColumnSchema:
        row_element_shape = data.get("row_element_shape")
        return cls(
            name=str(data["name"]),
            table_path=str(data["table_path"]),
            source_path=str(data["source_path"]),
            backend=str(data["backend"]),
            dataset=_DatasetSchema.from_json_dict(_as_mapping(data["dataset"])),
            is_metadata_table=bool(data.get("is_metadata_table", False)),
            is_timeseries=bool(data.get("is_timeseries", False)),
            is_timeseries_with_rate=bool(data.get("is_timeseries_with_rate", False)),
            is_timeseries_length_aligned=bool(
                data.get("is_timeseries_length_aligned", True)
            ),
            is_nominally_indexed=bool(data.get("is_nominally_indexed", False)),
            is_index_column=bool(data.get("is_index_column", False)),
            is_multidimensional=bool(data.get("is_multidimensional", False)),
            index_column_name=_optional_str(data.get("index_column_name")),
            data_column_name=_optional_str(data.get("data_column_name")),
            row_element_shape=(
                tuple(int(item) for item in row_element_shape)
                if row_element_shape is not None
                else None
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaSnapshot:
    """Versioned table schema/catalog snapshot for one exact table path."""

    PAYLOAD_VERSION = 3

    source_identity: _SourceIdentity
    table_path: str
    backend: str
    columns: tuple[_TableColumnSchema, ...]
    table_length: int | None = None
    payload_version: int = PAYLOAD_VERSION

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "payload_version": self.payload_version,
            "source_identity": self.source_identity.to_json_dict(),
            "table_path": self.table_path,
            "backend": self.backend,
            "table_length": self.table_length,
            "columns": [column.to_json_dict() for column in self.columns],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _TableSchemaSnapshot:
        return cls(
            payload_version=int(data["payload_version"]),
            source_identity=_SourceIdentity.from_json_dict(
                _as_mapping(data["source_identity"])
            ),
            table_path=str(data["table_path"]),
            backend=str(data["backend"]),
            table_length=_optional_int(data.get("table_length")),
            columns=tuple(
                _TableColumnSchema.from_json_dict(_as_mapping(column))
                for column in data.get("columns", ())
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _PathSummaryEntry:
    """Accessor-free path discovery facts for one internal HDF5/Zarr path."""

    path: str
    is_group: bool
    is_dataset: bool
    shape: tuple[int, ...] | None = None
    attrs_json: tuple[tuple[str, _JsonValue], ...] = _EMPTY_ATTRS

    @property
    def attrs(self) -> types.MappingProxyType:
        return types.MappingProxyType(dict(self.attrs_json))


def _source_identity_from_local_path(path: pathlib.Path) -> _SourceIdentity:
    stat = path.stat()
    last_modified = datetime.datetime.fromtimestamp(
        stat.st_mtime, tz=datetime.timezone.utc
    ).isoformat()
    return _SourceIdentity(
        source_url=path.as_posix(),
        content_length=stat.st_size,
        last_modified=last_modified,
    )


def _column_from_raw_metadata(
    column: lazynwb.table_metadata.RawTableColumnMetadata,
) -> _TableColumnSchema:
    accessor_path = getattr(column.accessor, "name", None)
    if not accessor_path:
        accessor_path = f"{column.table_path}/{column.name}"
    dataset = _DatasetSchema(
        path=str(accessor_path).removeprefix("/"),
        dtype=_NeutralDType.from_backend_dtype(column.dtype),
        shape=column.shape,
        ndim=column.ndim,
        maxshape=column.maxshape,
        chunks=column.chunks,
        storage_layout=column.storage_layout,
        compression=column.compression,
        compression_opts=_to_json_value(column.compression_opts),
        filters_json=tuple(_to_json_value(item) for item in column.filters),
        fill_value=_to_json_value(column.fill_value),
        read_capabilities=column.read_capabilities,
        attrs_json=_attrs_to_tuple(column.attrs),
        is_group=column.is_group,
        is_dataset=column.is_dataset,
    )
    return _TableColumnSchema(
        name=column.name,
        table_path=column.table_path,
        source_path=column.source_path,
        backend=column.backend,
        dataset=dataset,
        is_metadata_table=column.is_metadata_table,
        is_timeseries=column.is_timeseries,
        is_timeseries_with_rate=column.is_timeseries_with_rate,
        is_timeseries_length_aligned=column.is_timeseries_length_aligned,
        is_nominally_indexed=column.is_nominally_indexed,
        is_index_column=column.is_index_column,
        is_multidimensional=column.is_multidimensional,
        index_column_name=column.index_column_name,
        data_column_name=column.data_column_name,
        row_element_shape=column.row_element_shape,
    )


def _classify_numpy_dtype(np_dtype: np.dtype) -> str:
    if h5py.check_dtype(ref=np_dtype) is not None:
        return "reference"
    if h5py.check_dtype(enum=np_dtype) is not None:
        return "enum"
    if np_dtype.fields is not None:
        return "compound"
    if np_dtype.kind == "b":
        return "bool"
    if np_dtype.kind in {"i", "u", "f", "c"}:
        return "numeric"
    if np_dtype.kind in {"S", "U"}:
        return "string"
    if np_dtype.kind == "O":
        return "object"
    if np_dtype.kind == "V":
        return "opaque"
    return "unknown"


def _attrs_to_tuple(attrs: Mapping[str, object]) -> tuple[tuple[str, _JsonValue], ...]:
    return tuple(
        sorted((str(key), _to_json_value(value)) for key, value in attrs.items())
    )


def _to_json_value(value: object) -> _JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _to_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_to_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    logger.debug("serializing unsupported attr value %r as repr", type(value).__name__)
    return repr(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value
