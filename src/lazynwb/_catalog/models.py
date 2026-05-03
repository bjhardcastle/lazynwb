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

    @classmethod
    def from_backend_dtype(cls, dtype: object | None) -> _NeutralDType:
        if dtype is None:
            return cls(kind="unknown")
        with np.errstate(all="ignore"):
            np_dtype = np.dtype(dtype)
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
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _NeutralDType:
        return cls(
            kind=str(data["kind"]),
            numpy_dtype=_optional_str(data.get("numpy_dtype")),
            byte_order=_optional_str(data.get("byte_order")),
            itemsize=_optional_int(data.get("itemsize")),
            detail=_optional_str(data.get("detail")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _DatasetSchema:
    """Accessor-free facts for one backend dataset or group."""

    path: str
    dtype: _NeutralDType
    shape: tuple[int, ...] | None
    ndim: int | None
    attrs_json: tuple[tuple[str, _JsonValue], ...] = _EMPTY_ATTRS
    is_group: bool = False
    is_dataset: bool = False

    @property
    def attrs(self) -> types.MappingProxyType:
        return types.MappingProxyType(dict(self.attrs_json))

    def to_json_dict(self) -> dict[str, _JsonValue]:
        return {
            "path": self.path,
            "dtype": self.dtype.to_json_dict(),
            "shape": list(self.shape) if self.shape is not None else None,
            "ndim": self.ndim,
            "attrs": dict(self.attrs_json),
            "is_group": self.is_group,
            "is_dataset": self.is_dataset,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _DatasetSchema:
        attrs = _attrs_to_tuple(data.get("attrs", {}))
        shape = data.get("shape")
        return cls(
            path=str(data["path"]),
            dtype=_NeutralDType.from_json_dict(_as_mapping(data["dtype"])),
            shape=tuple(int(item) for item in shape) if shape is not None else None,
            ndim=_optional_int(data.get("ndim")),
            attrs_json=attrs,
            is_group=bool(data.get("is_group", False)),
            is_dataset=bool(data.get("is_dataset", False)),
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
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> _TableColumnSchema:
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
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaSnapshot:
    """Versioned table schema/catalog snapshot for one exact table path."""

    PAYLOAD_VERSION = 1

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
    )


def _classify_numpy_dtype(np_dtype: np.dtype) -> str:
    if h5py.check_dtype(ref=np_dtype) is not None:
        return "reference"
    if h5py.check_dtype(enum=np_dtype) is not None:
        return "enum"
    if h5py.check_string_dtype(np_dtype) is not None:
        return "string"
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
