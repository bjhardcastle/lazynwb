from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from collections.abc import Iterable
from typing import Any

import h5py
import zarr

import lazynwb._catalog._schema as catalog_schema
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)

TableColumnAccessor = h5py.Dataset | h5py.Group | zarr.Array | zarr.Group

__all__ = [
    "RawTableColumnMetadata",
    "get_table_column_metadata",
    "get_table_length_from_metadata",
]


@dataclasses.dataclass(frozen=True, slots=True)
class RawTableColumnMetadata:
    """Backend-derived facts for one raw NWB table column."""

    name: str
    table_path: str
    source_path: str
    backend: str
    dtype: Any | None
    shape: tuple[int, ...] | None
    ndim: int | None
    attrs: dict[str, Any]
    maxshape: tuple[int | None, ...] | None
    chunks: tuple[int, ...] | None
    storage_layout: str | None
    compression: str | None
    compression_opts: object | None
    filters: tuple[object, ...]
    fill_value: object | None
    read_capabilities: tuple[str, ...]
    is_group: bool
    is_dataset: bool
    is_metadata_table: bool
    is_timeseries: bool
    is_timeseries_with_rate: bool
    is_timeseries_length_aligned: bool
    is_nominally_indexed: bool
    is_index_column: bool
    is_multidimensional: bool
    index_column_name: str | None
    data_column_name: str | None
    row_element_shape: tuple[int, ...] | None
    _accessor: TableColumnAccessor = dataclasses.field(repr=False, compare=False)

    @property
    def accessor(self) -> TableColumnAccessor:
        return self._accessor


def get_table_column_metadata(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    use_thread_pool: bool = False,
    skip_references: bool = True,
) -> tuple[RawTableColumnMetadata, ...]:
    """Return raw backend metadata for every planning column in one NWB table."""
    t0 = time.time()
    file = (
        file_path
        if _is_file_accessor_like(file_path)
        else lazynwb.file_io._get_accessor(file_path)
    )
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    column_accessors = _get_table_column_accessors(
        file_path=file,
        table_path=normalized_table_path,
        use_thread_pool=use_thread_pool,
        skip_references=skip_references,
    )
    column_facts = _column_facts_from_accessors(column_accessors)
    table_rules = catalog_schema._get_table_schema_rules(
        normalized_table_path,
        column_facts.values(),
    )

    columns = tuple(
        _metadata_from_accessor(
            accessor=accessor,
            backend=file._hdmf_backend.value,
            column_facts=column_facts[name],
            name=name,
            source_path=file._path.as_posix(),
            table_path=normalized_table_path,
            table_rules=table_rules,
        )
        for name, accessor in column_accessors.items()
    )
    logger.debug(
        "derived raw metadata for %d columns from %r/%s in %.2f s",
        len(columns),
        file._path,
        normalized_table_path,
        time.time() - t0,
    )
    return columns


def get_table_length_from_metadata(
    columns: Iterable[RawTableColumnMetadata],
) -> int:
    """Compute table length from raw table metadata without reading column data."""
    columns = tuple(columns)
    table_length = catalog_schema._get_table_length(columns)
    if table_length is not None:
        return table_length

    raise lazynwb.exceptions.InternalPathError("Could not determine table length")


@dataclasses.dataclass(frozen=True, slots=True)
class _StorageFacts:
    """Backend storage facts needed later for direct read planning."""

    maxshape: tuple[int | None, ...] | None
    chunks: tuple[int, ...] | None
    storage_layout: str | None
    compression: str | None
    compression_opts: object | None
    filters: tuple[object, ...]
    fill_value: object | None
    read_capabilities: tuple[str, ...]


def _metadata_from_accessor(
    accessor: TableColumnAccessor,
    backend: str,
    column_facts: catalog_schema._ColumnFacts,
    name: str,
    source_path: str,
    table_path: str,
    table_rules: catalog_schema._TableSchemaRules,
) -> RawTableColumnMetadata:
    is_group = lazynwb.file_io.is_group(accessor)
    dtype = getattr(accessor, "dtype", None)
    shape = column_facts.shape
    ndim = column_facts.ndim
    attrs = dict(getattr(accessor, "attrs", {}))
    storage_facts = _get_storage_facts(accessor)
    column_rules = table_rules.column_rules(
        name,
        shape=shape,
        ndim=ndim,
    )
    if column_rules.is_metadata_table and ndim is not None and ndim <= 1:
        logger.debug(
            "using metadata-only schema facts for tiny metadata column %r at %s/%s "
            "(shape=%s dtype=%s)",
            name,
            source_path,
            table_path,
            shape,
            dtype,
        )
    if (
        name == "starting_time"
        and column_rules.is_timeseries_with_rate
        and "rate" in attrs
    ):
        logger.debug(
            "resolved selected TimeSeries rate attr for %s/%s from %r: %r",
            source_path,
            table_path,
            name,
            attrs["rate"],
        )
    return RawTableColumnMetadata(
        name=name,
        table_path=table_path,
        source_path=source_path,
        backend=backend,
        dtype=dtype,
        shape=shape,
        ndim=ndim,
        attrs=attrs,
        maxshape=storage_facts.maxshape,
        chunks=storage_facts.chunks,
        storage_layout=storage_facts.storage_layout,
        compression=storage_facts.compression,
        compression_opts=storage_facts.compression_opts,
        filters=storage_facts.filters,
        fill_value=storage_facts.fill_value,
        read_capabilities=storage_facts.read_capabilities,
        is_group=is_group,
        is_dataset=not is_group and dtype is not None,
        is_metadata_table=column_rules.is_metadata_table,
        is_timeseries=column_rules.is_timeseries,
        is_timeseries_with_rate=column_rules.is_timeseries_with_rate,
        is_timeseries_length_aligned=column_rules.is_timeseries_length_aligned,
        is_nominally_indexed=column_rules.is_nominally_indexed,
        is_index_column=column_rules.is_index_column,
        is_multidimensional=ndim is not None and ndim > 1,
        index_column_name=column_rules.index_column_name,
        data_column_name=column_rules.data_column_name,
        row_element_shape=column_rules.row_element_shape,
        _accessor=accessor,
    )


def _column_facts_from_accessors(
    column_accessors: dict[str, TableColumnAccessor],
) -> dict[str, catalog_schema._ColumnFacts]:
    return {
        name: catalog_schema._ColumnFacts(
            name=name,
            shape=_shape_as_tuple(getattr(accessor, "shape", None)),
            ndim=getattr(accessor, "ndim", None),
        )
        for name, accessor in column_accessors.items()
    }


def _get_storage_facts(accessor: TableColumnAccessor) -> _StorageFacts:
    maxshape = _shape_as_optional_int_tuple(getattr(accessor, "maxshape", None))
    chunks = _shape_as_tuple(getattr(accessor, "chunks", None))
    storage_layout = _get_storage_layout(accessor, chunks)
    compression, compression_opts = _get_compression_facts(accessor)
    filters = _get_filter_facts(accessor, compression, compression_opts)
    fill_value = _get_fill_value(accessor)
    read_capabilities = _get_read_capabilities(
        accessor=accessor,
        chunks=chunks,
        compression=compression,
        filters=filters,
    )
    logger.debug(
        "storage facts for %s: layout=%s chunks=%s compression=%s filters=%s",
        getattr(accessor, "name", type(accessor).__name__),
        storage_layout,
        chunks,
        compression,
        filters,
    )
    return _StorageFacts(
        maxshape=maxshape,
        chunks=chunks,
        storage_layout=storage_layout,
        compression=compression,
        compression_opts=compression_opts,
        filters=filters,
        fill_value=fill_value,
        read_capabilities=read_capabilities,
    )


def _get_storage_layout(
    accessor: TableColumnAccessor,
    chunks: tuple[int, ...] | None,
) -> str | None:
    if isinstance(accessor, h5py.Dataset):
        layout = accessor.id.get_create_plist().get_layout()
        if layout == h5py.h5d.CHUNKED:
            return "chunked"
        if layout == h5py.h5d.CONTIGUOUS:
            return "contiguous"
        if layout == h5py.h5d.COMPACT:
            return "compact"
        if layout == h5py.h5d.VIRTUAL:
            return "virtual"
        return f"hdf5:{layout}"
    if isinstance(accessor, zarr.Array):
        return "chunked" if chunks is not None else "array"
    if lazynwb.file_io.is_group(accessor):
        return "group"
    return None


def _get_compression_facts(
    accessor: TableColumnAccessor,
) -> tuple[str | None, object | None]:
    compression = getattr(accessor, "compression", None)
    compression_opts = getattr(accessor, "compression_opts", None)
    if compression is not None:
        return str(compression), compression_opts
    compressor = getattr(accessor, "compressor", None)
    if compressor is None:
        return None, None
    codec_id = getattr(compressor, "codec_id", None) or getattr(compressor, "id", None)
    compression_name = str(codec_id or type(compressor).__name__)
    get_config = getattr(compressor, "get_config", None)
    if callable(get_config):
        compression_opts = get_config()
    else:
        compression_opts = repr(compressor)
    return compression_name, compression_opts


def _get_filter_facts(
    accessor: TableColumnAccessor,
    compression: str | None,
    compression_opts: object | None,
) -> tuple[object, ...]:
    filters: list[object] = []
    if compression is not None:
        filters.append(
            {
                "id": "compression",
                "name": compression,
                "options": compression_opts,
            }
        )
    for attr_name in ("shuffle", "fletcher32", "scaleoffset"):
        value = getattr(accessor, attr_name, None)
        if value not in (None, False):
            filters.append({"id": attr_name, "value": value})
    zarr_filters = getattr(accessor, "filters", None)
    if zarr_filters:
        for zarr_filter in zarr_filters:
            get_config = getattr(zarr_filter, "get_config", None)
            filters.append(get_config() if callable(get_config) else repr(zarr_filter))
    return tuple(filters)


def _get_fill_value(accessor: TableColumnAccessor) -> object | None:
    if hasattr(accessor, "fillvalue"):
        return accessor.fillvalue
    if hasattr(accessor, "fill_value"):
        return accessor.fill_value
    return None


def _get_read_capabilities(
    accessor: TableColumnAccessor,
    chunks: tuple[int, ...] | None,
    compression: str | None,
    filters: tuple[object, ...],
) -> tuple[str, ...]:
    capabilities = ["metadata"]
    if hasattr(accessor, "dtype"):
        capabilities.extend(("shape", "dtype", "slice"))
    if getattr(accessor, "ndim", None) == 0:
        capabilities.append("scalar")
    if chunks is not None:
        capabilities.append("chunked")
    if compression is not None or filters:
        capabilities.append("filtered")
    if lazynwb.file_io.is_group(accessor):
        capabilities.append("children")
    return tuple(capabilities)


def _get_table_column_accessors(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    use_thread_pool: bool = False,
    skip_references: bool = True,
) -> dict[str, TableColumnAccessor]:
    """Get raw backend accessors for each planning column in an NWB table."""
    names_to_columns: dict[str, TableColumnAccessor] = {}
    t0 = time.time()
    file = (
        file_path
        if _is_file_accessor_like(file_path)
        else lazynwb.file_io._get_accessor(file_path)
    )
    table = file[table_path]
    if not lazynwb.file_io.is_group(table):
        raise lazynwb.exceptions.InternalPathError(
            f"{table_path!r} is not a group/table (found {type(table).__name__})"
        )

    if use_thread_pool:
        future_to_column = {
            lazynwb.utils.get_threadpool_executor().submit(
                table.get, column_name
            ): column_name
            for column_name in table.keys()
        }
        for future in concurrent.futures.as_completed(future_to_column):
            column_name = future_to_column[future]
            names_to_columns[column_name] = future.result()
    else:
        for column_name in table:
            names_to_columns[column_name] = table.get(column_name)

    if lazynwb.utils.normalize_internal_file_path(table_path) == "general":
        names_to_columns.update(_get_general_metadata_accessors(file))
        names_to_columns = {
            name: accessor
            for name, accessor in names_to_columns.items()
            if not lazynwb.file_io.is_group(accessor)
        }

    logger.debug(
        "retrieved %d raw column accessors from %r/%s in %.2f s (%s)",
        len(names_to_columns),
        file._path,
        table_path,
        time.time() - t0,
        f"{use_thread_pool=}",
    )

    if skip_references:
        _drop_known_reference_columns(names_to_columns)
    else:
        raise NotImplementedError(
            "Keeping references is not implemented yet: see https://pynwb.readthedocs.io/en/stable/pynwb.base.html#pynwb.base.TimeSeriesReferenceVectorData"
        )
    return names_to_columns


def _get_general_metadata_accessors(
    root: lazynwb.file_io.FileAccessor,
) -> dict[str, TableColumnAccessor]:
    names_to_columns: dict[str, TableColumnAccessor] = {}
    for path in (
        "session_start_time",
        "session_description",
        "identifier",
        "timestamps_reference_time",
        "file_create_date",
    ):
        value = root.get(path)
        if value is not None:
            names_to_columns[path] = value

    for path in root.get("general/metadata", {}).keys():
        if path in names_to_columns:
            continue
        value = root.get(f"general/metadata/{path}")
        if value is not None and not lazynwb.file_io.is_group(value):
            names_to_columns[path] = value

    return {
        name: accessor
        for name, accessor in names_to_columns.items()
        if not lazynwb.file_io.is_group(accessor)
    }


def _drop_known_reference_columns(
    names_to_columns: dict[str, TableColumnAccessor],
) -> None:
    known_references = {
        "timeseries": "TimeSeriesReferenceVectorData",
    }
    for name, neurodata_type in known_references.items():
        accessor = names_to_columns.get(name)
        if accessor is None or accessor.attrs.get("neurodata_type") != neurodata_type:
            continue
        logger.debug(
            "Skipping reference column %r with neurodata_type %r",
            name,
            neurodata_type,
        )
        names_to_columns.pop(name, None)
        names_to_columns.pop(f"{name}_index", None)


def _is_timeseries(group_keys: Iterable[str]) -> bool:
    return catalog_schema._is_timeseries(group_keys)


def _is_timeseries_with_rate(group_keys: Iterable[str]) -> bool:
    return catalog_schema._is_timeseries_with_rate(group_keys)


def _is_metadata(column_accessors: dict[str, TableColumnAccessor]) -> bool:
    """Check whether a group is metadata rather than a row-oriented table."""
    return catalog_schema._is_metadata(
        _column_facts_from_accessors(column_accessors).values(),
        is_timeseries_with_rate=catalog_schema._is_timeseries_with_rate(
            column_accessors.keys()
        ),
    )


def _is_nominally_indexed_column(
    column_name: str, all_column_names: Iterable[str]
) -> bool:
    return catalog_schema._is_nominally_indexed_column(column_name, all_column_names)


def _get_indexed_column_names(column_names: Iterable[str]) -> set[str]:
    return catalog_schema._get_indexed_column_names(column_names)


def _get_timeseries_data_length(
    column_accessors: dict[str, TableColumnAccessor],
    is_timeseries: bool,
) -> int | None:
    return catalog_schema._get_timeseries_data_length(
        columns=_column_facts_from_accessors(column_accessors).values(),
        is_timeseries=is_timeseries,
    )


def _is_timeseries_length_aligned(
    is_timeseries: bool,
    shape: tuple[int, ...] | None,
    timeseries_len: int | None,
) -> bool:
    return catalog_schema._is_timeseries_length_aligned(
        is_timeseries=is_timeseries,
        shape=shape,
        timeseries_len=timeseries_len,
    )


def _get_index_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    return catalog_schema._get_index_column_name(column_name, all_column_names)


def _get_data_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    return catalog_schema._get_data_column_name(column_name, all_column_names)


def _shape_as_tuple(shape: Iterable[int] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(shape)


def _shape_as_optional_int_tuple(
    shape: Iterable[int | None] | None,
) -> tuple[int | None, ...] | None:
    if shape is None:
        return None
    return tuple(None if item is None else int(item) for item in shape)


def _get_row_element_shape(
    shape: tuple[int, ...] | None,
    ndim: int | None,
    is_index_column: bool,
) -> tuple[int, ...] | None:
    return catalog_schema._get_row_element_shape(
        shape=shape,
        ndim=ndim,
        is_index_column=is_index_column,
    )


def _is_file_accessor_like(value: object) -> bool:
    return isinstance(value, lazynwb.file_io.FileAccessor) or all(
        hasattr(value, attr)
        for attr in ("_accessor", "_hdmf_backend", "_path", "get", "__getitem__")
    )
