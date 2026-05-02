from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from collections.abc import Iterable
from typing import Any

import h5py
import zarr

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
        if isinstance(file_path, lazynwb.file_io.FileAccessor)
        else lazynwb.file_io._get_accessor(file_path)
    )
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    column_accessors = _get_table_column_accessors(
        file_path=file,
        table_path=normalized_table_path,
        use_thread_pool=use_thread_pool,
        skip_references=skip_references,
    )
    column_names = tuple(column_accessors)
    is_metadata_table = normalized_table_path == "general" or _is_metadata(
        column_accessors
    )
    is_timeseries = _is_timeseries(column_names)
    is_timeseries_with_rate = _is_timeseries_with_rate(column_names)
    timeseries_len = _get_timeseries_data_length(column_accessors, is_timeseries)

    columns = tuple(
        _metadata_from_accessor(
            accessor=accessor,
            backend=file._hdmf_backend.value,
            column_names=column_names,
            is_metadata_table=is_metadata_table,
            is_timeseries=is_timeseries,
            is_timeseries_with_rate=is_timeseries_with_rate,
            name=name,
            source_path=file._path.as_posix(),
            table_path=normalized_table_path,
            timeseries_len=timeseries_len,
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


def _metadata_from_accessor(
    accessor: TableColumnAccessor,
    backend: str,
    column_names: Iterable[str],
    is_metadata_table: bool,
    is_timeseries: bool,
    is_timeseries_with_rate: bool,
    name: str,
    source_path: str,
    table_path: str,
    timeseries_len: int | None,
) -> RawTableColumnMetadata:
    is_group = lazynwb.file_io.is_group(accessor)
    dtype = getattr(accessor, "dtype", None)
    shape = _shape_as_tuple(getattr(accessor, "shape", None))
    ndim = getattr(accessor, "ndim", None)
    is_nominally_indexed = _is_nominally_indexed_column(name, column_names)
    is_index_column = is_nominally_indexed and name.endswith("_index")
    index_column_name = _get_index_column_name(name, column_names)
    data_column_name = _get_data_column_name(name, column_names)
    is_timeseries_length_aligned = _is_timeseries_length_aligned(
        is_timeseries=is_timeseries,
        shape=shape,
        timeseries_len=timeseries_len,
    )
    return RawTableColumnMetadata(
        name=name,
        table_path=table_path,
        source_path=source_path,
        backend=backend,
        dtype=dtype,
        shape=shape,
        ndim=ndim,
        attrs=dict(getattr(accessor, "attrs", {})),
        is_group=is_group,
        is_dataset=not is_group and dtype is not None,
        is_metadata_table=is_metadata_table,
        is_timeseries=is_timeseries,
        is_timeseries_with_rate=is_timeseries_with_rate,
        is_timeseries_length_aligned=is_timeseries_length_aligned,
        is_nominally_indexed=is_nominally_indexed,
        is_index_column=is_index_column,
        is_multidimensional=ndim is not None and ndim > 1,
        index_column_name=index_column_name,
        data_column_name=data_column_name,
        _accessor=accessor,
    )


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
        if isinstance(file_path, lazynwb.file_io.FileAccessor)
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
    group_keys = set(group_keys)
    return "data" in group_keys and (
        "timestamps" in group_keys or "starting_time" in group_keys
    )


def _is_timeseries_with_rate(group_keys: Iterable[str]) -> bool:
    group_keys = set(group_keys)
    return (
        "data" in group_keys
        and "starting_time" in group_keys
        and "timestamps" not in group_keys
    )


def _is_metadata(column_accessors: dict[str, TableColumnAccessor]) -> bool:
    """Check whether a group is metadata rather than a row-oriented table."""
    no_multi_dim_columns = all(
        accessor.ndim <= 1
        for accessor in column_accessors.values()
        if hasattr(accessor, "dtype")
    )
    some_scalar_columns = any(
        accessor.ndim == 0
        for accessor in column_accessors.values()
        if hasattr(accessor, "dtype")
    )
    return (
        no_multi_dim_columns
        and some_scalar_columns
        and not _is_timeseries_with_rate(column_accessors.keys())
    )


def _is_nominally_indexed_column(
    column_name: str, all_column_names: Iterable[str]
) -> bool:
    all_column_names = set(all_column_names)
    if column_name not in all_column_names:
        return False
    if column_name.endswith("_index"):
        return column_name.split("_index")[0] in all_column_names
    return f"{column_name}_index" in all_column_names


def _get_indexed_column_names(column_names: Iterable[str]) -> set[str]:
    return {
        column_name
        for column_name in column_names
        if _is_nominally_indexed_column(column_name, column_names)
    }


def _get_timeseries_data_length(
    column_accessors: dict[str, TableColumnAccessor],
    is_timeseries: bool,
) -> int | None:
    if not is_timeseries:
        return None
    data = column_accessors.get("data")
    shape = getattr(data, "shape", None)
    if not shape:
        return None
    return shape[0]


def _is_timeseries_length_aligned(
    is_timeseries: bool,
    shape: tuple[int, ...] | None,
    timeseries_len: int | None,
) -> bool:
    if not is_timeseries or timeseries_len is None or not shape:
        return True
    return shape[0] == timeseries_len


def _get_index_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    all_column_names = set(all_column_names)
    if column_name.endswith("_index"):
        return (
            column_name
            if _is_nominally_indexed_column(column_name, all_column_names)
            else None
        )
    index_column_name = f"{column_name}_index"
    if index_column_name in all_column_names:
        return index_column_name
    return None


def _get_data_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    all_column_names = set(all_column_names)
    if not column_name.endswith("_index"):
        return column_name
    data_column_name = column_name.split("_index")[0]
    if data_column_name in all_column_names:
        return data_column_name
    return None


def _shape_as_tuple(shape: Iterable[int] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(shape)
