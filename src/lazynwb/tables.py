from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import dataclasses
import difflib
import logging
import math
import time
import typing
import urllib.parse
import zlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeVar

import h5py
import numcodecs
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars._typing
import polars.datatypes.convert
import tqdm
import zarr

import lazynwb._catalog._schema as catalog_schema
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._catalog.polars as catalog_polars
import lazynwb._config as lazynwb_config
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb._zarr.reader as zarr_reader
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.table_metadata
import lazynwb.types_
import lazynwb.utils

pd.options.mode.copy_on_write = True

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame, pd.DataFrame)
ColumnMetadataType = TypeVar(
    "ColumnMetadataType",
    lazynwb.table_metadata.RawTableColumnMetadata,
    catalog_models._TableColumnSchema,
)
AsyncValueType = TypeVar("AsyncValueType")
_FastCatalogBackendName = Literal["hdf5", "zarr"]

logger = logging.getLogger(__name__)

NWB_PATH_COLUMN_NAME = "_nwb_path"
TABLE_PATH_COLUMN_NAME = "_table_path"
TABLE_INDEX_COLUMN_NAME = "_table_index"

INTERNAL_COLUMN_NAMES = {
    NWB_PATH_COLUMN_NAME,
    TABLE_PATH_COLUMN_NAME,
    TABLE_INDEX_COLUMN_NAME,
}

INTERVALS_TABLE_INDEX_COLUMN_NAME = "_intervals" + TABLE_INDEX_COLUMN_NAME
UNITS_TABLE_INDEX_COLUMN_NAME = "_units" + TABLE_INDEX_COLUMN_NAME

_INDEXED_COLUMN_FULL_READ_MIN_COVERAGE = 0.8
_INDEXED_COLUMN_MAX_COALESCE_GAP_ELEMENTS = 1_048_576
_SUPPORTED_DIRECT_HDF5_FILTER_IDS = frozenset({1, 2, 3, 32001, 32015})


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaInferenceResult:
    schema: pl.Schema
    catalog_snapshots: dict[str, catalog_models._TableSchemaSnapshot]


@dataclasses.dataclass(frozen=True, slots=True)
class _DirectHDF5IndexedColumnReadPlan:
    data_column: catalog_models._TableColumnSchema
    index_column: catalog_models._TableColumnSchema


@dataclasses.dataclass(frozen=True, slots=True)
class _DirectHDF5TableReadPlan:
    scalar_columns: tuple[catalog_models._TableColumnSchema, ...] = ()
    indexed_columns: tuple[_DirectHDF5IndexedColumnReadPlan, ...] = ()
    fallback_columns: tuple[catalog_models._TableColumnSchema, ...] = ()

    @property
    def has_direct_columns(self) -> bool:
        return bool(self.scalar_columns or self.indexed_columns)


@dataclasses.dataclass(frozen=True, slots=True)
class _DirectHDF5IndexedRangeReadPlan:
    row_indices: npt.NDArray[np.intp]
    row_starts: npt.NDArray[np.intp]
    row_ends: npt.NDArray[np.intp]
    spans: tuple[tuple[int, int], ...]
    requested_element_count: int
    spanned_element_count: int
    full_element_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class _HDF5ChunkRecord:
    address: int
    stored_size: int
    filter_mask: int
    offsets: tuple[int, ...]


class _DirectHDF5ReadError(RuntimeError):
    """Private direct-reader coverage error, shaped at the public boundary."""

    def __init__(self, column_name: str, reason: str) -> None:
        super().__init__(reason)
        self.column_name = column_name
        self.reason = reason


@typing.overload
def get_df(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    nwb_path_to_row_indices: Mapping[str, Sequence[int]] | None = None,
    exclude_array_columns: bool = True,
    parallel: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    low_memory: bool = False,
    as_polars: None = None,
    _catalog_snapshots: Mapping[
        str,
        catalog_models._TableSchemaSnapshot,
    ]
    | None = None,
    _allow_missing_columns: bool = False,
) -> pd.DataFrame | pl.DataFrame: ...


@typing.overload
def get_df(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    nwb_path_to_row_indices: Mapping[str, Sequence[int]] | None = None,
    exclude_array_columns: bool = True,
    parallel: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    low_memory: bool = False,
    as_polars: Literal[False] = False,
    _catalog_snapshots: Mapping[
        str,
        catalog_models._TableSchemaSnapshot,
    ]
    | None = None,
    _allow_missing_columns: bool = False,
) -> pd.DataFrame: ...


@typing.overload
def get_df(
    nwb_data_sources: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    nwb_path_to_row_indices: Mapping[str, Sequence[int]] | None = None,
    exclude_array_columns: bool = True,
    parallel: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    low_memory: bool = False,
    as_polars: Literal[True] = True,
    _catalog_snapshots: Mapping[
        str,
        catalog_models._TableSchemaSnapshot,
    ]
    | None = None,
    _allow_missing_columns: bool = False,
) -> pl.DataFrame: ...


@typing.overload
def get_df(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    nwb_path_to_row_indices: Mapping[str, Sequence[int]] | None = None,
    exclude_array_columns: bool = True,
    parallel: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    low_memory: bool = False,
    as_polars: bool | None = None,
    _catalog_snapshots: Mapping[
        str,
        catalog_models._TableSchemaSnapshot,
    ]
    | None = None,
    _allow_missing_columns: bool = False,
) -> pd.DataFrame | pl.DataFrame: ...


def get_df(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    nwb_path_to_row_indices: Mapping[str, Sequence[int]] | None = None,
    exclude_array_columns: bool = True,
    parallel: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    low_memory: bool = False,
    as_polars: bool | None = None,
    _catalog_snapshots: Mapping[
        str,
        catalog_models._TableSchemaSnapshot,
    ]
    | None = None,
    _allow_missing_columns: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    """ ""Get a DataFrame from one or more NWB files.

    Parameters
    ----------
    nwb_data_sources : str or PathLike, or iterable of these
        Paths to the NWB file(s) to read from. May be hdf5 or zarr.
    search_term : str
        An exact path to the table within each file, e.g. '/intervals/trials' or '/units', or a
        partial path, e.g. 'trials' or 'units'. If a partial path is provided, the function will
        scan the entire file for a match, which takes time - so be specific if you can.
        If the exact path is used, also set `exact_path=True`.
    exact_path : bool, default False
        Set to True if `search_term` is an exact path to the table within each file: this is
        important when a table is not present in all files, to ensure that the next closest match is
        not returned.
    include_column_names : str or iterable of str, default None
        Columns within the table to include in the DataFrame. If None, all columns are included.
    exclude_column_names : str or iterable of str, default None
        Columns within the table to exclude from the DataFrame. If None, no columns are excluded.
    exclude_array_columns : bool, default True
        If True, any column containing list- or array-like data (which can potentially be large)
        will not be returned. These can be merged after filtering the DataFrame, e.g.
        `get_df(nwb_paths, '/units').query('structure == MOs').pipe(merge_array_column, 'spike_times')`.
    use_process_pool : bool, default False
        If True, a process pool will be used to read the data from the files. This will not
        generally be faster than the default, which uses a thread pool.
    disable_progress : bool, default False
        If True, the progress bar will not be shown.
    raise_on_missing : bool, default False
        If True, a KeyError will be raised if the table is not found in any of the files.
    ignore_errors : bool, default False
        If True, any errors encountered while reading the files will be suppressed and a warning
        will be logged.
    low_memory : bool, default False
        If True, the data will be read in smaller chunks to reduce memory usage, at the cost of speed.
    as_polars : bool or None, default None
        If True, a Polars DataFrame will be returned. If False, a Pandas DataFrame will be
        returned. If None, ``lazynwb.config.use_polars`` decides the DataFrame backend.
    """
    t0 = time.time()
    as_polars = lazynwb_config._resolve_as_polars(as_polars)

    if nwb_path_to_row_indices is not None:
        paths = tuple(nwb_path_to_row_indices.keys())
    else:
        if isinstance(nwb_data_sources, (str, bytes)) or not isinstance(
            nwb_data_sources, Iterable
        ):
            paths = (nwb_data_sources,)  # type: ignore[assignment]
        else:
            paths = tuple(nwb_data_sources)  # type: ignore[arg-type]

    if exclude_column_names is not None:
        exclude_column_names = tuple(exclude_column_names)
        if len(paths) > 1 and (set(exclude_column_names) & set(INTERNAL_COLUMN_NAMES)):
            raise ValueError(
                "Cannot exclude internal column names when reading multiple files: they are required for identifying source of rows"
            )

    # speedup known table locations:
    if search_term in lazynwb.utils.TABLE_SHORTCUTS:
        search_term = lazynwb.utils.TABLE_SHORTCUTS[search_term]
        exact_path = True

    path_to_row_indices = _normalize_path_to_row_indices(nwb_path_to_row_indices)

    results: list[dict] = []
    if not parallel or len(paths) == 1:  # don't use a pool for a single file
        for path in paths:
            path_key = _catalog_snapshot_key(path)
            results.append(
                _get_table_data(
                    path=path,
                    search_term=search_term,
                    exact_path=exact_path,
                    exclude_column_names=exclude_column_names,
                    include_column_names=include_column_names,
                    exclude_array_columns=exclude_array_columns,
                    table_row_indices=path_to_row_indices.get(path_key),
                    catalog_snapshot=(
                        _catalog_snapshots.get(path_key)
                        if _catalog_snapshots is not None
                        else None
                    ),
                    low_memory=low_memory,
                    as_polars=as_polars,
                    allow_missing_columns=_allow_missing_columns,
                )
            )
    else:
        if exclude_array_columns and use_process_pool:
            logger.warning(
                "exclude_array_columns is True: setting use_process_pool=False for speed"
            )
            use_process_pool = False

        executor = (
            lazynwb.utils.get_processpool_executor()
            if use_process_pool
            else lazynwb.utils.get_threadpool_executor()
        )
        future_to_path = {}
        for path in paths:
            path_key = _catalog_snapshot_key(path)
            future = executor.submit(
                _get_table_data,
                path=path,
                search_term=search_term,
                exact_path=exact_path,
                exclude_column_names=exclude_column_names,
                include_column_names=include_column_names,
                exclude_array_columns=exclude_array_columns,
                table_row_indices=path_to_row_indices.get(path_key),
                catalog_snapshot=(
                    _catalog_snapshots.get(path_key)
                    if _catalog_snapshots is not None
                    else None
                ),
                low_memory=low_memory,
                as_polars=as_polars,
                allow_missing_columns=_allow_missing_columns,
            )
            future_to_path[future] = path
        futures = concurrent.futures.as_completed(future_to_path)
        if not disable_progress:
            futures = tqdm.tqdm(
                futures,
                total=len(future_to_path),
                desc=f"Getting multi-NWB {search_term} table",
                unit="NWB",
                ncols=120,
            )
        for future in futures:
            try:
                results.append(future.result())
            except KeyError:
                if raise_on_missing:
                    raise
                else:
                    logger.warning(
                        f"Table {search_term!r} not found in {lazynwb.file_io.from_pathlike(future_to_path[future]).as_posix()}"
                    )
                    continue
            except Exception:
                if not ignore_errors:
                    raise
                else:
                    logger.exception(
                        f"Error getting DataFrame for {lazynwb.file_io.from_pathlike(future_to_path[future]).as_posix()}:"
                    )
                    continue
    if not as_polars:
        df = pd.concat((pd.DataFrame(r) for r in results), ignore_index=True)
    else:
        df = pl.concat(
            (pl.DataFrame(r) for r in results), how="diagonal_relaxed", rechunk=False
        )
    logger.debug(
        f"Created {search_term!r} DataFrame ({len(df)} rows) from {len(paths)} NWB files in {time.time() - t0:.2f} s"
    )
    return df


def _get_timeseries_length_from_metadata(
    columns: Iterable[ColumnMetadataType],
) -> int | None:
    for column in columns:
        if column.name == "data" and column.shape:
            return column.shape[0]
    return None


def _filter_table_metadata_for_materialization(
    columns: Iterable[ColumnMetadataType],
    exclude_array_columns: bool,
    exclude_column_names: Iterable[str] | None,
    include_column_names: Iterable[str] | None,
) -> tuple[ColumnMetadataType, ...]:
    columns = tuple(columns)
    remaining_column_names = {column.name for column in columns}
    has_include_filter = include_column_names is not None
    exclude_column_names = set(exclude_column_names or ())
    include_column_names = set(include_column_names or ())

    for column in columns:
        if column.is_index_column:
            # users include/exclude ragged columns by the data column name, not the raw *_index
            continue
        is_synthetic_timestamps = (
            column.name == "starting_time" and column.is_timeseries_with_rate
        )
        is_excluded = column.name in exclude_column_names or (
            is_synthetic_timestamps and "timestamps" in exclude_column_names
        )
        is_included = column.name in include_column_names or (
            is_synthetic_timestamps and "timestamps" in include_column_names
        )
        is_not_included = has_include_filter and not is_included
        is_unrequested_indexed_array = (
            exclude_array_columns and column.is_nominally_indexed and not is_included
        )
        if is_not_included or is_excluded or is_unrequested_indexed_array:
            remaining_column_names.discard(column.name)
            remaining_column_names.discard(f"{column.name}_index")
            remaining_column_names.discard(column.name.removesuffix("_index"))

    filtered_columns = tuple(
        column for column in columns if column.name in remaining_column_names
    )
    logger.debug(
        "planned %d/%d raw columns for materialization: %s",
        len(filtered_columns),
        len(columns),
        [column.name for column in filtered_columns],
    )
    return filtered_columns


def _is_string_or_object_dtype(dtype: object | None) -> bool:
    return getattr(dtype, "kind", None) in ("S", "O", "U") or dtype in ("S", "O", "U")


def _normalize_column_name_filter(
    column_names: str | Iterable[str] | None,
) -> tuple[str, ...] | None:
    if isinstance(column_names, str):
        return (column_names,)
    if column_names is None:
        return None
    return tuple(column_names)


def _validate_column_name_filters(
    include_column_names: Iterable[str] | None,
    exclude_column_names: Iterable[str] | None,
) -> None:
    if include_column_names and exclude_column_names:
        ambiguous_column_names = set(include_column_names).intersection(
            exclude_column_names
        )
        if ambiguous_column_names:
            raise ValueError(
                f"Column names {ambiguous_column_names} are both included "
                "and excluded: unclear how to proceed"
            )


def _only_internal_columns_requested(
    include_column_names: Iterable[str] | None,
) -> bool:
    return bool(
        include_column_names
        and set(include_column_names).issubset(INTERNAL_COLUMN_NAMES)
    )


def _run_async_value(
    coroutine: typing.Coroutine[object, object, AsyncValueType],
) -> AsyncValueType:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    future = lazynwb.utils.get_threadpool_executor().submit(asyncio.run, coroutine)
    return future.result()


async def _read_table_schema_snapshot_and_close(
    reader: catalog_backend._BackendReader,
    exact_table_path: str,
) -> catalog_models._TableSchemaSnapshot:
    try:
        return await reader.read_table_schema_snapshot(exact_table_path)
    finally:
        await reader.close()


def _fast_catalog_backend_order(
    file_path: lazynwb.types_.PathLike,
) -> tuple[_FastCatalogBackendName, _FastCatalogBackendName]:
    if zarr_reader._source_name_has_zarr_suffix(file_path):
        return ("zarr", "hdf5")
    return ("hdf5", "zarr")


def _get_fast_hdf5_catalog_snapshot_if_available(
    file_path: lazynwb.types_.PathLike,
    exact_table_path: str,
) -> catalog_models._TableSchemaSnapshot | None:
    if not hdf5_reader._is_fast_hdf5_candidate(file_path):
        return None
    reader = hdf5_reader._default_hdf5_backend_reader(file_path)
    try:
        logger.debug(
            "using fast HDF5 catalog snapshot for materialization planning: %r/%s",
            file_path,
            exact_table_path,
        )
        return _run_async_value(
            _read_table_schema_snapshot_and_close(reader, exact_table_path)
        )
    except hdf5_reader._NotHDF5Error:
        logger.debug("fast HDF5 backend rejected non-HDF5 source %r", file_path)
        return None


def _get_fast_zarr_catalog_snapshot_if_available(
    file_path: lazynwb.types_.PathLike,
    exact_table_path: str,
) -> catalog_models._TableSchemaSnapshot | None:
    if not zarr_reader._is_fast_zarr_candidate(file_path):
        return None
    reader: catalog_backend._BackendReader = zarr_reader._default_zarr_backend_reader(
        file_path
    )
    logger.debug(
        "using fast Zarr catalog snapshot for materialization planning: %r/%s",
        file_path,
        exact_table_path,
    )
    return _run_async_value(
        _read_table_schema_snapshot_and_close(reader, exact_table_path)
    )


def _get_fast_catalog_snapshot_if_available(
    file_path: lazynwb.types_.PathLike,
    exact_table_path: str,
) -> catalog_models._TableSchemaSnapshot | None:
    backend_order = _fast_catalog_backend_order(file_path)
    logger.debug(
        "using fast catalog backend order for %r: %s",
        file_path,
        " -> ".join(backend_order),
    )
    for backend_name in backend_order:
        if backend_name == "hdf5":
            snapshot = _get_fast_hdf5_catalog_snapshot_if_available(
                file_path,
                exact_table_path,
            )
        else:
            snapshot = _get_fast_zarr_catalog_snapshot_if_available(
                file_path,
                exact_table_path,
            )
        if snapshot is not None:
            return snapshot
    return None


def _catalog_snapshot_key(file_path: lazynwb.types_.PathLike) -> str:
    return lazynwb.file_io.from_pathlike(file_path).as_posix()


def _normalize_path_to_row_indices(
    path_to_row_indices: Mapping[str, Sequence[int]] | None,
) -> dict[str, Sequence[int]]:
    if path_to_row_indices is None:
        return {}
    return {
        _catalog_snapshot_key(path): row_indices
        for path, row_indices in path_to_row_indices.items()
    }


def _raw_metadata_from_catalog_column(
    column: catalog_models._TableColumnSchema,
    accessor: lazynwb.table_metadata.TableColumnAccessor,
) -> lazynwb.table_metadata.RawTableColumnMetadata:
    return lazynwb.table_metadata.RawTableColumnMetadata(
        name=column.name,
        table_path=column.table_path,
        source_path=column.source_path,
        backend=column.backend,
        dtype=getattr(accessor, "dtype", None),
        shape=column.shape,
        ndim=column.ndim,
        attrs=dict(column.attrs),
        maxshape=column.dataset.maxshape,
        chunks=column.dataset.chunks,
        storage_layout=column.dataset.storage_layout,
        compression=column.dataset.compression,
        compression_opts=column.dataset.compression_opts,
        filters=column.dataset.filters,
        fill_value=column.dataset.fill_value,
        read_capabilities=column.dataset.read_capabilities,
        is_group=column.is_group,
        is_dataset=column.is_dataset,
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
        _accessor=accessor,
    )


def _is_direct_hdf5_scalar_column(
    column: catalog_models._TableColumnSchema,
) -> bool:
    if not _has_direct_hdf5_layout(column):
        return False
    if column.is_nominally_indexed or column.is_index_column:
        return False
    if column.is_timeseries and not column.is_timeseries_length_aligned:
        return False
    if not _has_direct_hdf5_dtype(column):
        return False
    if column.ndim is None:
        return False
    return True


def _has_direct_hdf5_contiguous_layout(
    column: catalog_models._TableColumnSchema,
) -> bool:
    return (
        "direct_contiguous" in column.dataset.read_capabilities
        and column.dataset.hdf5_data_offset is not None
        and column.dataset.hdf5_storage_size is not None
    )


def _has_direct_hdf5_chunked_layout(
    column: catalog_models._TableColumnSchema,
) -> bool:
    dataset = column.dataset
    if not (
        dataset.storage_layout == "chunked"
        and dataset.chunks
        and dataset.hdf5_chunk_index_offset is not None
        and dataset.hdf5_chunk_rank is not None
        and dataset.hdf5_offset_size is not None
    ):
        return False
    filter_ids = {
        int(item["id"])
        for item in dataset.filters
        if isinstance(item, Mapping) and isinstance(item.get("id"), int)
    }
    return filter_ids.issubset(_SUPPORTED_DIRECT_HDF5_FILTER_IDS)


def _has_direct_hdf5_layout(column: catalog_models._TableColumnSchema) -> bool:
    if (
        column.shape
        and math.prod(column.shape) == 0
        and column.dataset.storage_layout in {"contiguous", "chunked"}
    ):
        return True
    return _has_direct_hdf5_contiguous_layout(column) or _has_direct_hdf5_chunked_layout(
        column
    )


def _has_direct_hdf5_dtype(
    column: catalog_models._TableColumnSchema,
) -> bool:
    dtype = column.dtype
    if dtype.kind in {"numeric", "bool", "string"}:
        return dtype.numpy_dtype is not None
    if dtype.kind == "vlen_string":
        return (
            dtype.itemsize is not None
            and column.dataset.hdf5_base_address is not None
            and column.dataset.hdf5_offset_size is not None
            and column.dataset.hdf5_length_size is not None
        )
    if dtype.kind == "array":
        return dtype.element_numpy_dtype is not None and dtype.itemsize is not None
    return False


def _is_direct_hdf5_indexed_data_column(
    column: catalog_models._TableColumnSchema,
    columns_by_name: Mapping[str, catalog_models._TableColumnSchema],
) -> bool:
    if column.is_index_column or not column.is_nominally_indexed:
        return False
    if column.ndim is None or column.ndim < 1:
        return False
    if not _has_direct_hdf5_dtype(column):
        return False
    if not _has_direct_hdf5_layout(column):
        return False
    index_column_name = column.index_column_name or f"{column.name}_index"
    index_column = columns_by_name.get(index_column_name)
    if index_column is None:
        return False
    return (
        index_column.is_index_column
        and _has_direct_hdf5_dtype(index_column)
        and _has_direct_hdf5_layout(index_column)
    )


def _plan_direct_hdf5_table_reads(
    columns: Iterable[catalog_models._TableColumnSchema],
) -> _DirectHDF5TableReadPlan:
    columns = tuple(columns)
    columns_by_name = {column.name: column for column in columns}
    direct_columns: list[catalog_models._TableColumnSchema] = []
    direct_indexed_columns: list[_DirectHDF5IndexedColumnReadPlan] = []
    fallback_columns: list[catalog_models._TableColumnSchema] = []
    direct_indexed_raw_names: set[str] = set()
    for column in columns:
        if _is_direct_hdf5_indexed_data_column(column, columns_by_name):
            index_column_name = column.index_column_name or f"{column.name}_index"
            direct_indexed_columns.append(
                _DirectHDF5IndexedColumnReadPlan(
                    data_column=column,
                    index_column=columns_by_name[index_column_name],
                )
            )
            direct_indexed_raw_names.add(column.name)
            direct_indexed_raw_names.add(index_column_name)
    for column in columns:
        if column.name in direct_indexed_raw_names:
            continue
        if _is_direct_hdf5_scalar_column(column):
            direct_columns.append(column)
        else:
            fallback_columns.append(column)
    read_plan = _DirectHDF5TableReadPlan(
        scalar_columns=tuple(direct_columns),
        indexed_columns=tuple(direct_indexed_columns),
        fallback_columns=tuple(fallback_columns),
    )
    logger.debug(
        "planned direct HDF5 table reads: scalar_columns=%s indexed_columns=%s "
        "fallback_columns=%s",
        [column.name for column in read_plan.scalar_columns],
        [plan.data_column.name for plan in read_plan.indexed_columns],
        [column.name for column in read_plan.fallback_columns],
    )
    return read_plan


def _direct_hdf5_unsupported_reason(
    column: catalog_models._TableColumnSchema,
) -> str:
    dataset = column.dataset
    if column.is_timeseries and not column.is_timeseries_length_aligned:
        return "TimeSeries sidecar length is not aligned with the data column"
    if column.dtype.kind not in {"numeric", "bool", "string", "vlen_string", "array"}:
        return f"datatype {column.dtype.kind!r} is not supported"
    if dataset.storage_layout == "chunked":
        filter_ids = [
            item.get("id") for item in dataset.filters if isinstance(item, Mapping)
        ]
        unsupported_filters = [
            filter_id
            for filter_id in filter_ids
            if not isinstance(filter_id, int)
            or filter_id not in _SUPPORTED_DIRECT_HDF5_FILTER_IDS
        ]
        if unsupported_filters:
            return f"chunk filters {unsupported_filters!r} are not supported"
        return "chunk index metadata is missing or unsupported"
    return f"storage layout {dataset.storage_layout!r} is not directly readable"


def _is_remote_source(path: lazynwb.types_.PathLike) -> bool:
    return urllib.parse.urlsplit(str(path)).scheme.casefold() not in {"", "file"}


def _unsupported_hdf5_layout_error(
    *,
    path: lazynwb.types_.PathLike,
    table_path: str,
    columns_and_reasons: Sequence[tuple[str, str]],
) -> lazynwb.exceptions.UnsupportedHDF5LayoutError:
    return lazynwb.exceptions.UnsupportedHDF5LayoutError(
        source_url=str(path),
        table_path=table_path,
        columns=tuple(column for column, _ in columns_and_reasons),
        reasons=tuple(reason for _, reason in columns_and_reasons),
    )


def _source_path_strings(
    path: lazynwb.types_.PathLike,
) -> tuple[str, str]:
    raw_path = str(path)
    parsed_path = urllib.parse.urlsplit(raw_path)
    if parsed_path.scheme.casefold() == "file":
        # Preserve the caller's canonical file URI. Windows UPath stringification
        # otherwise collapses file:///C:/... to file://C:/....
        if parsed_path.netloc.endswith(":"):
            raw_path = f"file:///{parsed_path.netloc}{parsed_path.path}"
        return raw_path, raw_path
    u_path = lazynwb.file_io.from_pathlike(path)
    log_source_path = u_path.as_posix()
    if getattr(u_path, "protocol", None) not in (None, "", "file"):
        return log_source_path, log_source_path
    try:
        source_path = u_path.resolve().as_posix()
    except Exception:
        source_path = log_source_path
    return source_path, log_source_path


def _materialize_direct_hdf5_read_plan(
    path: lazynwb.types_.PathLike,
    read_plan: _DirectHDF5TableReadPlan,
    table_row_indices: Sequence[int] | None,
) -> dict[str, Any]:
    if not read_plan.has_direct_columns:
        return {}
    reader = hdf5_reader._default_hdf5_backend_reader(path)
    request_count_before = int(getattr(reader._range_reader, "request_count", 0))
    fetched_bytes_before = int(getattr(reader._range_reader, "bytes_fetched", 0))
    try:
        column_data = _run_async_value(
            _materialize_direct_hdf5_read_plan_async(
                reader._range_reader,
                read_plan,
                table_row_indices,
            )
        )
    finally:
        _run_async_value(reader.close())
    logger.debug(
        "direct HDF5 materialization for %r: scalar_columns=%s indexed_columns=%s "
        "requests=%d bytes=%d",
        path,
        [column.name for column in read_plan.scalar_columns],
        [plan.data_column.name for plan in read_plan.indexed_columns],
        int(getattr(reader._range_reader, "request_count", 0)) - request_count_before,
        int(getattr(reader._range_reader, "bytes_fetched", 0)) - fetched_bytes_before,
    )
    return column_data


async def _materialize_direct_hdf5_read_plan_async(
    range_reader: hdf5_range_reader._RangeReader,
    read_plan: _DirectHDF5TableReadPlan,
    table_row_indices: Sequence[int] | None,
) -> dict[str, Any]:
    column_names = [column.name for column in read_plan.scalar_columns]
    reads: list[typing.Awaitable[Any]] = [
        _read_direct_hdf5_column_array(range_reader, column, table_row_indices)
        for column in read_plan.scalar_columns
    ]
    column_names.extend(plan.data_column.name for plan in read_plan.indexed_columns)
    reads.extend(
        _read_direct_hdf5_indexed_column(
            range_reader,
            indexed_plan,
            table_row_indices,
        )
        for indexed_plan in read_plan.indexed_columns
    )
    values = await asyncio.gather(*reads)
    column_data: dict[str, Any] = dict(zip(column_names, values, strict=True))
    logger.debug(
        "completed %d direct HDF5 column reads concurrently",
        len(reads),
    )
    return column_data


async def _read_direct_hdf5_column_array(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    table_row_indices: Sequence[int] | None,
) -> npt.NDArray[Any] | np.generic:
    if column.shape and math.prod(column.shape) == 0:
        return _empty_direct_hdf5_array(column, leading_size=column.shape[0])
    if column.dataset.storage_layout == "contiguous":
        return await _read_direct_hdf5_contiguous_array(
            range_reader,
            column,
            table_row_indices,
        )
    if column.dataset.storage_layout == "chunked":
        return await _read_direct_hdf5_chunked_array(
            range_reader,
            column,
            table_row_indices,
        )
    raise _DirectHDF5ReadError(
        column.name,
        f"unsupported storage layout {column.dataset.storage_layout!r}",
    )


async def _read_direct_hdf5_contiguous_array(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    table_row_indices: Sequence[int] | None,
) -> npt.NDArray[Any] | np.generic:
    dataset = column.dataset
    if dataset.hdf5_data_offset is None or dataset.hdf5_storage_size is None:
        raise _DirectHDF5ReadError(column.name, "missing contiguous byte layout facts")
    itemsize = _hdf5_storage_itemsize(column)
    if column.ndim == 0:
        payload = await range_reader.read_range(
            dataset.hdf5_data_offset,
            length=itemsize,
        )
        decoded = await _decode_direct_hdf5_payload(
            range_reader,
            column,
            payload,
            logical_shape=(),
        )
        return decoded.item() if isinstance(decoded, np.ndarray) else decoded

    row_count = int(column.shape[0]) if column.shape else 0
    trailing_count = math.prod(column.shape[1:]) if column.shape else 1
    row_byte_length = trailing_count * itemsize
    if table_row_indices is None:
        byte_length = min(dataset.hdf5_storage_size, row_count * row_byte_length)
        payload = await range_reader.read_range(
            dataset.hdf5_data_offset,
            length=byte_length,
        )
        return await _decode_direct_hdf5_payload(
            range_reader,
            column,
            payload,
            logical_shape=column.shape or (),
        )

    row_indices = _normalize_indexed_table_row_indices(
        table_row_indices,
        row_count=row_count,
    )
    if row_indices.size == 0:
        return _empty_direct_hdf5_array(column, leading_size=0)
    ranges = _row_indices_to_byte_ranges(
        row_indices,
        data_offset=dataset.hdf5_data_offset,
        itemsize=row_byte_length,
    )
    payloads = await range_reader.read_ranges(ranges)
    selected_shape = (len(row_indices), *(column.shape or ())[1:])
    selected_payload = b"".join(
        payloads[
            hdf5_range_reader._ByteRange(
                dataset.hdf5_data_offset + (int(row_index) * row_byte_length),
                dataset.hdf5_data_offset + ((int(row_index) + 1) * row_byte_length),
            )
        ]
        for row_index in row_indices
    )
    return await _decode_direct_hdf5_payload(
        range_reader,
        column,
        selected_payload,
        logical_shape=selected_shape,
    )


def _hdf5_storage_itemsize(column: catalog_models._TableColumnSchema) -> int:
    itemsize = column.dtype.itemsize
    if itemsize is None and column.dtype.numpy_dtype is not None:
        itemsize = np.dtype(column.dtype.numpy_dtype).itemsize
    if itemsize is None or itemsize <= 0:
        raise _DirectHDF5ReadError(column.name, "missing or invalid datatype size")
    return int(itemsize)


def _hdf5_element_numpy_dtype(
    column: catalog_models._TableColumnSchema,
) -> np.dtype[Any]:
    dtype_string = (
        column.dtype.element_numpy_dtype
        if column.dtype.kind == "array"
        else column.dtype.numpy_dtype
    )
    if dtype_string is None:
        raise _DirectHDF5ReadError(column.name, "missing NumPy datatype")
    return np.dtype(dtype_string)


def _hdf5_decoded_shape(
    column: catalog_models._TableColumnSchema,
    logical_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if column.dtype.kind == "array" and column.dtype.element_shape:
        return (*logical_shape, *column.dtype.element_shape)
    return logical_shape


def _empty_direct_hdf5_array(
    column: catalog_models._TableColumnSchema,
    *,
    leading_size: int,
) -> npt.NDArray[Any]:
    shape = (leading_size, *(column.shape or ())[1:])
    shape = _hdf5_decoded_shape(column, shape)
    if column.dtype.kind in {"string", "vlen_string"}:
        return np.empty(shape, dtype=object)
    return np.empty(shape, dtype=_hdf5_element_numpy_dtype(column))


async def _decode_direct_hdf5_payload(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    payload: bytes,
    *,
    logical_shape: tuple[int, ...],
) -> npt.NDArray[Any]:
    element_count = math.prod(logical_shape) if logical_shape else 1
    decoded_shape = _hdf5_decoded_shape(column, logical_shape)
    if column.dtype.kind == "vlen_string":
        references = _parse_direct_hdf5_vlen_references(column, payload, element_count)
        strings = await _resolve_direct_hdf5_vlen_strings(
            range_reader,
            column,
            references,
        )
        return np.asarray(strings, dtype=object).reshape(decoded_shape)
    if column.dtype.kind == "string":
        raw_dtype = np.dtype(column.dtype.numpy_dtype)
        raw_values = np.frombuffer(payload, dtype=raw_dtype, count=element_count)
        strings = [
            bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            for value in raw_values
        ]
        return np.asarray(strings, dtype=object).reshape(decoded_shape)
    np_dtype = _hdf5_element_numpy_dtype(column)
    decoded_count = math.prod(decoded_shape) if decoded_shape else 1
    return np.frombuffer(payload, dtype=np_dtype, count=decoded_count).copy().reshape(
        decoded_shape
    )


def _parse_direct_hdf5_vlen_references(
    column: catalog_models._TableColumnSchema,
    payload: bytes,
    count: int,
) -> tuple[tuple[int, int, int], ...]:
    offset_size = column.dataset.hdf5_offset_size
    if offset_size is None:
        raise _DirectHDF5ReadError(column.name, "missing HDF5 offset size")
    stride = 4 + offset_size + 4
    if len(payload) < count * stride:
        raise _DirectHDF5ReadError(
            column.name,
            f"short vlen reference payload: need {count * stride}, got {len(payload)}",
        )
    return tuple(
        (
            int.from_bytes(payload[index * stride : index * stride + 4], "little"),
            int.from_bytes(
                payload[index * stride + 4 : index * stride + 4 + offset_size],
                "little",
            ),
            int.from_bytes(
                payload[
                    index * stride
                    + 4
                    + offset_size : index * stride
                    + 8
                    + offset_size
                ],
                "little",
            ),
        )
        for index in range(count)
    )


async def _resolve_direct_hdf5_vlen_strings(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    references: Sequence[tuple[int, int, int]],
) -> list[str]:
    dataset = column.dataset
    base_address = dataset.hdf5_base_address
    length_size = dataset.hdf5_length_size
    if base_address is None or length_size is None:
        raise _DirectHDF5ReadError(column.name, "missing global-heap address facts")
    heap_addresses = tuple(
        dict.fromkeys(address for length, address, index in references if length and index)
    )
    if not heap_addresses:
        return ["" for _ in references]
    heap_probe_size = 4096
    source_identity = await range_reader.get_source_identity()
    content_length = source_identity.content_length
    header_ranges = tuple(
        hdf5_range_reader._ByteRange(
            base_address + address,
            min(base_address + address + heap_probe_size, content_length)
            if content_length is not None
            else base_address + address + heap_probe_size,
        )
        for address in heap_addresses
    )
    header_payloads = await range_reader.read_ranges(header_ranges)
    header_payload_by_address = {
        address: header_payloads[byte_range]
        for address, byte_range in zip(heap_addresses, header_ranges, strict=True)
    }
    collection_ranges: dict[int, hdf5_range_reader._ByteRange] = {}
    collection_payload_by_address: dict[int, bytes] = {}
    for address in heap_addresses:
        header = header_payload_by_address[address]
        if header[:4] != b"GCOL":
            raise _DirectHDF5ReadError(
                column.name,
                f"expected global heap collection at {address}, found {header[:4]!r}",
            )
        collection_size = int.from_bytes(header[8 : 8 + length_size], "little")
        if collection_size <= len(header):
            collection_payload_by_address[address] = header[:collection_size]
        else:
            collection_ranges[address] = hdf5_range_reader._ByteRange(
                base_address + address + len(header),
                base_address + address + collection_size,
            )
    collection_payloads = await range_reader.read_ranges(collection_ranges.values())
    collection_payload_by_address.update(
        {
            address: (
                header_payload_by_address[address] + collection_payloads[byte_range]
            )
            for address, byte_range in collection_ranges.items()
        }
    )
    heap_objects = {
        address: _parse_direct_hdf5_global_heap_collection(
            collection_payload_by_address[address],
            length_size=length_size,
        )
        for address in heap_addresses
    }
    strings: list[str] = []
    for length, address, index in references:
        if not length or not index:
            strings.append("")
            continue
        try:
            value = heap_objects[address][index][:length]
        except KeyError as exc:
            raise _DirectHDF5ReadError(
                column.name,
                f"global heap object {index} was not found at {address}",
            ) from exc
        strings.append(value.rstrip(b"\x00").decode("utf-8", errors="replace"))
    logger.debug(
        "resolved %d HDF5 vlen strings from %d coalesced global heap collections",
        len(references),
        len(heap_addresses),
    )
    return strings


def _parse_direct_hdf5_global_heap_collection(
    payload: bytes,
    *,
    length_size: int,
) -> dict[int, bytes]:
    collection_size = min(
        int.from_bytes(payload[8 : 8 + length_size], "little"),
        len(payload),
    )
    cursor = 8 + length_size
    objects: dict[int, bytes] = {}
    while cursor + 8 + length_size <= collection_size:
        object_index = int.from_bytes(payload[cursor : cursor + 2], "little")
        object_size = int.from_bytes(
            payload[cursor + 8 : cursor + 8 + length_size], "little"
        )
        cursor += 8 + length_size
        if object_index == 0 and object_size == 0:
            break
        if object_index:
            objects[object_index] = bytes(payload[cursor : cursor + object_size])
        cursor += (object_size + 7) & ~7
    return objects


async def _read_direct_hdf5_chunked_array(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    table_row_indices: Sequence[int] | None,
) -> npt.NDArray[Any] | np.generic:
    dataset = column.dataset
    shape = column.shape or ()
    chunks = dataset.chunks or ()
    if not shape or len(chunks) != len(shape):
        raise _DirectHDF5ReadError(
            column.name,
            f"invalid chunk shape {chunks!r} for dataset shape {shape!r}",
        )
    if table_row_indices is None:
        row_indices = np.arange(shape[0], dtype=np.intp)
    else:
        row_indices = _normalize_indexed_table_row_indices(
            table_row_indices,
            row_count=shape[0],
        )
    if row_indices.size == 0:
        return _empty_direct_hdf5_array(column, leading_size=0)

    records = await _read_direct_hdf5_v1_chunk_index(range_reader, column)
    needed_first_offsets = {int(row // chunks[0]) * chunks[0] for row in row_indices}
    needed_records = tuple(
        record
        for record in records
        if record.offsets
        and record.offsets[0] in needed_first_offsets
        and all(
            offset < dimension
            for offset, dimension in zip(record.offsets, shape, strict=True)
        )
    )
    chunk_ranges = tuple(
        hdf5_range_reader._ByteRange(
            record.address,
            record.address + record.stored_size,
        )
        for record in needed_records
    )
    stored_payloads = await range_reader.read_ranges(chunk_ranges)
    output_shape = _hdf5_decoded_shape(column, (len(row_indices), *shape[1:]))
    if column.dtype.kind in {"string", "vlen_string"}:
        output = np.full(
            output_shape,
            _direct_hdf5_fill_value(column, default=""),
            dtype=object,
        )
    else:
        output = np.full(
            output_shape,
            _direct_hdf5_fill_value(column, default=0),
            dtype=_hdf5_element_numpy_dtype(column),
        )
    output_rows_by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_row, selected_input_row in enumerate(row_indices):
        chunk_offset = int(selected_input_row // chunks[0]) * chunks[0]
        output_rows_by_chunk.setdefault(chunk_offset, []).append(
            (int(output_row), int(selected_input_row))
        )

    raw_payloads = tuple(
        _decode_direct_hdf5_chunk_filters(
            column,
            stored_payloads[byte_range],
            filter_mask=record.filter_mask,
        )
        for record, byte_range in zip(needed_records, chunk_ranges, strict=True)
    )
    if column.dtype.kind == "vlen_string":
        references_by_chunk = tuple(
            _parse_direct_hdf5_vlen_references(
                column,
                raw_payload,
                math.prod(chunks),
            )
            for raw_payload in raw_payloads
        )
        all_references = tuple(
            reference
            for chunk_references in references_by_chunk
            for reference in chunk_references
        )
        all_strings = await _resolve_direct_hdf5_vlen_strings(
            range_reader,
            column,
            all_references,
        )
        strings_per_chunk = math.prod(chunks)
        chunk_arrays = tuple(
            np.asarray(
                all_strings[
                    index * strings_per_chunk : (index + 1) * strings_per_chunk
                ],
                dtype=object,
            ).reshape(chunks)
            for index in range(len(raw_payloads))
        )
    else:
        chunk_arrays = tuple(
            await asyncio.gather(
                *(
                    _decode_direct_hdf5_payload(
                        range_reader,
                        column,
                        raw_payload,
                        logical_shape=chunks,
                    )
                    for raw_payload in raw_payloads
                )
            )
        )

    for record, chunk_array in zip(needed_records, chunk_arrays, strict=True):
        valid_lengths = tuple(
            min(chunk_size, dimension - offset)
            for chunk_size, dimension, offset in zip(
                chunks,
                shape,
                record.offsets,
                strict=True,
            )
        )
        trailing_target = tuple(
            slice(offset, offset + length)
            for offset, length in zip(record.offsets[1:], valid_lengths[1:])
        )
        trailing_source = tuple(slice(0, length) for length in valid_lengths[1:])
        element_rank = len(column.dtype.element_shape or ())
        element_slices = (slice(None),) * element_rank
        for output_row, input_row_value in output_rows_by_chunk.get(
            record.offsets[0], ()
        ):
            source_row = input_row_value - record.offsets[0]
            target_index: Any = (output_row, *trailing_target, *element_slices)
            source_index: Any = (source_row, *trailing_source, *element_slices)
            output[target_index] = chunk_array[source_index]
    logger.debug(
        "direct HDF5 chunked materialization for %r: selected_rows=%d "
        "index_records=%d selected_chunks=%d",
        column.name,
        len(row_indices),
        len(records),
        len(needed_records),
    )
    return output


def _direct_hdf5_fill_value(
    column: catalog_models._TableColumnSchema,
    *,
    default: object,
) -> object:
    fill_facts = column.dataset.fill_value
    if not isinstance(fill_facts, Mapping):
        return default
    value_hex = fill_facts.get("value_hex")
    if not isinstance(value_hex, str) or not value_hex:
        return default
    payload = bytes.fromhex(value_hex)
    if column.dtype.kind == "string":
        return payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if column.dtype.kind == "vlen_string":
        return default
    np_dtype = _hdf5_element_numpy_dtype(column)
    if len(payload) < np_dtype.itemsize:
        return default
    value = np.frombuffer(payload, dtype=np_dtype, count=1)[0]
    return value.item() if isinstance(value, np.generic) else value


async def _read_direct_hdf5_v1_chunk_index(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
) -> tuple[_HDF5ChunkRecord, ...]:
    dataset = column.dataset
    root_offset = dataset.hdf5_chunk_index_offset
    offset_size = dataset.hdf5_offset_size
    chunk_rank = dataset.hdf5_chunk_rank
    base_address = dataset.hdf5_base_address or 0
    if root_offset is None or offset_size is None or chunk_rank is None:
        raise _DirectHDF5ReadError(column.name, "missing v1 chunk-index facts")
    node_header_size = 8 + (2 * offset_size)
    key_size = 8 + (chunk_rank * 8)
    source_identity = await range_reader.get_source_identity()
    content_length = source_identity.content_length
    pending: tuple[int, ...] = (root_offset,)
    seen: set[int] = set()
    records: list[_HDF5ChunkRecord] = []
    while pending:
        level_addresses = tuple(address for address in pending if address not in seen)
        if not level_addresses:
            break
        seen.update(level_addresses)
        probe_ranges = tuple(
            hdf5_range_reader._ByteRange(
                address,
                min(address + 4096, content_length)
                if content_length is not None
                else address + 4096,
            )
            for address in level_addresses
        )
        probes = await range_reader.read_ranges(probe_ranges)
        node_facts: list[tuple[int, int, int]] = []
        oversized_node_ranges: list[hdf5_range_reader._ByteRange] = []
        node_payloads: dict[int, bytes] = {}
        for address, byte_range in zip(level_addresses, probe_ranges, strict=True):
            header = probes[byte_range]
            if header[:4] != b"TREE" or header[4] != 1:
                raise _DirectHDF5ReadError(
                    column.name,
                    f"unsupported chunk index at byte {address}: {header[:5]!r}",
                )
            level = header[5]
            entries_used = int.from_bytes(header[6:8], "little")
            node_size = node_header_size + (entries_used * (key_size + offset_size))
            node_size += key_size
            node_facts.append((address, level, entries_used))
            if node_size <= len(header):
                node_payloads[address] = header[:node_size]
            else:
                oversized_node_ranges.append(
                    hdf5_range_reader._ByteRange(address, address + node_size)
                )
        oversized_nodes = await range_reader.read_ranges(oversized_node_ranges)
        node_payloads.update(
            {byte_range.start: payload for byte_range, payload in oversized_nodes.items()}
        )
        next_pending: list[int] = []
        for address, level, entries_used in node_facts:
            node = node_payloads[address]
            cursor = node_header_size
            for _ in range(entries_used):
                key = node[cursor : cursor + key_size]
                cursor += key_size
                child_address = int.from_bytes(
                    node[cursor : cursor + offset_size], "little"
                )
                cursor += offset_size
                if level:
                    next_pending.append(base_address + child_address)
                    continue
                records.append(
                    _HDF5ChunkRecord(
                        address=base_address + child_address,
                        stored_size=int.from_bytes(key[:4], "little"),
                        filter_mask=int.from_bytes(key[4:8], "little"),
                        offsets=tuple(
                            int.from_bytes(
                                key[8 + (index * 8) : 16 + (index * 8)],
                                "little",
                            )
                            for index in range(chunk_rank - 1)
                        ),
                    )
                )
        pending = tuple(next_pending)
    logger.debug(
        "resolved HDF5 v1 chunk index for %r: nodes=%d records=%d",
        column.name,
        len(seen),
        len(records),
    )
    return tuple(records)


def _decode_direct_hdf5_chunk_filters(
    column: catalog_models._TableColumnSchema,
    payload: bytes,
    *,
    filter_mask: int,
) -> bytes:
    decoded = payload
    filters = tuple(
        item for item in column.dataset.filters if isinstance(item, Mapping)
    )
    for pipeline_index in range(len(filters) - 1, -1, -1):
        if filter_mask & (1 << pipeline_index):
            continue
        filter_info = filters[pipeline_index]
        filter_id = filter_info.get("id")
        options = tuple(int(value) for value in filter_info.get("options", ()))
        if filter_id == 1:
            decoded = zlib.decompress(decoded)
        elif filter_id == 2:
            element_size = options[0] if options else _hdf5_storage_itemsize(column)
            decoded = _unshuffle_hdf5_chunk(decoded, element_size=element_size)
        elif filter_id == 3:
            if len(decoded) < 4:
                raise _DirectHDF5ReadError(column.name, "short Fletcher32 payload")
            decoded = decoded[:-4]
        elif filter_id == 32001:
            decoded = bytes(numcodecs.Blosc().decode(decoded))
        elif filter_id == 32015:
            decoded = bytes(numcodecs.Zstd().decode(decoded))
        else:
            raise _DirectHDF5ReadError(
                column.name,
                f"unsupported HDF5 filter {filter_id!r}",
            )
    expected_size = math.prod(column.dataset.chunks or ()) * _hdf5_storage_itemsize(
        column
    )
    if len(decoded) < expected_size:
        raise _DirectHDF5ReadError(
            column.name,
            f"decoded chunk is short: need {expected_size} bytes, got {len(decoded)}",
        )
    return decoded[:expected_size]


def _unshuffle_hdf5_chunk(payload: bytes, *, element_size: int) -> bytes:
    if element_size <= 1 or not payload:
        return payload
    if len(payload) % element_size:
        raise ValueError(
            f"shuffled payload length {len(payload)} is not divisible by {element_size}"
        )
    element_count = len(payload) // element_size
    shuffled = np.frombuffer(payload, dtype=np.uint8).reshape(
        element_size, element_count
    )
    return shuffled.T.copy().tobytes()


async def _read_direct_hdf5_indexed_column(
    range_reader: hdf5_range_reader._RangeReader,
    indexed_plan: _DirectHDF5IndexedColumnReadPlan,
    table_row_indices: Sequence[int] | None,
) -> list[list[Any]]:
    data_column = indexed_plan.data_column
    index_column = indexed_plan.index_column
    request_count_before = int(getattr(range_reader, "request_count", 0))
    fetched_bytes_before = int(getattr(range_reader, "bytes_fetched", 0))
    index_values = np.asarray(
        await _read_direct_hdf5_column_array(range_reader, index_column, None),
        dtype=np.intp,
    )
    range_plan = _plan_direct_hdf5_indexed_column_ranges(
        index_values=index_values,
        table_row_indices=table_row_indices,
    )
    if range_plan.row_indices.size == 0:
        logger.debug(
            "direct HDF5 indexed materialization for %r selected zero rows",
            data_column.name,
        )
        return []

    if range_plan.requested_element_count == 0:
        logger.debug(
            "direct HDF5 indexed materialization for %r selected %d empty rows",
            data_column.name,
            len(range_plan.row_indices),
        )
        return [[] for _ in range_plan.row_indices]

    span_payloads = await _read_direct_hdf5_element_spans(
        range_reader,
        data_column,
        range_plan.spans,
    )
    column_data = _split_direct_indexed_span_payloads(
        row_starts=range_plan.row_starts,
        row_ends=range_plan.row_ends,
        spans=range_plan.spans,
        span_payloads=span_payloads,
    )
    logger.debug(
        "direct HDF5 indexed materialization for %r: rows=%d spans=%d "
        "requested_elements=%d spanned_elements=%d full_elements=%d "
        "range_requests=%d fetched_bytes=%d",
        data_column.name,
        len(range_plan.row_indices),
        len(range_plan.spans),
        range_plan.requested_element_count,
        range_plan.spanned_element_count,
        range_plan.full_element_count,
        int(getattr(range_reader, "request_count", 0)) - request_count_before,
        int(getattr(range_reader, "bytes_fetched", 0)) - fetched_bytes_before,
    )
    return column_data


def _plan_direct_hdf5_indexed_column_ranges(
    *,
    index_values: npt.NDArray[Any],
    table_row_indices: Sequence[int] | None,
) -> _DirectHDF5IndexedRangeReadPlan:
    index_values = np.asarray(index_values, dtype=np.intp)
    index_array = np.empty(index_values.size + 1, dtype=np.intp)
    index_array[0] = 0
    index_array[1:] = index_values
    if table_row_indices is None:
        row_indices = np.arange(index_values.size, dtype=np.intp)
    else:
        row_indices = _normalize_indexed_table_row_indices(
            table_row_indices,
            row_count=index_values.size,
        )
    if row_indices.size == 0:
        empty_bounds = np.asarray([], dtype=np.intp)
        return _DirectHDF5IndexedRangeReadPlan(
            row_indices=row_indices,
            row_starts=empty_bounds,
            row_ends=empty_bounds,
            spans=(),
            requested_element_count=0,
            spanned_element_count=0,
            full_element_count=int(index_array[-1]),
        )

    row_starts = index_array[row_indices]
    row_ends = index_array[row_indices + 1]
    row_lengths = row_ends - row_starts
    requested_element_count = int(row_lengths.sum())
    spans = tuple(
        _coalesce_indexed_data_spans(
            row_starts,
            row_ends,
            max_gap=0,
        )
    )
    full_element_count = int(index_array[-1])
    spanned_element_count = sum(end - start for start, end in spans)
    materialized_spans = spans
    if _should_read_full_indexed_data(
        full_element_count=full_element_count,
        spanned_element_count=spanned_element_count,
    ):
        materialized_spans = ((0, full_element_count),)
    return _DirectHDF5IndexedRangeReadPlan(
        row_indices=row_indices,
        row_starts=row_starts,
        row_ends=row_ends,
        spans=materialized_spans,
        requested_element_count=requested_element_count,
        spanned_element_count=spanned_element_count,
        full_element_count=full_element_count,
    )


async def _read_direct_hdf5_element_spans(
    range_reader: hdf5_range_reader._RangeReader,
    column: catalog_models._TableColumnSchema,
    spans: Sequence[tuple[int, int]],
) -> list[npt.NDArray[Any]]:
    if not spans:
        return []
    selected_indices = np.concatenate(
        [np.arange(start, end, dtype=np.intp) for start, end in spans]
    )
    selected = np.asarray(
        await _read_direct_hdf5_column_array(
            range_reader,
            column,
            selected_indices.tolist(),
        )
    )
    payloads: list[npt.NDArray[Any]] = []
    cursor = 0
    for start, end in spans:
        length = end - start
        payloads.append(selected[cursor : cursor + length])
        cursor += length
    return payloads


def _split_direct_indexed_span_payloads(
    *,
    row_starts: npt.NDArray[np.intp],
    row_ends: npt.NDArray[np.intp],
    spans: Sequence[tuple[int, int]],
    span_payloads: Sequence[npt.NDArray[Any]],
) -> list[list[Any]]:
    if not spans:
        return [[] for _ in row_starts]
    span_starts = np.asarray([start for start, _ in spans], dtype=np.intp)
    column_data: list[list[Any]] = []
    for start, end in zip(row_starts, row_ends):
        if end <= start:
            column_data.append([])
            continue
        span_index = int(np.searchsorted(span_starts, start, side="right") - 1)
        span_start, span_end = spans[span_index]
        if span_start > start or end > span_end:
            raise AssertionError(
                "direct indexed span planner produced a span that does not contain "
                "a requested row"
            )
        payload = span_payloads[span_index]
        column_data.append(
            payload[int(start) - span_start : int(end) - span_start].tolist()
        )
    return column_data


def _row_indices_to_byte_ranges(
    row_indices: npt.NDArray[np.intp],
    *,
    data_offset: int,
    itemsize: int,
) -> tuple[hdf5_range_reader._ByteRange, ...]:
    return tuple(
        hdf5_range_reader._ByteRange(
            data_offset + (int(row_index) * itemsize),
            data_offset + ((int(row_index) + 1) * itemsize),
        )
        for row_index in row_indices
    )


def _get_fast_table_data_if_available(
    path: lazynwb.types_.PathLike,
    exact_table_path: str,
    include_column_names: Iterable[str] | None,
    exclude_column_names: Iterable[str] | None,
    exclude_array_columns: bool,
    table_row_indices: Sequence[int] | None,
    catalog_snapshot: catalog_models._TableSchemaSnapshot | None,
    low_memory: bool,
    as_polars: bool,
    allow_missing_columns: bool,
) -> dict[str, Any] | None:
    if catalog_snapshot is not None and catalog_snapshot.table_path == exact_table_path:
        snapshot = catalog_snapshot
        logger.debug(
            "using scan-carried fast catalog snapshot for materialization: %r/%s",
            path,
            exact_table_path,
        )
    else:
        if catalog_snapshot is not None:
            logger.debug(
                "ignoring scan-carried catalog snapshot for %r/%s: snapshot table is %s",
                path,
                exact_table_path,
                catalog_snapshot.table_path,
            )
        snapshot = _get_fast_catalog_snapshot_if_available(path, exact_table_path)
    if snapshot is None:
        return None

    filtered_catalog_columns = _filter_table_metadata_for_materialization(
        columns=snapshot.columns,
        exclude_array_columns=exclude_array_columns,
        exclude_column_names=exclude_column_names,
        include_column_names=include_column_names,
    )
    if not filtered_catalog_columns and not _only_internal_columns_requested(
        include_column_names
    ):
        logger.debug(
            "fast catalog materialization for %r/%s selected no raw columns",
            path,
            exact_table_path,
        )

    read_plan = _plan_direct_hdf5_table_reads(filtered_catalog_columns)
    if snapshot.backend == "hdf5" and read_plan.fallback_columns and _is_remote_source(
        path
    ):
        unsupported = tuple(
            (column.name, _direct_hdf5_unsupported_reason(column))
            for column in read_plan.fallback_columns
        )
        logger.debug(
            "rejecting remote accessor fallback for %r/%s: %s",
            path,
            exact_table_path,
            unsupported,
        )
        raise _unsupported_hdf5_layout_error(
            path=path,
            table_path=exact_table_path,
            columns_and_reasons=unsupported,
        )
    try:
        direct_column_data = _materialize_direct_hdf5_read_plan(
            path,
            read_plan,
            table_row_indices,
        )
    except _DirectHDF5ReadError as exc:
        raise _unsupported_hdf5_layout_error(
            path=path,
            table_path=exact_table_path,
            columns_and_reasons=((exc.column_name, exc.reason),),
        ) from exc
    logger.debug(
        "fast HDF5 materialization selected direct scalar columns=%s "
        "direct indexed columns=%s fallback columns=%s for %r/%s",
        [column.name for column in read_plan.scalar_columns],
        [plan.data_column.name for plan in read_plan.indexed_columns],
        [column.name for column in read_plan.fallback_columns],
        path,
        exact_table_path,
    )

    source_path, log_source_path = _source_path_strings(path)
    materialization_columns = tuple(
        _raw_metadata_from_catalog_column(column, accessor[column.dataset.path])
        for accessor in (
            (lazynwb.file_io._get_accessor(path),) if read_plan.fallback_columns else ()
        )
        for column in read_plan.fallback_columns
        if column.dataset.path
    )
    logger.debug(
        "materializing %d/%d raw columns from fast catalog snapshot for %s/%s",
        len(materialization_columns),
        len(snapshot.columns),
        log_source_path,
        exact_table_path,
    )
    return _materialize_table_data_from_columns(
        source_path=source_path,
        log_source_path=log_source_path,
        normalized_table_path=exact_table_path,
        all_columns=snapshot.columns,
        selected_columns=materialization_columns,
        include_column_names=include_column_names,
        exclude_column_names=exclude_column_names,
        table_row_indices=table_row_indices,
        exclude_array_columns=exclude_array_columns,
        low_memory=low_memory,
        as_polars=as_polars,
        table_length_from_metadata=snapshot.table_length,
        prefetched_column_data=direct_column_data,
        allow_missing_columns=allow_missing_columns,
    )


def _materialize_table_data_from_columns(  # noqa: C901
    *,
    source_path: str,
    log_source_path: str,
    normalized_table_path: str,
    all_columns: Iterable[ColumnMetadataType],
    selected_columns: Iterable[lazynwb.table_metadata.RawTableColumnMetadata],
    include_column_names: Iterable[str] | None,
    exclude_column_names: Iterable[str] | None,
    table_row_indices: Sequence[int] | None,
    exclude_array_columns: bool,
    low_memory: bool,
    as_polars: bool,
    table_length_from_metadata: int | None = None,
    prefetched_column_data: Mapping[str, Any] | None = None,
    allow_missing_columns: bool = False,
) -> dict[str, Any]:
    all_columns = tuple(all_columns)
    selected_columns = tuple(selected_columns)
    column_by_name = {column.name: column for column in selected_columns}
    is_metadata_table = any(column.is_metadata_table for column in all_columns)
    timeseries_len = _get_timeseries_length_from_metadata(all_columns)
    only_internal_columns_requested = _only_internal_columns_requested(
        include_column_names
    )

    selected_column_names = {column.name for column in selected_columns}
    indexed_column_names = catalog_schema._get_indexed_column_names(
        selected_column_names
    )
    non_indexed_columns = tuple(
        column for column in selected_columns if column.name not in indexed_column_names
    )
    multi_dim_columns: list[lazynwb.table_metadata.RawTableColumnMetadata] = []

    column_data: dict[str, Any] = dict(prefetched_column_data or {})
    logger.debug(
        "materializing non-indexed columns for %s/%s: %s",
        log_source_path,
        normalized_table_path,
        [column.name for column in non_indexed_columns],
    )
    if table_row_indices is not None:
        _idx: Sequence[int] | slice = table_row_indices
        table_length = len(table_row_indices)
    else:
        _idx = slice(None)
        table_length = table_length_from_metadata
    starting_time_column = next(
        (
            column
            for column in all_columns
            if column.name == "starting_time" and column.is_timeseries_with_rate
        ),
        None,
    )
    if starting_time_column is not None and "starting_time" in column_data:
        starting_time = np.asarray(column_data.pop("starting_time")).item()
        rate = starting_time_column.attrs["rate"]
        if timeseries_len is None:
            raise lazynwb.exceptions.InternalPathError(
                f"Could not determine TimeSeries length for {normalized_table_path!r}"
            )
        timestamps = np.linspace(
            starting_time,
            starting_time + timeseries_len / rate,
            num=timeseries_len,
        )
        column_data["timestamps"] = timestamps[_idx]
        logger.debug(
            "generated %d TimeSeries timestamps from range-read starting_time/rate",
            len(column_data["timestamps"]),
        )
    for column in non_indexed_columns:
        if column.ndim is None:
            continue
        if column.is_multidimensional:
            logger.debug(
                "non-indexed column %r has ndim=%d: will be treated as an array column",
                column.name,
                column.ndim,
            )
            multi_dim_columns.append(column)
            continue
        if column.is_timeseries and not column.is_timeseries_length_aligned:
            logger.debug(
                "skipping column %r with shape %s from TimeSeries table: "
                "length does not match data length %s",
                column.name,
                column.shape,
                timeseries_len,
            )
            continue
        if column.name == "starting_time" and column.is_timeseries_with_rate:
            starting_time = column.accessor[()]
            rate = column.attrs["rate"]
            if timeseries_len is None:
                raise lazynwb.exceptions.InternalPathError(
                    f"Could not determine TimeSeries length for {normalized_table_path!r}"
                )
            timestamps = np.linspace(
                starting_time, starting_time + timeseries_len / rate, num=timeseries_len
            )
            column_data["timestamps"] = timestamps[_idx]
            continue
        if _is_string_or_object_dtype(column.dtype):
            if not column.shape:
                column_data[column.name] = column.accessor.asstr()[()]
                continue
            try:
                column_data[column.name] = column.accessor.asstr()[_idx]
            except (AttributeError, TypeError):
                column_data[column.name] = column.accessor[_idx].astype(str)
        else:
            column_data[column.name] = column.accessor[_idx]

    if indexed_column_names and (
        include_column_names is not None or not exclude_array_columns
    ):
        data_columns = tuple(
            column_by_name[name]
            for name in indexed_column_names
            if not name.endswith("_index") and name in column_by_name
        )
        logger.debug(
            "materializing indexed columns for %s/%s: %s",
            log_source_path,
            normalized_table_path,
            [column.name for column in data_columns],
        )
        for column in data_columns:
            if column.is_timeseries and not column.is_timeseries_length_aligned:
                logger.debug(
                    "skipping column %r with shape %s from TimeSeries table: "
                    "length does not match data length %s",
                    column.name,
                    column.shape,
                    timeseries_len,
                )
                continue
            if _is_string_or_object_dtype(column.dtype):
                try:
                    data_column_accessor = column.accessor.asstr()
                except TypeError:
                    data_column_accessor = column.accessor.astype(str)
                except AttributeError:
                    data_column_accessor = column.accessor
            else:
                data_column_accessor = column.accessor
            index_column = column_by_name[column.index_column_name or ""]
            column_data[column.name] = _get_indexed_column_data(
                data_column_accessor=data_column_accessor,
                index_column_accessor=index_column.accessor,
                table_row_indices=table_row_indices,
                low_memory=low_memory,
            )
    if multi_dim_columns and (
        include_column_names is not None or not exclude_array_columns
    ):
        logger.debug(
            "materializing multi-dimensional array columns for %s/%s: %s",
            log_source_path,
            normalized_table_path,
            [column.name for column in multi_dim_columns],
        )
        for column in multi_dim_columns:
            multi_dim_column_data = column.accessor[_idx]
            if not as_polars:
                multi_dim_column_data = _format_multi_dim_column_pd(
                    multi_dim_column_data
                )
            column_data[column.name] = multi_dim_column_data

    if is_metadata_table:
        column_data = {k: [v] for k, v in column_data.items() if v is not None}

    if only_internal_columns_requested or (
        allow_missing_columns and include_column_names is not None and not column_data
    ):
        if allow_missing_columns and not only_internal_columns_requested:
            logger.debug(
                "No requested raw columns exist in %s/%s; emitting identifier-only "
                "rows for scan schema alignment",
                log_source_path,
                normalized_table_path,
            )
        if table_length is None:
            table_length = lazynwb.table_metadata.get_table_length_from_metadata(
                typing.cast(
                    Iterable[lazynwb.table_metadata.RawTableColumnMetadata], all_columns
                )
            )
    else:
        try:
            table_length = len(next(iter(column_data.values())))
        except StopIteration:
            raise lazynwb.exceptions.InternalPathError(
                f"Table matching {normalized_table_path!r} not found "
                f"in {log_source_path}"
            ) from None

    identifier_column_data = {
        NWB_PATH_COLUMN_NAME: [source_path] * table_length,
        TABLE_PATH_COLUMN_NAME: [normalized_table_path] * table_length,
        TABLE_INDEX_COLUMN_NAME: (
            table_row_indices
            if table_row_indices is not None
            else np.arange(table_length)
        ),
    }
    if exclude_column_names is not None:
        for column_name in set(exclude_column_names) & set(identifier_column_data):
            identifier_column_data.pop(column_name)

    return column_data | identifier_column_data


def _get_table_data(
    path: lazynwb.types_.PathLike,
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    exclude_array_columns: bool = True,
    table_row_indices: Sequence[int] | None = None,
    catalog_snapshot: catalog_models._TableSchemaSnapshot | None = None,
    low_memory: bool = False,
    as_polars: bool = False,
    allow_missing_columns: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    normalized_search_term = lazynwb.utils.normalize_internal_file_path(search_term)
    include_column_names = _normalize_column_name_filter(include_column_names)
    exclude_column_names = _normalize_column_name_filter(exclude_column_names)
    _validate_column_name_filters(include_column_names, exclude_column_names)

    if exact_path:
        fast_data = _get_fast_table_data_if_available(
            path=path,
            exact_table_path=normalized_search_term,
            include_column_names=include_column_names,
            exclude_column_names=exclude_column_names,
            exclude_array_columns=exclude_array_columns,
            table_row_indices=table_row_indices,
            catalog_snapshot=catalog_snapshot,
            low_memory=low_memory,
            as_polars=as_polars,
            allow_missing_columns=allow_missing_columns,
        )
        if fast_data is not None:
            logger.debug(
                "fetched data for %r/%s via fast catalog materialization in %.2f s",
                path,
                normalized_search_term,
                time.time() - t0,
            )
            return fast_data

    file = lazynwb.file_io._get_accessor(path)
    if not exact_path and normalized_search_term not in file:
        internal_paths = lazynwb.file_io.get_internal_paths(path)
        matches = difflib.get_close_matches(
            search_term, internal_paths, n=1, cutoff=0.3
        )
        if not matches:
            raise KeyError(f"Table {search_term!r} not found in {file._path}")
        match_ = matches[0]
        if (
            search_term not in match_
            or len(
                [
                    internal_path
                    for internal_path in internal_paths
                    if match_ in internal_path
                ]
            )
            > 1
        ):
            # only warn if there are multiple matches or if user-provided search term is not a
            # substring of the match
            logger.warning(f"Using {match_!r} instead of {search_term!r}")
        search_term = match_
        normalized_search_term = lazynwb.utils.normalize_internal_file_path(search_term)
    columns = lazynwb.table_metadata.get_table_column_metadata(
        file_path=file,
        table_path=normalized_search_term,
        use_thread_pool=False,
    )
    filtered_columns = _filter_table_metadata_for_materialization(
        columns=columns,
        exclude_array_columns=exclude_array_columns,
        exclude_column_names=exclude_column_names,
        include_column_names=include_column_names,
    )

    data = _materialize_table_data_from_columns(
        source_path=file._path.resolve().as_posix(),
        log_source_path=file._path.as_posix(),
        normalized_table_path=normalized_search_term,
        all_columns=columns,
        selected_columns=filtered_columns,
        include_column_names=include_column_names,
        exclude_column_names=exclude_column_names,
        table_row_indices=table_row_indices,
        exclude_array_columns=exclude_array_columns,
        low_memory=low_memory,
        as_polars=as_polars,
        allow_missing_columns=allow_missing_columns,
    )
    logger.debug(
        "fetched data for %s/%s via accessor materialization in %.2f s",
        file._path,
        normalized_search_term,
        time.time() - t0,
    )
    return data


def _get_indexed_column_data(
    data_column_accessor: zarr.Array | h5py.Dataset,
    index_column_accessor: zarr.Array | h5py.Dataset,
    table_row_indices: Sequence[int] | None = None,
    low_memory: bool = False,
) -> list[list[Any]]:
    """Get the data for an indexed column in a table, given the data and index array accessors.

    - default behavior is to return the data for all rows in the table
    - the data for a specified subset of rows can be returned by passing a sequence of row indices

    Notes:
    - the data array contains the actual values for all rows, concatenated:
        e.g. data_array = [0.1, 0.2, 0.3, 0.1, 0.4, 0.1, 0.2, ...]
    - the index array contains the indices at which each row's data ends and the next starts (with
      the first row implicitly starting at 0):
        e.g. index_array = [3, 5, 7, ...]
        data_for_each_row = {
            'row_0_data': data_array[0:3],
            'row_1_data': data_array[3:5],
            'row_2_data': data_array[5:7],
            ...
        }
    """
    # The index column is one value per table row, so it is small enough to read in one go.
    index_values = np.asarray(index_column_accessor[:], dtype=np.intp)
    index_array = np.empty(index_values.size + 1, dtype=np.intp)
    index_array[0] = 0
    index_array[1:] = index_values
    if table_row_indices is None:
        row_lengths = np.diff(index_array)
        logger.debug(
            "reading full indexed column data: rows=%d elements=%d",
            len(row_lengths),
            int(index_array[-1]) if len(index_array) else 0,
        )
        return _split_indexed_data_array(data_column_accessor[:], row_lengths)

    row_indices = _normalize_indexed_table_row_indices(
        table_row_indices,
        row_count=len(index_array) - 1,
    )
    if row_indices.size == 0:
        return []

    row_starts = index_array[row_indices]
    row_ends = index_array[row_indices + 1]
    row_lengths = row_ends - row_starts
    selected_element_count = int(row_lengths.sum())
    if selected_element_count == 0:
        return [[] for _ in row_indices]

    if low_memory:
        logger.debug(
            "reading indexed column subset row-by-row: rows=%d elements=%d",
            len(row_indices),
            selected_element_count,
        )
        return _read_indexed_rows_by_slice(data_column_accessor, row_starts, row_ends)

    spans = _coalesce_indexed_data_spans(
        row_starts,
        row_ends,
        max_gap=_get_indexed_data_coalesce_gap(data_column_accessor),
    )
    full_element_count = int(index_array[-1])
    spanned_element_count = sum(end - start for start, end in spans)

    if _should_read_full_indexed_data(
        full_element_count=full_element_count,
        spanned_element_count=spanned_element_count,
    ):
        logger.debug(
            "reading full indexed column data for subset because %d spans cover "
            "%d/%d elements",
            len(spans),
            spanned_element_count,
            full_element_count,
        )
        full_data = data_column_accessor[:]
        return [
            full_data[start:end].tolist()
            for start, end in zip(row_starts, row_ends)
        ]

    logger.debug(
        "reading indexed column subset with %d spans: rows=%d requested_elements=%d "
        "spanned_elements=%d full_elements=%d",
        len(spans),
        len(row_indices),
        selected_element_count,
        spanned_element_count,
        full_element_count,
    )
    return _read_indexed_rows_from_spans(
        data_column_accessor=data_column_accessor,
        row_starts=row_starts,
        row_ends=row_ends,
        spans=spans,
    )


def _normalize_indexed_table_row_indices(
    table_row_indices: Sequence[int],
    row_count: int,
) -> npt.NDArray[np.intp]:
    row_indices = np.asarray(table_row_indices, dtype=np.intp)
    if row_indices.ndim == 0:
        row_indices = row_indices.reshape(1)
    if row_indices.ndim != 1:
        raise ValueError("table_row_indices must be a one-dimensional sequence")
    if row_indices.size and (row_indices.min() < 0 or row_indices.max() >= row_count):
        raise IndexError(
            f"table_row_indices contains values outside table row range 0:{row_count}"
        )
    return row_indices


def _split_indexed_data_array(
    data_array: npt.NDArray[Any],
    row_lengths: npt.NDArray[np.intp],
) -> list[list[Any]]:
    column_data: list[list[Any]] = []
    start = 0
    for row_length in row_lengths:
        end = start + int(row_length)
        column_data.append(data_array[start:end].tolist())
        start = end
    return column_data


def _read_indexed_rows_by_slice(
    data_column_accessor: zarr.Array | h5py.Dataset,
    row_starts: npt.NDArray[np.intp],
    row_ends: npt.NDArray[np.intp],
) -> list[list[Any]]:
    return [
        data_column_accessor[int(start) : int(end)].tolist()
        if end > start
        else []
        for start, end in zip(row_starts, row_ends)
    ]


def _get_indexed_data_coalesce_gap(
    data_column_accessor: zarr.Array | h5py.Dataset,
) -> int:
    chunks = getattr(data_column_accessor, "chunks", None)
    if not chunks:
        return 0
    first_chunk = chunks[0]
    if first_chunk is None:
        return 0
    return min(int(first_chunk), _INDEXED_COLUMN_MAX_COALESCE_GAP_ELEMENTS)


def _coalesce_indexed_data_spans(
    row_starts: npt.NDArray[np.intp],
    row_ends: npt.NDArray[np.intp],
    max_gap: int,
) -> list[tuple[int, int]]:
    spans = sorted(
        (int(start), int(end))
        for start, end in zip(row_starts, row_ends)
        if end > start
    )
    if not spans:
        return []

    coalesced = [spans[0]]
    for start, end in spans[1:]:
        current_start, current_end = coalesced[-1]
        if start <= current_end + max_gap:
            coalesced[-1] = (current_start, max(current_end, end))
        else:
            coalesced.append((start, end))
    return coalesced


def _should_read_full_indexed_data(
    full_element_count: int,
    spanned_element_count: int,
) -> bool:
    if full_element_count <= 0:
        return False
    return (
        spanned_element_count / full_element_count
        >= _INDEXED_COLUMN_FULL_READ_MIN_COVERAGE
    )


def _read_indexed_rows_from_spans(
    data_column_accessor: zarr.Array | h5py.Dataset,
    row_starts: npt.NDArray[np.intp],
    row_ends: npt.NDArray[np.intp],
    spans: Sequence[tuple[int, int]],
) -> list[list[Any]]:
    if not spans:
        return [[] for _ in row_starts]

    span_starts = np.asarray([start for start, _ in spans], dtype=np.intp)
    span_payloads = [data_column_accessor[start:end] for start, end in spans]
    column_data: list[list[Any]] = []
    for start, end in zip(row_starts, row_ends):
        if end <= start:
            column_data.append([])
            continue
        span_index = int(np.searchsorted(span_starts, start, side="right") - 1)
        span_start, span_end = spans[span_index]
        if span_start > start or end > span_end:
            raise AssertionError(
                "indexed data span planner produced a span that does not contain "
                "a requested row"
            )
        payload = span_payloads[span_index]
        column_data.append(
            payload[int(start) - span_start : int(end) - span_start].tolist()
        )
    return column_data


def _array_column_helper(
    nwb_path: lazynwb.types_.PathLike,
    table_path: str,
    column_name: str,
    table_row_indices: Sequence[int],
    as_polars: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    fast_data = _get_fast_table_data_if_available(
        path=nwb_path,
        exact_table_path=normalized_table_path,
        include_column_names=(column_name,),
        exclude_column_names=None,
        exclude_array_columns=False,
        table_row_indices=table_row_indices,
        catalog_snapshot=None,
        low_memory=False,
        as_polars=as_polars,
        allow_missing_columns=False,
    )
    if fast_data is not None:
        if column_name not in fast_data:
            raise lazynwb.exceptions.ColumnError(column_name)
        logger.debug(
            "materialized large array column %r for %r/%s with range reads",
            column_name,
            nwb_path,
            normalized_table_path,
        )
        df_cls = pl.DataFrame if as_polars else pd.DataFrame
        column_data = fast_data[column_name]
        if not as_polars and isinstance(column_data, np.ndarray):
            column_data = _format_multi_dim_column_pd(column_data)
        return df_cls(
            {
                column_name: column_data,
                TABLE_INDEX_COLUMN_NAME: table_row_indices,
                NWB_PATH_COLUMN_NAME: [nwb_path] * len(table_row_indices),
            }
        )

    file = lazynwb.file_io._get_accessor(nwb_path)
    try:
        columns = lazynwb.table_metadata.get_table_column_metadata(file, table_path)
        column_by_name = {column.name: column for column in columns}
        column = column_by_name[column_name]
    except KeyError as exc:
        if exc.args[0] == column_name:
            raise lazynwb.exceptions.ColumnError(column_name) from None
        if exc.args[0] == table_path:
            raise lazynwb.exceptions.InternalPathError(table_path) from None
        raise
    if column.is_multidimensional:
        column_data = column.accessor[table_row_indices]
        if not as_polars:
            column_data = _format_multi_dim_column_pd(column_data)
    else:
        index_column_name = column.index_column_name or f"{column_name}_index"
        try:
            index_column = column_by_name[index_column_name]
        except KeyError:
            raise lazynwb.exceptions.ColumnError(column_name) from None
        column_data = _get_indexed_column_data(
            data_column_accessor=column.accessor,
            index_column_accessor=index_column.accessor,
            table_row_indices=table_row_indices,
        )
    df_cls = pl.DataFrame if as_polars else pd.DataFrame
    return df_cls(
        {
            column_name: column_data,
            TABLE_INDEX_COLUMN_NAME: table_row_indices,
            NWB_PATH_COLUMN_NAME: [nwb_path] * len(table_row_indices),
        },
    )


def _format_multi_dim_column_pd(
    column_data: npt.NDArray | list[npt.NDArray],
) -> list[list[Any]]:
    """Pandas inists 'Per-column arrays must each be 1-dimensional': this converts to a list of
    arrays, if not already"""
    if len(column_data) == 0:
        return []
    if isinstance(column_data[0], list):
        return list(column_data)  # type: ignore[arg-type]
    else:
        # np array-like
        return [x.tolist() for x in column_data]  # type: ignore[misc]


def _get_original_table_path(df: FrameType, assert_unique: bool = True) -> str:
    if isinstance(df, pl.LazyFrame):
        df = df.select(TABLE_PATH_COLUMN_NAME).collect()  # type: ignore[assignment]
    assert not isinstance(df, pl.LazyFrame)
    if len(df) == 0:
        raise ValueError("dataframe is empty: cannot determine original table path")
    try:
        series = df[TABLE_PATH_COLUMN_NAME]
    except KeyError:
        raise lazynwb.exceptions.ColumnError(
            f"Column {TABLE_PATH_COLUMN_NAME!r} not found in DataFrame"
        ) from None
    if assert_unique:
        assert len(set(series)) == 1, f"multiple table paths found: {set(series)}"
    return series[0]


def _get_table_column(df: FrameType, column_name: str) -> list[Any]:
    if isinstance(df, pl.LazyFrame):
        df = df.select(column_name).collect()  # type: ignore[assignment]
    assert not isinstance(df, pl.LazyFrame)
    if column_name not in df.columns:
        raise lazynwb.exceptions.ColumnError(
            f"Column {column_name!r} not found in DataFrame"
        )
    if isinstance(df, pd.DataFrame):
        return df[column_name].values.tolist()
    else:
        return df[column_name].to_list()


def merge_array_column(
    df: FrameType,
    column_name: str,
    missing_ok: bool = True,
) -> FrameType:
    column_data: list[pd.DataFrame] = []
    if isinstance(df, pl.LazyFrame):
        df = df.select(column_name).collect()  # type: ignore[assignment]
    assert not isinstance(df, pl.LazyFrame)
    if isinstance(df, pd.DataFrame):
        df = df.sort_values(by=[NWB_PATH_COLUMN_NAME, TABLE_INDEX_COLUMN_NAME])
    else:
        df = df.sort(NWB_PATH_COLUMN_NAME, TABLE_INDEX_COLUMN_NAME)
    future_to_path = {}
    for nwb_path, session_df in (
        df.groupby(NWB_PATH_COLUMN_NAME)
        if isinstance(df, pd.DataFrame)
        else df.group_by(NWB_PATH_COLUMN_NAME)
    ):
        if isinstance(nwb_path, tuple):
            nwb_path = nwb_path[0]
        assert isinstance(nwb_path, str)
        future = lazynwb.utils.get_threadpool_executor().submit(
            _array_column_helper,
            nwb_path=nwb_path,
            table_path=_get_original_table_path(session_df, assert_unique=True),
            column_name=column_name,
            table_row_indices=_get_table_column(session_df, TABLE_INDEX_COLUMN_NAME),
            as_polars=not isinstance(df, pd.DataFrame),
        )
        future_to_path[future] = nwb_path
    missing_column_already_warned = False
    for future in concurrent.futures.as_completed(future_to_path):
        try:
            column_data.append(future.result())
        except lazynwb.exceptions.ColumnError as exc:
            if not missing_ok:
                logger.error(
                    f"error getting indexed column data for {lazynwb.file_io.from_pathlike(future_to_path[future])}:"
                )
                raise
            if not missing_column_already_warned:
                logger.warning(
                    f"Column {exc.args[0]!r} not found: data will be missing from DataFrame"
                )
                missing_column_already_warned = True
            continue
        except:
            logger.error(
                f"error getting indexed column data for {lazynwb.file_io.from_pathlike(future_to_path[future])}:"
            )
            raise
    if not column_data:
        logger.debug(f"no {column_name!r} data found in any file")
        return df
    if isinstance(df, pd.DataFrame):
        return df.merge(
            pd.concat(column_data),
            how="left",
            on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
        ).set_index(df[TABLE_INDEX_COLUMN_NAME].values)
    else:
        return df.join(
            pl.concat(column_data, how="diagonal_relaxed"),
            on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
            how="left",
        )


def _get_table_column_accessors(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    use_thread_pool: bool = False,
    skip_references: bool = True,
) -> dict[str, zarr.Array | h5py.Dataset]:
    """Get the accessor objects for each column of an NWB table, as a dict of zarr.Array or
    h5py.Dataset objects. Note that the data from each column is not read into memory.

    Optionally use a thread pool to speed up retrieval of the columns - faster for zarr files.

    Parameters
    ----------
    file_path : lazynwb.types_.PathLike
        Path to the NWB file to read from.
    table_path : str
        Path to the table within the NWB file, e.g. '/intervals/trials'
    use_thread_pool : bool, default False
        If True, a thread pool will be used to retrieve the columns for speed.
    skip_references : bool, default True
        If True, columns that include references to other objects within the NWB file (e.g. TimeSeriesReferenceVectorData) will be skipped.
        These columns are added for convenience but are convoluted to interpret and impact performance when reading data from the cloud.
    """
    return lazynwb.table_metadata._get_table_column_accessors(
        file_path=file_path,
        table_path=table_path,
        use_thread_pool=use_thread_pool,
        skip_references=skip_references,
    )


def _get_polars_dtype(
    column: lazynwb.table_metadata.RawTableColumnMetadata,
    all_columns: Iterable[lazynwb.table_metadata.RawTableColumnMetadata],
) -> polars._typing.PolarsDataType:
    dtype = column.dtype
    if _is_string_or_object_dtype(dtype):
        dtype = pl.String
    else:
        dtype = polars.datatypes.convert.numpy_char_code_to_dtype(dtype)
    if column.is_metadata_table and column.shape:
        # this is a regular-looking array among a bunch of single-value metadata: it should be a list
        return pl.List(dtype)
    elif column.ndim is not None and column.ndim > 1:
        dtype = pl.Array(
            dtype, shape=column.shape[1:]
        )  # shape reported is (Ncols, (*shape for each row)
    if column.is_nominally_indexed:
        # - indexed = variable length list-like (e.g. spike times)
        # - it's possible to have a list of fixed-length arrays (e.g. obs_intervals)
        all_column_names = [raw_column.name for raw_column in all_columns]
        index_cols = [
            c
            for c in catalog_schema._get_indexed_column_names(all_column_names)
            if c.startswith(column.name) and c.endswith("_index")
        ]
        for _ in index_cols:
            # add as many levels of nested list as there are _index columns for this column
            dtype = pl.List(dtype)
    return dtype


def get_table_schema_from_metadata(
    columns: Iterable[lazynwb.table_metadata.RawTableColumnMetadata],
) -> pl.Schema:
    """Derive a per-file Polars schema from raw NWB table column metadata."""
    columns = tuple(columns)
    file_schema = pl.Schema()
    for column in columns:
        if column.is_index_column:
            # skip raw index columns; user-facing ragged columns include list dtype instead
            continue
        if column.is_group:
            continue
        if column.name == "starting_time" and column.is_timeseries_with_rate:
            # this is a TimeSeries object with start/rate: we'll generate timestamps
            file_schema["timestamps"] = pl.Float64
            continue
        if column.is_timeseries and not column.is_timeseries_length_aligned:
            logger.debug(
                "skipping column %r with shape %s from TimeSeries table: length does not match data length",
                column.name,
                column.shape,
            )
            continue
        file_schema[column.name] = _get_polars_dtype(column, columns)
    logger.debug(
        "derived Polars schema from %d raw metadata columns: %s",
        len(columns),
        file_schema,
    )
    return file_schema


def _get_table_length(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    *,
    catalog_snapshot: catalog_models._TableSchemaSnapshot | None = None,
) -> int:
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    if (
        catalog_snapshot is not None
        and catalog_snapshot.table_path == normalized_table_path
    ):
        snapshot = catalog_snapshot
        logger.debug(
            "resolved table length for %r/%s from scan-carried catalog snapshot",
            file_path,
            normalized_table_path,
        )
    else:
        snapshot = _get_fast_catalog_snapshot_if_available(
            file_path,
            normalized_table_path,
        )
    if snapshot is not None and snapshot.table_length is not None:
        logger.debug(
            "resolved table length for %r/%s from fast catalog snapshot: %d",
            file_path,
            normalized_table_path,
            snapshot.table_length,
        )
        return snapshot.table_length
    columns = lazynwb.table_metadata.get_table_column_metadata(file_path, table_path)
    return lazynwb.table_metadata.get_table_length_from_metadata(columns)


def _get_path_to_row_indices(df: pl.DataFrame) -> dict[str, list[int]]:
    return {
        d[NWB_PATH_COLUMN_NAME]: d[TABLE_INDEX_COLUMN_NAME]
        for d in df.group_by(NWB_PATH_COLUMN_NAME)
        .agg(TABLE_INDEX_COLUMN_NAME)
        .to_dicts()
    }


def _get_table_schema_helper(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    raise_on_missing: bool,
    fast_backend_order: Sequence[_FastCatalogBackendName] | None = None,
) -> dict[str, Any] | None:
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    backend_order = tuple(fast_backend_order or _fast_catalog_backend_order(file_path))
    logger.debug(
        "using fast schema backend order for %r: %s",
        file_path,
        " -> ".join(backend_order),
    )
    for backend_name in backend_order:
        try:
            if backend_name == "hdf5":
                fast_schema = _get_fast_hdf5_table_schema_if_available(
                    file_path=file_path,
                    table_path=normalized_table_path,
                )
            else:
                fast_schema = _get_fast_zarr_table_schema_if_available(
                    file_path=file_path,
                    table_path=normalized_table_path,
                )
        except KeyError:
            return _handle_missing_schema_table(
                file_path=file_path,
                table_path=table_path,
                raise_on_missing=raise_on_missing,
            )
        else:
            if fast_schema is not None:
                return fast_schema
    try:
        columns = lazynwb.table_metadata.get_table_column_metadata(
            file_path, normalized_table_path
        )
    except KeyError:
        return _handle_missing_schema_table(
            file_path=file_path,
            table_path=table_path,
            raise_on_missing=raise_on_missing,
        )
    else:
        return get_table_schema_from_metadata(columns)


def _run_async_schema_snapshot_batch(
    coroutine: typing.Coroutine[
        Any,
        Any,
        list[
            tuple[
                lazynwb.types_.PathLike,
                catalog_models._TableSchemaSnapshot | None,
                bool,
            ]
        ],
    ],
) -> list[
    tuple[
        lazynwb.types_.PathLike,
        catalog_models._TableSchemaSnapshot | None,
        bool,
    ]
]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    future = lazynwb.utils.get_threadpool_executor().submit(asyncio.run, coroutine)
    return future.result()


async def _get_fast_hdf5_table_schema_snapshots(
    file_paths: Sequence[lazynwb.types_.PathLike],
    table_path: str,
    raise_on_missing: bool,
) -> list[
    tuple[
        lazynwb.types_.PathLike,
        catalog_models._TableSchemaSnapshot | None,
        bool,
    ]
]:
    readers = [
        hdf5_reader._default_hdf5_backend_reader(file_path) for file_path in file_paths
    ]
    logger.debug(
        "getting fast HDF5 table schema for %r from %d files in one async batch",
        table_path,
        len(file_paths),
    )

    async def _read_schema(
        reader: hdf5_reader._HDF5BackendReader,
        file_path: lazynwb.types_.PathLike,
    ) -> tuple[
        lazynwb.types_.PathLike,
        catalog_models._TableSchemaSnapshot | None,
        bool,
    ]:
        try:
            snapshot = await reader.read_table_schema_snapshot(table_path)
        except KeyError:
            _handle_missing_schema_table(
                file_path=file_path,
                table_path=table_path,
                raise_on_missing=raise_on_missing,
            )
            return file_path, None, True
        except hdf5_reader._NotHDF5Error:
            logger.debug("fast HDF5 backend rejected non-HDF5 source %r", file_path)
            return file_path, None, False
        except FileNotFoundError as exc:
            logger.debug(
                "fast HDF5 backend could not find single-object source %r: %r",
                file_path,
                exc,
            )
            return file_path, None, False
        return file_path, snapshot, True

    try:
        return list(
            await asyncio.gather(
                *(
                    _read_schema(reader, file_path)
                    for reader, file_path in zip(readers, file_paths, strict=True)
                )
            )
        )
    finally:
        await asyncio.gather(
            *(reader.close() for reader in readers),
            return_exceptions=True,
        )


def _handle_missing_schema_table(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
    raise_on_missing: bool,
) -> None:
    if raise_on_missing:
        raise lazynwb.exceptions.InternalPathError(
            f"Table {table_path!r} not found in {file_path!r}"
        ) from None
    logger.info("Table %r not found in %r: skipping", table_path, file_path)
    return None


def _get_fast_hdf5_table_schema_if_available(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
) -> pl.Schema | None:
    if not hdf5_reader._is_fast_hdf5_candidate(file_path):
        return None
    reader = hdf5_reader._default_hdf5_backend_reader(file_path)
    try:
        snapshot = _run_async_value(
            _read_table_schema_snapshot_and_close(reader, table_path)
        )
    except hdf5_reader._NotHDF5Error:
        logger.debug("fast HDF5 backend rejected non-HDF5 source %r", file_path)
        return None
    except FileNotFoundError as exc:
        logger.debug(
            "fast HDF5 backend could not find single-object source %r: %r",
            file_path,
            exc,
        )
        return None
    return catalog_polars._snapshot_to_polars_schema(snapshot)


def _get_fast_zarr_table_schema_if_available(
    file_path: lazynwb.types_.PathLike,
    table_path: str,
) -> pl.Schema | None:
    if not zarr_reader._is_fast_zarr_candidate(file_path):
        return None
    reader = zarr_reader._default_zarr_backend_reader(file_path)
    snapshot = _run_async_value(
        _read_table_schema_snapshot_and_close(reader, table_path)
    )
    return catalog_polars._snapshot_to_polars_schema(snapshot)


def _get_table_schema_with_catalog_snapshots(
    file_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    table_path: str,
    first_n_files_to_infer_schema: int | None = None,
    exclude_array_columns: bool = False,
    exclude_internal_columns: bool = False,
    raise_on_missing: bool = False,
) -> _TableSchemaInferenceResult:
    if not isinstance(file_paths, Iterable) or isinstance(file_paths, (str, bytes)):
        file_paths = (file_paths,)
    file_paths = tuple(file_paths)
    if first_n_files_to_infer_schema is not None:
        file_paths = file_paths[: min(first_n_files_to_infer_schema, len(file_paths))]
    per_file_schemas: list[dict[str, polars.DataType]] = []
    catalog_snapshots: dict[str, catalog_models._TableSchemaSnapshot] = {}
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    fast_hdf5_paths: list[lazynwb.types_.PathLike] = []
    fallback_jobs: list[
        tuple[lazynwb.types_.PathLike, Sequence[_FastCatalogBackendName] | None]
    ] = []
    for file_path in file_paths:
        backend_order = _fast_catalog_backend_order(file_path)
        if (
            backend_order[0] == "hdf5"
            and hdf5_reader._is_fast_hdf5_candidate(file_path)
        ):
            fast_hdf5_paths.append(file_path)
        else:
            fallback_jobs.append((file_path, None))
    if fast_hdf5_paths:
        logger.debug(
            "using batched fast HDF5 schema path for %d/%d files at %r",
            len(fast_hdf5_paths),
            len(file_paths),
            normalized_table_path,
        )
        for file_path, snapshot, used_fast_hdf5 in _run_async_schema_snapshot_batch(
            _get_fast_hdf5_table_schema_snapshots(
                fast_hdf5_paths,
                normalized_table_path,
                raise_on_missing,
            )
        ):
            if not used_fast_hdf5:
                fallback_jobs.append((file_path, ("zarr",)))
                continue
            if snapshot is not None:
                file_schema = catalog_polars._snapshot_to_polars_schema(snapshot)
                per_file_schemas.append(file_schema)
                catalog_snapshots[_catalog_snapshot_key(file_path)] = snapshot
    future_to_file_path = {}
    for file_path, fast_backend_order in fallback_jobs:
        future = lazynwb.utils.get_threadpool_executor().submit(
            _get_table_schema_helper,
            file_path=file_path,
            table_path=table_path,
            raise_on_missing=raise_on_missing,
            fast_backend_order=fast_backend_order,
        )
        future_to_file_path[future] = file_path
    is_first_missing = True  # used to warn only once
    for future in concurrent.futures.as_completed(future_to_file_path):
        try:
            file_schema = future.result()
        except lazynwb.exceptions.InternalPathError:
            if raise_on_missing:
                raise
            else:
                if is_first_missing:
                    logger.warning(f"Table {table_path!r} missing in one or more files")
                    is_first_missing = False
                continue
        except Exception as exc:
            logger.error(
                f"Error getting schema for {table_path!r} in {future_to_file_path[future]!r}:"
            )
            raise exc from None
        if file_schema is not None:
            per_file_schemas.append(file_schema)
    if not per_file_schemas:
        raise lazynwb.exceptions.InternalPathError(
            f"Table {table_path!r} not found in any files"
            + (
                f": try increasing `infer_schema_length` (currently: {first_n_files_to_infer_schema})"
                if first_n_files_to_infer_schema
                else ""
            )
        )

    # merge schemas and warn on inconsistent types:
    counts: dict[str, collections.Counter] = {}
    for file_schema in per_file_schemas:
        for column_name, pl_dtype in file_schema.items():
            if column_name not in counts:
                counts[column_name] = collections.Counter()
            counts[column_name][pl_dtype] += 1
    schema = pl.Schema()
    for column_name, counter in counts.items():
        if len(counter) > 1:
            logger.warning(
                f"Column {column_name!r} has inconsistent types across files - using most common: {counter}"
            )
        schema[column_name] = counter.most_common(1)[0][0]

    if not exclude_internal_columns:
        # add the internal columns to the schema:
        schema[NWB_PATH_COLUMN_NAME] = pl.String
        schema[TABLE_PATH_COLUMN_NAME] = pl.String
        schema[TABLE_INDEX_COLUMN_NAME] = pl.UInt32
    if exclude_array_columns:
        # remove the array columns from the schema:
        for column_name in tuple(schema.keys()):
            if isinstance(schema[column_name], (pl.List, pl.Array)):
                schema.pop(column_name, None)
    return _TableSchemaInferenceResult(
        schema=pl.Schema(schema),
        catalog_snapshots=catalog_snapshots,
    )


def get_table_schema(
    file_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    table_path: str,
    first_n_files_to_infer_schema: int | None = None,
    exclude_array_columns: bool = False,
    exclude_internal_columns: bool = False,
    raise_on_missing: bool = False,
) -> pl.Schema:
    return _get_table_schema_with_catalog_snapshots(
        file_paths=file_paths,
        table_path=table_path,
        first_n_files_to_infer_schema=first_n_files_to_infer_schema,
        exclude_array_columns=exclude_array_columns,
        exclude_internal_columns=exclude_internal_columns,
        raise_on_missing=raise_on_missing,
    ).schema


def insert_is_observed(
    intervals_frame: polars._typing.FrameType,
    units_frame: polars._typing.FrameType | None = None,
    col_name: str = "is_observed",
) -> polars._typing.FrameType:

    if isinstance(intervals_frame, pl.LazyFrame):
        intervals_lf = intervals_frame
    elif isinstance(intervals_frame, pd.DataFrame):
        intervals_lf = pl.from_pandas(intervals_frame).lazy()
    else:
        intervals_lf = intervals_frame.lazy()
    intervals_schema = intervals_lf.collect_schema()
    if not all(c in intervals_schema for c in ("start_time", "stop_time")):
        raise lazynwb.exceptions.ColumnError(
            "intervals_frame must contain 'start_time' and 'stop_time' columns"
        )

    if isinstance(units_frame, pl.LazyFrame):
        units_lf = units_frame
    elif isinstance(units_frame, pd.DataFrame):
        units_lf = pl.from_pandas(units_frame).lazy()
    elif isinstance(units_frame, pl.DataFrame):
        units_lf = units_frame.lazy()
    else:
        units_lf = (
            get_df(
                nwb_data_sources=intervals_lf.select(NWB_PATH_COLUMN_NAME)
                .collect()[NWB_PATH_COLUMN_NAME]
                .unique(),
                search_term="units",
                as_polars=True,
            )
            .pipe(merge_array_column, column_name="obs_intervals")
            .lazy()
        )

    units_lf = units_lf.rename(
        {TABLE_INDEX_COLUMN_NAME: UNITS_TABLE_INDEX_COLUMN_NAME}, strict=False
    )
    units_schema = units_lf.collect_schema()
    if "obs_intervals" not in units_schema:
        raise lazynwb.exceptions.ColumnError(
            "units frame does not contain 'obs_intervals' column"
        )
    unit_table_index_col = UNITS_TABLE_INDEX_COLUMN_NAME
    if unit_table_index_col not in units_schema:
        raise lazynwb.exceptions.ColumnError(
            f"units frame does not contain a row index column to link rows to original table position (e.g {TABLE_INDEX_COLUMN_NAME!r})"
        )
    unique_units = (
        units_lf.select(unit_table_index_col, NWB_PATH_COLUMN_NAME).collect().unique()
    )
    intervals_schema = intervals_lf.collect_schema()
    unique_intervals = (
        intervals_lf.select(unit_table_index_col, NWB_PATH_COLUMN_NAME)
        .collect()
        .unique()
    )
    if not all(d in unique_units.to_dicts() for d in unique_intervals.to_dicts()):
        raise ValueError(
            "units frame does not contain all unique units in intervals frame"
        )

    if units_schema["obs_intervals"] in (
        pl.List(pl.List(pl.Float64)),
        pl.List(pl.List(pl.Null)),
    ) or (
        isinstance(units_schema["obs_intervals"], pl.Array)
        and len(units_schema["obs_intervals"].shape) > 1
    ):
        logger.debug(
            "Exploding nested 'obs_intervals' column to create list[float] column for join"
        )
        units_lf = units_lf.explode("obs_intervals")
    assert (type_ := units_lf.collect_schema()["obs_intervals"]) in (
        pl.List(pl.Float64),
        pl.List(pl.Null),  # in case all obs_intervals are empty
        pl.Array(pl.Float64, shape=(2,)),
        pl.Array(pl.Float64, shape=(0,)),  # in case all obs_intervals are empty
    ), f"Expected exploded obs_intervals to be pl.List(f64) or pl.Array(f64), got {type_}"
    intervals_lf = (
        intervals_lf.join(
            units_lf.select(unit_table_index_col, "obs_intervals"),
            on=unit_table_index_col,
            how="left",
        )
        .cast({"obs_intervals": pl.List(pl.Float64)})  # before using list namespace
        .with_columns(
            pl.when(
                pl.col("obs_intervals").list.get(0).gt(pl.col("start_time"))
                | pl.col("obs_intervals").list.get(1).lt(pl.col("stop_time")),
            )
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(col_name),
        )
        .group_by(unit_table_index_col, NWB_PATH_COLUMN_NAME, "start_time")
        .agg(
            pl.all().exclude("obs_intervals", col_name).first(),
            pl.col(col_name).any(),
        )
    )
    if isinstance(intervals_frame, pl.LazyFrame):
        return intervals_lf
    return intervals_lf.collect()


def _spikes_times_in_intervals_helper(
    nwb_path: str,
    col_name_to_intervals: dict[str, tuple[pl.Expr, pl.Expr]],
    intervals_table_path: str,
    intervals_table_filter: str | pl.Expr | None,
    intervals_table_row_indices: Sequence[int] | None,
    units_table_indices: Sequence[int],
    apply_obs_intervals: bool,
    as_counts: bool,
    align_times: bool,
    keep_only_necessary_cols: bool,
) -> pl.DataFrame:
    units_df: pl.DataFrame = (
        get_df(nwb_path, search_term="units", exact_path=True, as_polars=True)
        .filter(pl.col(TABLE_INDEX_COLUMN_NAME).is_in(units_table_indices))
        .pipe(merge_array_column, column_name="spike_times")
    )
    if isinstance(intervals_table_filter, str):
        # pandas:
        intervals_df = pl.from_pandas(
            get_df(nwb_path, search_term=intervals_table_path, as_polars=False).query(
                intervals_table_filter
            )
        )
    elif isinstance(intervals_table_filter, pl.Expr):
        intervals_df = get_df(
            nwb_path, search_term=intervals_table_path, as_polars=True
        ).filter(intervals_table_filter)
    elif intervals_table_filter is None:
        intervals_df = get_df(
            nwb_path, search_term=intervals_table_path, as_polars=True
        )
    else:
        raise ValueError(
            f"`intervals_table_filter` must be str or pl.Expr or None, got {type(intervals_table_filter)}"
        )

    if intervals_table_row_indices is not None:
        intervals_df = intervals_df.filter(
            pl.col(TABLE_INDEX_COLUMN_NAME).is_in(intervals_table_row_indices)
        )

    temp_col_prefix = "__temp_interval"
    for col_name, (start, end) in col_name_to_intervals.items():
        intervals_df = intervals_df.with_columns(
            pl.concat_list(start, end).alias(f"{temp_col_prefix}_{col_name}"),
        )
    results: dict[str, list] = {
        UNITS_TABLE_INDEX_COLUMN_NAME: [],
        INTERVALS_TABLE_INDEX_COLUMN_NAME: [],
        NWB_PATH_COLUMN_NAME: [],
    }
    for col_name in col_name_to_intervals.keys():
        results[col_name] = []
    results[INTERVALS_TABLE_INDEX_COLUMN_NAME].extend(
        intervals_df[TABLE_INDEX_COLUMN_NAME].to_list() * len(units_df)
    )

    for row in units_df.iter_rows(named=True):
        results[UNITS_TABLE_INDEX_COLUMN_NAME].extend(
            [row[TABLE_INDEX_COLUMN_NAME]] * len(intervals_df)
        )
        results[NWB_PATH_COLUMN_NAME].extend([nwb_path] * len(intervals_df))

        for col_name in col_name_to_intervals:
            # get spike times with start:end interval for each row of the trials table
            spike_times = row["spike_times"]
            spikes_in_intervals: list[float | list[float]] = []
            for trial_idx, (a, b) in enumerate(
                np.searchsorted(
                    spike_times, intervals_df[f"{temp_col_prefix}_{col_name}"].to_list()
                )
            ):
                spike_times_in_interval = spike_times[a:b]
                #! spikes coincident with end of interval are not included
                if as_counts:
                    spikes_in_intervals.append(len(spike_times_in_interval))
                elif align_times:
                    start_time = intervals_df["start_time"].to_list()[trial_idx]
                    spikes_in_intervals.append(
                        [t - start_time for t in spike_times_in_interval]
                    )
                else:
                    spikes_in_intervals.append(spike_times_in_interval)
            results[col_name].extend(spikes_in_intervals)

    if keep_only_necessary_cols and not apply_obs_intervals:
        return pl.DataFrame(results)

    results_df = pl.DataFrame(results).join(
        other=intervals_df.drop(pl.selectors.starts_with(temp_col_prefix)),
        left_on=INTERVALS_TABLE_INDEX_COLUMN_NAME,
        right_on=TABLE_INDEX_COLUMN_NAME,
        how="inner",
    )

    if apply_obs_intervals:
        results_df = insert_is_observed(
            intervals_frame=results_df,
            units_frame=units_df.drop("spike_times").pipe(
                merge_array_column, column_name="obs_intervals"
            ),
        ).with_columns(
            *[
                pl.when(pl.col("is_observed").not_())
                .then(pl.lit(None))
                .otherwise(pl.col(col_name))
                .alias(col_name)
                for col_name in col_name_to_intervals
            ]
        )
        if keep_only_necessary_cols:
            results_df = results_df.drop(pl.all().exclude(NWB_PATH_COLUMN_NAME, UNITS_TABLE_INDEX_COLUMN_NAME, INTERVALS_TABLE_INDEX_COLUMN_NAME, *col_name_to_intervals.keys()))  # type: ignore[arg-type]

    return results_df


def _get_pl_df(df: FrameType) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    elif isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    assert isinstance(
        df, pl.DataFrame
    ), f"Expected pandas or polars dataframe, got {type(df)}"
    return df


def get_spike_times_in_intervals(
    filtered_units_df: FrameType,
    intervals: dict[str, tuple[pl.Expr, pl.Expr]],
    intervals_df: str | FrameType = "/intervals/trials",
    intervals_df_filter: str | pl.Expr | None = None,
    apply_obs_intervals: bool = True,
    as_counts: bool = False,
    keep_only_necessary_cols: bool = False,
    use_process_pool: bool = True,
    disable_progress: bool = False,
    as_polars: bool | None = None,
    align_times: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    """"""
    as_polars = lazynwb_config._resolve_as_polars(as_polars)
    if align_times and as_counts:
        raise ValueError(
            "Cannot use `align_times` and `as_counts` at the same time: please choose one"
        )
    units_df: pl.DataFrame = _get_pl_df(filtered_units_df)
    assert not isinstance(units_df, pl.LazyFrame)
    n_sessions = units_df[NWB_PATH_COLUMN_NAME].n_unique()

    if not isinstance(intervals_df, str):
        intervals_df_row_indices = _get_pl_df(intervals_df)[
            TABLE_INDEX_COLUMN_NAME
        ].to_list()
    else:
        intervals_df_row_indices = None  # all rows will be used when table fetched from NWB, but `filter` can be applied

    def _get_intervals_table_path(
        nwb_path: lazynwb.types_.PathLike,
        intervals_df: str | FrameType,
    ) -> str:
        if isinstance(intervals_df, str):
            return intervals_df
        return _get_original_table_path(
            _get_pl_df(intervals_df).filter(pl.col(NWB_PATH_COLUMN_NAME) == nwb_path)
        )

    results: list[pl.DataFrame] = []

    iterable: Iterable
    if n_sessions == 1 or not use_process_pool:
        iterable = units_df.group_by(NWB_PATH_COLUMN_NAME)
        if not disable_progress:
            iterable = tqdm.tqdm(
                iterable,
                desc="Getting spike times in intervals",
                unit="NWB",
                ncols=120,
            )
        for (nwb_path, *_), df in iterable:
            nwb_path = str(nwb_path)
            result = _spikes_times_in_intervals_helper(
                nwb_path=nwb_path,
                col_name_to_intervals=intervals,
                intervals_table_path=_get_intervals_table_path(nwb_path, intervals_df),
                intervals_table_filter=intervals_df_filter,
                intervals_table_row_indices=intervals_df_row_indices,
                units_table_indices=df[TABLE_INDEX_COLUMN_NAME].to_list(),
                apply_obs_intervals=apply_obs_intervals,
                as_counts=as_counts,
                keep_only_necessary_cols=keep_only_necessary_cols,
                align_times=align_times,
            )
            results.append(result)
    else:
        future_to_nwb_path = {}
        for (nwb_path, *_), df in units_df.group_by(NWB_PATH_COLUMN_NAME):
            nwb_path = str(nwb_path)
            future = lazynwb.utils.get_processpool_executor().submit(
                _spikes_times_in_intervals_helper,
                nwb_path=nwb_path,
                col_name_to_intervals=intervals,
                intervals_table_path=_get_intervals_table_path(nwb_path, intervals_df),
                intervals_table_filter=intervals_df_filter,
                intervals_table_row_indices=intervals_df_row_indices,
                units_table_indices=df[TABLE_INDEX_COLUMN_NAME].to_list(),
                apply_obs_intervals=apply_obs_intervals,
                as_counts=as_counts,
                keep_only_necessary_cols=keep_only_necessary_cols,
                align_times=align_times,
            )
            future_to_nwb_path[future] = nwb_path
        iterable = tuple(concurrent.futures.as_completed(future_to_nwb_path))
        if not disable_progress:
            iterable = tqdm.tqdm(
                iterable,
                desc="Getting spike times in intervals",
                unit="NWB",
                ncols=120,
            )
        for future in iterable:
            try:
                result = future.result()
            except Exception as exc:
                logger.error(
                    f"error getting spike times for {lazynwb.file_io.from_pathlike(future_to_nwb_path[future])}: {exc!r}"
                )
            else:
                results.append(result)
    columns_to_drop = pl.selectors.starts_with(TABLE_PATH_COLUMN_NAME)
    # original table paths are ambiguous now we've joined rows from units and trials
    # - we find all that start with, in case any joins added a suffix
    if keep_only_necessary_cols:
        df = pl.concat(results, how="diagonal_relaxed").drop(
            columns_to_drop, strict=False
        )
    else:
        df = (
            pl.concat(results, how="diagonal_relaxed")
            .join(
                pl.DataFrame(units_df),
                left_on=[UNITS_TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
                right_on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
                how="inner",
            )
            .drop(
                columns_to_drop, strict=False
            )  # original table paths are ambiguous now we've joined rows from units and trials
        )
    if as_polars:
        return df
    else:
        return df.to_pandas()


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
