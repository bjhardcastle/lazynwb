from __future__ import annotations

import concurrent.futures
import contextlib
import difflib
import logging
import time
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar

import h5py
import npc_io
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars._typing
import polars.datatypes.convert
import tqdm
import zarr

import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.utils

pd.options.mode.copy_on_write = True

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame, pd.DataFrame)

logger = logging.getLogger(__name__)

NWB_PATH_COLUMN_NAME = "_nwb_path"
TABLE_PATH_COLUMN_NAME = "_table_path"
TABLE_INDEX_COLUMN_NAME = "_table_index"


def _get_df_helper(nwb_path: npc_io.PathLike, **get_df_kwargs) -> dict[str, Any]:
    if isinstance(nwb_path, lazynwb.file_io.FileAccessor):
        context = contextlib.nullcontext(nwb_path)
    else:
        context = lazynwb.file_io.FileAccessor(nwb_path)  # type: ignore[assignment]
    with context as file:
        return _get_table_data(
            file=file,
            **get_df_kwargs,
        )


@typing.overload
def get_df(
    nwb_data_sources: (
        str
        | npc_io.PathLike
        | lazynwb.file_io.FileAccessor
        | Iterable[str | npc_io.PathLike | lazynwb.file_io.FileAccessor]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    table_row_indices: Sequence[int] | None = None,
    exclude_array_columns: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    suppress_errors: bool = False,
    as_polars: Literal[False] = False,
) -> pd.DataFrame: ...


@typing.overload
def get_df(
    nwb_data_sources: (
        str
        | npc_io.PathLike
        | lazynwb.file_io.FileAccessor
        | Iterable[str | npc_io.PathLike | lazynwb.file_io.FileAccessor]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    table_row_indices: Sequence[int] | None = None,
    exclude_array_columns: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    suppress_errors: bool = False,
    as_polars: Literal[True] = True,
) -> pl.DataFrame: ...


def get_df(
    nwb_data_sources: (
        str
        | npc_io.PathLike
        | lazynwb.file_io.FileAccessor
        | Iterable[str | npc_io.PathLike | lazynwb.file_io.FileAccessor]
    ),
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    table_row_indices: Sequence[int] | None = None,
    exclude_array_columns: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    suppress_errors: bool = False,
    as_polars: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    """ ""Get a DataFrame from one or more NWB files.

    Parameters
    ----------
    nwb_data_sources : str, PathLike, FileAccessor, or iterable of these
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
    suppress_errors : bool, default False
        If True, any errors encountered while reading the files will be suppressed and a warning
        will be logged.
    as_polars : bool, default False
        If True, a Polars DataFrame will be returned. Otherwise, a Pandas DataFrame will be returned.
    """
    t0 = time.time()

    if isinstance(
        nwb_data_sources, (str, lazynwb.file_io.FileAccessor)
    ) or not isinstance(nwb_data_sources, Iterable):
        paths = (nwb_data_sources,)
    else:
        paths = tuple(nwb_data_sources)

    # speedup known table locations:
    if search_term == "trials":
        search_term = "/intervals/trials"
        exact_path = True


    if len(paths) == 1:  # don't use a pool for a single file
        frame_cls = pl.DataFrame if as_polars else pd.DataFrame
        return frame_cls(
            _get_df_helper(
                nwb_path=paths[0],
                search_term=search_term,
                exact_path=exact_path,
                exclude_column_names=exclude_column_names,
                include_column_names=include_column_names,
                exclude_array_columns=exclude_array_columns,
                table_row_indices=table_row_indices,
            )
        )

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
    results: list[dict] = []
    for path in paths:
        future = executor.submit(
            _get_df_helper,
            nwb_path=path,
            search_term=search_term,
            exact_path=exact_path,
            exclude_column_names=exclude_column_names,
            include_column_names=include_column_names,
            exclude_array_columns=exclude_array_columns,
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
                    f"Table {search_term!r} not found in {npc_io.from_pathlike(future_to_path[future])}"
                )
                continue
        except Exception:
            if not suppress_errors:
                raise
            else:
                logger.exception(
                    f"Error getting DataFrame for {npc_io.from_pathlike(future_to_path[future])}:"
                )
                continue
    if not as_polars:
        df = pd.concat((pd.DataFrame(r) for r in results), ignore_index=True)
    else:
        df = pl.concat(
            (pl.DataFrame(r) for r in results), how="diagonal_relaxed", rechunk=True
        )
    logger.debug(
        f"Created {search_term!r} DataFrame ({len(df)} rows) from {len(paths)} NWB files in {time.time() - t0:.2f} s"
    )
    return df


def _get_table_data(
    file: lazynwb.file_io.FileAccessor,
    search_term: str,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    exclude_array_columns: bool = True,
    table_row_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    if (
        not exact_path
        and lazynwb.utils.normalize_internal_file_path(search_term) not in file
    ):
        path_to_accessor = _get_internal_file_paths(file._accessor)
        matches = difflib.get_close_matches(
            search_term, path_to_accessor.keys(), n=1, cutoff=0.3
        )
        if not matches:
            raise KeyError(f"Table {search_term!r} not found in {file._path}")
        match_ = matches[0]
        if (
            search_term not in match_
            or len([k for k in path_to_accessor if match_ in k]) > 1
        ):
            # only warn if there are multiple matches or if user-provided search term is not a
            # substring of the match
            logger.warning(f"Using {match_!r} instead of {search_term!r}")
        search_term = match_
    column_accessors: dict[str, zarr.Array | h5py.Dataset] = (
        _get_table_column_accessors(
            file=file,
            table_path=search_term,
            use_thread_pool=(
                file._hdmf_backend == lazynwb.file_io.FileAccessor.HDMFBackend.ZARR
            ),
        )
    )

    if isinstance(exclude_column_names, str):
        exclude_column_names = (exclude_column_names,)
    elif exclude_column_names is not None:
        exclude_column_names = tuple(exclude_column_names)
    if isinstance(include_column_names, str):
        include_column_names = (include_column_names,)
    elif include_column_names is not None:
        include_column_names = tuple(include_column_names)
    if include_column_names and exclude_column_names:
        ambiguous_column_names = set(include_column_names).intersection(
            exclude_column_names
        )
        if ambiguous_column_names:
            raise ValueError(
                f"Column names {ambiguous_column_names} are both included and excluded: unclear how to proceed"
            )
    # get filtered set of column names:
    for name in tuple(column_accessors.keys()):
        is_indexed = is_nominally_indexed_column(name, column_accessors.keys())
        is_excluded = exclude_column_names is not None and name in exclude_column_names
        is_included = include_column_names is not None and name in include_column_names
        is_not_included = (
            include_column_names is not None and name not in include_column_names
        )
        if (
            is_not_included
            or is_excluded
            or (exclude_array_columns and is_indexed and not is_included)
        ):
            column_accessors.pop(name, None)
            column_accessors.pop(f"{name}_index", None)
            column_accessors.pop(name.removesuffix("_index"), None)

    # indexed columns (columns containing lists) need to be handled differently:
    indexed_column_names: set[str] = get_indexed_column_names(column_accessors.keys())
    non_indexed_column_names = column_accessors.keys() - indexed_column_names
    # some columns have >2 dims but no index - they also need to be handled differently
    multi_dim_column_names = []

    column_data: dict[str, npt.NDArray | list[npt.NDArray]] = {}
    logger.debug(
        f"materializing non-indexed columns for {file._path}/{search_term}: {non_indexed_column_names}"
    )
    if table_row_indices is not None:
        _idx: Sequence[int] | slice = table_row_indices
    else:
        _idx = slice(None)
    for column_name in non_indexed_column_names:
        if (ndim := column_accessors[column_name].ndim) >= 2:
            logger.debug(
                f"non-indexed column {column_name!r} has {ndim=}: will be treated as an indexed column"
            )
            multi_dim_column_names.append(column_name)
            continue
        if column_accessors[column_name].dtype.kind in ("S", "O"):
            column_data[column_name] = column_accessors[column_name][_idx].astype(str)
        else:
            column_data[column_name] = column_accessors[column_name][_idx]

    if not exclude_array_columns and indexed_column_names:
        data_column_names = {
            name for name in indexed_column_names if not name.endswith("_index")
        }
        logger.debug(
            f"materializing indexed columns for {file._path}/{search_term}: {data_column_names}"
        )
        for column_name in data_column_names:
            column_data[column_name] = get_indexed_column_data(
                data_column_accessor=column_accessors[column_name],
                index_column_accessor=column_accessors[f"{column_name}_index"],
                table_row_indices=table_row_indices,
            )
    if not exclude_array_columns and multi_dim_column_names:
        logger.debug(
            f"materializing multi-dimensional array columns for {file._path}/{search_term}: {multi_dim_column_names}"
        )
        for column_name in multi_dim_column_names:
            column_data[column_name] = _format_multi_dim_column(
                column_accessors[column_name][_idx]
            )

    column_length = len(next(iter(column_data.values())))

    # add identifiers to each row, so they can be linked back their source at a later time:
    identifier_column_data = {
        NWB_PATH_COLUMN_NAME: [file._path.as_posix()] * column_length,
        TABLE_PATH_COLUMN_NAME: [
            lazynwb.utils.normalize_internal_file_path(search_term)
        ]
        * column_length,
        TABLE_INDEX_COLUMN_NAME: np.arange(column_length),
    }
    logger.debug(
        f"fetched data for {file._path}/{search_term} in {time.time() - t0:.2f} s"
    )
    return column_data | identifier_column_data


def get_indexed_column_data(
    data_column_accessor: zarr.Array | h5py.Dataset,
    index_column_accessor: zarr.Array | h5py.Dataset,
    table_row_indices: Sequence[int] | None = None,
) -> list[npt.NDArray[np.float64]]:
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
    # get indices in the data array for all requested rows, so we can read from accessor in one go:
    index_array: npt.NDArray[np.int32] = np.concatenate(
        ([0], index_column_accessor[:])
    )  # small enough to read in one go
    if table_row_indices is None:
        table_row_indices = list(
            range(len(index_array) - 1)
        )  # -1 because of the inserted 0 above
    data_indices: list[int] = []
    for i in table_row_indices:
        data_indices.extend(range(index_array[i], index_array[i + 1]))
    assert len(data_indices) == np.sum(
        np.diff(index_array)[table_row_indices]
    ), "length of data_indices is incorrect"

    # read actual data and split into sub-vectors for each row of the table:
    data_array: npt.NDArray[np.float64] = data_column_accessor[data_indices]
    column_data = []
    start_idx = 0
    for run_length in np.diff(index_array)[table_row_indices]:
        end_idx = start_idx + run_length
        column_data.append(data_array[start_idx:end_idx])
        start_idx = end_idx
    return column_data


def is_nominally_indexed_column(column_name: str, all_column_names: Iterable[str]) -> bool:
    """
    >>> is_nominally_indexed_column('spike_times', ['spike_times', 'spike_times_index'])
    True
    >>> is_nominally_indexed_column('spike_times_index', ['spike_times', 'spike_times_index'])
    True
    >>> is_nominally_indexed_column('spike_times', ['spike_times'])
    False
    >>> is_nominally_indexed_column('unit_index', ['unit_index'])
    False
    """
    all_column_names = set(all_column_names)  # in case object is an iterator
    if column_name not in all_column_names:
        return False
    if column_name.endswith("_index"):
        return column_name.removesuffix("_index") in all_column_names
    else:
        return f"{column_name}_index" in all_column_names


def get_indexed_column_names(column_names: Iterable[str]) -> set[str]:
    """
    >>> get_indexed_columns(['spike_times', 'presence_ratio'])
    set()
    >>> sorted(get_indexed_columns(['spike_times', 'spike_times_index', 'presence_ratio']))
    ['spike_times', 'spike_times_index']
    """
    return {k for k in column_names if is_nominally_indexed_column(k, column_names)}


def _indexed_column_helper(
    nwb_path: npc_io.PathLike,
    table_path: str,
    column_name: str,
    table_row_indices: Sequence[int],
) -> pd.DataFrame:
    with lazynwb.file_io.FileAccessor(nwb_path) as file:
        try:
            data_column_accessor = file[table_path][column_name]
        except KeyError as exc:
            if exc.args[0] == column_name:
                raise lazynwb.exceptions.ColumnError(column_name) from None
            elif exc.args[0] == table_path:
                raise lazynwb.exceptions.InternalPathError(table_path) from None
            else:
                raise
        if data_column_accessor.ndim >= 2:
            column_data = data_column_accessor[table_row_indices][:]
        else:
            column_data = get_indexed_column_data(
                data_column_accessor=data_column_accessor,
                index_column_accessor=file[table_path][f"{column_name}_index"],
                table_row_indices=table_row_indices,
            )
    return pd.DataFrame(
        {
            column_name: _format_multi_dim_column(column_data),
            TABLE_INDEX_COLUMN_NAME: table_row_indices,
            NWB_PATH_COLUMN_NAME: [nwb_path] * len(table_row_indices),
        },
    )


def _format_multi_dim_column(
    column_data: npt.NDArray | list[npt.NDArray],
) -> list[npt.NDArray]:
    """Pandas inists 'Per-column arrays must each be 1-dimensional': this converts to a list of
    arrays, if not already"""
    if isinstance(column_data, list):
        return column_data
    return list(column_data)


def get_table_path(df: FrameType, assert_unique: bool = True) -> str:
    if isinstance(df, pl.LazyFrame):
        df = df.select(TABLE_PATH_COLUMN_NAME).collect()  # type: ignore[assignment]
    assert not isinstance(df, pl.LazyFrame)
    series = df[TABLE_PATH_COLUMN_NAME]
    if assert_unique:
        assert len(set(series)) == 1, f"multiple table paths found: {series.unique()}"
    return series[0]


def get_table_column(df: FrameType, column_name: str) -> list[Any]:
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
            _indexed_column_helper,
            nwb_path=nwb_path,
            table_path=get_table_path(session_df, assert_unique=True),
            column_name=column_name,
            table_row_indices=get_table_column(session_df, TABLE_INDEX_COLUMN_NAME),
        )
        future_to_path[future] = nwb_path
    missing_column_already_warned = False
    for future in concurrent.futures.as_completed(future_to_path):
        try:
            column_data.append(future.result())
        except lazynwb.exceptions.ColumnError as exc:
            if not missing_ok:
                logger.error(
                    f"error getting indexed column data for {npc_io.from_pathlike(future_to_path[future])}:"
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
                f"error getting indexed column data for {npc_io.from_pathlike(future_to_path[future])}:"
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
            pl.concat((pl.from_pandas(d) for d in column_data), how="diagonal_relaxed"),
            on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
            how="left",
        )


def _get_table_column_accessors(
    file: lazynwb.file_io.FileAccessor,
    table_path: str,
    use_thread_pool: bool = False,
) -> dict[str, zarr.Array | h5py.Dataset]:
    """Get the accessor objects for each column of an NWB table, as a dict of zarr.Array or
    h5py.Dataset objects. Note that the data from each column is not read into memory.

    Optionally use a thread pool to speed up retrieval of the columns - faster for zarr files.
    """
    names_to_columns: dict[str, zarr.Array | h5py.Dataset] = {}
    t0 = time.time()
    if use_thread_pool:
        future_to_column = {
            lazynwb.utils.get_threadpool_executor().submit(
                file[table_path].get, column_name
            ): column_name
            for column_name in file[table_path].keys()
        }
        for future in concurrent.futures.as_completed(future_to_column):
            column_name = future_to_column[future]
            names_to_columns[column_name] = future.result()
    else:
        for column_name in file[table_path]:
            names_to_columns[column_name] = file[table_path].get(column_name)
    logger.debug(
        f"retrieved {len(names_to_columns)} column accessors from {file._accessor} in {time.time() - t0:.2f} s ({use_thread_pool=})"
    )
    return names_to_columns

def _get_polars_dtype(dataset: zarr.Array | h5py.Dataset) -> polars._typing.PolarsDataType:
    dtype = dataset.dtype
    if dtype == 'O':
        return pl.String
    return polars.datatypes.convert.numpy_char_code_to_dtype(dtype)

def _get_table_schema(
    files: lazynwb.file_io.FileAccessor | Sequence[lazynwb.file_io.FileAccessor],
    table_path: str,
    first_n_files_to_read: int | None = 1,
    include_array_columns: bool = True,
) -> pl.Schema:
    if isinstance(files, lazynwb.file_io.FileAccessor):
        files = [files]
    if first_n_files_to_read is not None:
        files = files[:first_n_files_to_read]
    schema = {}
    for file in files:
        schema.update({
            name: _get_polars_dtype(dataset)
            for name, dataset
            in _get_table_column_accessors(file, table_path).items()
        })
    if include_array_columns:
        # the _index column won't be part of the final dataframe, so don't include in schema
        schema = {k:v for k, v in schema.items() if not (k in get_indexed_column_names(schema) and k.endswith('_index'))}
    else:
        schema = {k:v for k, v in schema.items() if k not in get_indexed_column_names(schema)}
        if table_path == 'units':
            # in older version of nwb these large array cols are not indexed:
            schema.pop('waveform_sd')
            schema.pop('waveform_mean', None)

    return pl.Schema(schema)

def _get_internal_file_paths(
    group: h5py.Group | zarr.Group | zarr.Array,
    exclude_specifications: bool = True,
    exclude_table_columns: bool = True,
    exclude_metadata: bool = True,
) -> dict[str, h5py.Dataset | zarr.Array]:
    results: dict[str, h5py.Dataset | zarr.Array] = {}
    if exclude_specifications and group.name == "/specifications":
        return results
    if not hasattr(group, "keys") or (
        exclude_table_columns and "colnames" in getattr(group, "attrs", {})
    ):
        if exclude_metadata and (
            group.name.count("/") == 1 or group.name.startswith("/general")
        ):
            return {}
        else:
            results[group.name] = group
            return results
    for subpath in group.keys():
        try:
            results = {
                **results,
                **_get_internal_file_paths(
                    group[subpath],
                    exclude_specifications=exclude_specifications,
                    exclude_table_columns=exclude_table_columns,
                    exclude_metadata=exclude_metadata,
                ),
            }
        except (AttributeError, IndexError, TypeError):
            results[group.name] = group
    return results


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
        {TABLE_INDEX_COLUMN_NAME: f"{TABLE_INDEX_COLUMN_NAME}_units"}, strict=False
    )
    units_schema = units_lf.collect_schema()
    if "obs_intervals" not in units_schema:
        raise lazynwb.exceptions.ColumnError(
            "units frame does not contain 'obs_intervals' column"
        )
    unit_table_index_col = f"{TABLE_INDEX_COLUMN_NAME}_units"
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
        pl.List(pl.List(pl.Float64())),
        pl.List(pl.List(pl.Int64())),
        pl.List(pl.List(pl.Null())),
    ):
        logger.info("Converting 'obs_intervals' column to list of lists")
        units_lf = units_lf.explode("obs_intervals")
    assert (type_ := units_lf.collect_schema()["obs_intervals"]) == pl.List(
        pl.Float64
    ), f"Expected exploded obs_intervals to be pl.List(f64), got {type_}"
    intervals_lf = (
        intervals_lf.join(
            units_lf.select(unit_table_index_col, "obs_intervals"),
            on=unit_table_index_col,
            how="left",
        )
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
    keep_only_necessary_cols: bool,
) -> dict[str, list[int | list[float]]]:
    units_df: pl.DataFrame = (
        get_df(nwb_path, search_term="units", as_polars=True)
        # TODO speedup by only getting rows in initial get_df for units requested
        .filter(pl.col(TABLE_INDEX_COLUMN_NAME).is_in(units_table_indices)).pipe(
            merge_array_column, column_name="spike_times"
        )
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
    trials_id_col = f"{TABLE_INDEX_COLUMN_NAME}_trials"
    units_id_col = f"{TABLE_INDEX_COLUMN_NAME}_units"
    results: dict[str, list] = {
        units_id_col: [],
        trials_id_col: [],
    }
    for col_name in col_name_to_intervals.keys():
        results[col_name] = []
    results[trials_id_col].extend(
        intervals_df[TABLE_INDEX_COLUMN_NAME].to_list() * len(units_df)
    )

    for row in units_df.iter_rows(named=True):
        results[units_id_col].extend([row[TABLE_INDEX_COLUMN_NAME]] * len(intervals_df))

        for col_name, (start, end) in col_name_to_intervals.items():
            # get spike times with start:end interval for each row of the trials table
            spike_times = row["spike_times"]
            spikes_in_intervals: list[float | list[float]] = []
            for a, b in np.searchsorted(
                spike_times, intervals_df[f"{temp_col_prefix}_{col_name}"].to_list()
            ):
                spike_times_in_interval = spike_times[a:b]
                #! spikes coincident with end of interval are not included
                if as_counts:
                    spikes_in_intervals.append(len(spike_times_in_interval))
                else:
                    spikes_in_intervals.append(spike_times_in_interval)
            results[col_name].extend(spikes_in_intervals)

    if keep_only_necessary_cols and not apply_obs_intervals:
        return results

    results_df = pl.DataFrame(results).join(
        other=intervals_df.drop(pl.selectors.starts_with(temp_col_prefix)),
        left_on=trials_id_col,
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
            results_df = results_df.drop(pl.all().exclude(units_id_col, trials_id_col, *col_name_to_intervals.keys()))  # type: ignore[arg-type]

    return results_df.to_dict(as_series=False)


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
    as_polars: bool = False,
) -> pl.DataFrame:
    """"""
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
        nwb_path: str | npc_io.PathLike | lazynwb.file_io.FileAccessor,
        intervals_df: str | FrameType,
    ) -> str:
        if isinstance(intervals_df, str):
            return intervals_df
        return get_table_path(
            _get_pl_df(intervals_df).filter(pl.col(NWB_PATH_COLUMN_NAME) == nwb_path)
        )

    results: list[pl.DataFrame] = []

    def _handle_result(result):
        if not all(
            len(v) == len(result[f"{TABLE_INDEX_COLUMN_NAME}_trials"])
            for v in result.values()
        ):
            return
        results.append(pl.DataFrame(result))

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
            )
            _handle_result(result)
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
                    f"error getting spike times for {npc_io.from_pathlike(future_to_nwb_path[future])}: {exc!r}"
                )
            else:
                _handle_result(result)
    if keep_only_necessary_cols:
        df = pl.concat(results, how="diagonal_relaxed").drop(
            pl.selectors.starts_with(TABLE_PATH_COLUMN_NAME), strict=False
        )  # table paths is ambiguous now we've joined rows from units and trials
    else:
        df = (
            pl.concat(results, how="diagonal_relaxed")
            .join(
                pl.DataFrame(units_df),
                left_on=[f"{TABLE_INDEX_COLUMN_NAME}_units", NWB_PATH_COLUMN_NAME],
                right_on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
                how="inner",
            )
            .drop(
                pl.selectors.starts_with(TABLE_PATH_COLUMN_NAME), strict=False
            )  # table paths is ambiguous now we've joined rows from units and trials
        )
    if as_polars:
        return df
    else:
        return df.to_pandas()


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
