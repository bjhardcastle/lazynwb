from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

import h5py
import npc_io
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
import zarr

from lazynwb.file_io import LazyFile, normalize_internal_file_path

pd.options.mode.copy_on_write = True

logger = logging.getLogger(__name__)

NWB_PATH_COLUMN_NAME = "_nwb_path"
TABLE_PATH_COLUMN_NAME = "_table_path"
TABLE_INDEX_COLUMN_NAME = "_table_index"

thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
process_pool_executor: concurrent.futures.ProcessPoolExecutor | None = None


def get_threadpool_executor() -> concurrent.futures.ThreadPoolExecutor:
    global thread_pool_executor
    if thread_pool_executor is None:
        thread_pool_executor = concurrent.futures.ThreadPoolExecutor()
    return thread_pool_executor


def get_processpool_executor() -> concurrent.futures.ProcessPoolExecutor:
    global process_pool_executor
    if process_pool_executor is None:
        process_pool_executor = concurrent.futures.ProcessPoolExecutor()
    return process_pool_executor


def _get_df_helper(nwb_path: npc_io.PathLike, **get_df_kwargs) -> pd.DataFrame:
    if isinstance(nwb_path, LazyFile):
        context = contextlib.nullcontext(nwb_path)
    else:
        context = LazyFile(nwb_path) # type: ignore[assignment]
    with context as file:
        return _get_df(
            file=file,
            **get_df_kwargs,
        )


def get_df(
    nwb_data_sources: npc_io.PathLike | Iterable[npc_io.PathLike | LazyFile] | LazyFile,
    table_path: str,
    exclude_column_names: str | Iterable[str] | None = None,
    exclude_indexed_columns: bool = False,
    use_process_pool: bool = False,
    disable_progress: bool = False,
) -> pd.DataFrame:
    t0 = time.time()

    if isinstance(nwb_data_sources, (str, LazyFile)) or not isinstance(
        nwb_data_sources, Iterable
    ):
        paths = (nwb_data_sources,)
    else:
        paths = tuple(nwb_data_sources)

    if len(paths) == 1:  # don't use a pool for a single file
        return _get_df_helper(
            nwb_path=paths[0],
            table_path=table_path,
            exclude_column_names=exclude_column_names,
            exclude_indexed_columns=exclude_indexed_columns,
        )

    if exclude_indexed_columns and use_process_pool:
        logger.warning(
            "exclude_indexed_columns is True: setting use_process_pool=False for speed"
        )
        use_process_pool = False

    executor = (
        get_processpool_executor() if use_process_pool else get_threadpool_executor()
    )
    future_to_path = {}
    results: list[pd.DataFrame] = []
    for path in paths:
        future = executor.submit(
            _get_df_helper,
            nwb_path=path,
            table_path=table_path,
            exclude_column_names=exclude_column_names,
            exclude_indexed_columns=exclude_indexed_columns,
        )
        future_to_path[future] = path
    futures = concurrent.futures.as_completed(future_to_path)
    if (
        use_process_pool and not disable_progress
    ):  # only show progress bar for process pool - threadpool will be fast
        futures = tqdm.tqdm(
            futures,
            total=len(future_to_path),
            desc=f"Getting multi-NWB {table_path} table",
            unit="NWB",
            ncols=80,
        )
    for future in futures:
        try:
            results.append(future.result())
        except:
            logger.exception(
                f"Error getting DataFrame for {npc_io.from_pathlike(future_to_path[future])}:"
            )
            raise
    df = pd.concat(results)
    logger.debug(
        f"created {table_path!r} DataFrame ({len(df)} rows) from {len(paths)} NWB files in {time.time() - t0:.2f} s"
    )
    return df


def _get_df(
    file: LazyFile,
    table_path: str,
    exclude_column_names: str | Iterable[str] | None = None,
    exclude_indexed_columns: bool = False,
) -> pd.DataFrame:
    t0 = time.time()
    column_accessors: dict[str, zarr.Array | h5py.Dataset] = (
        _get_table_column_accessors(
            file=file,
            table_path=table_path,
            use_thread_pool=(file._hdmf_backend == LazyFile.HDMFBackend.ZARR),
        )
    )

    if isinstance(exclude_column_names, str):
        exclude_column_names = (exclude_column_names,)
    elif exclude_column_names is not None:
        exclude_column_names = tuple(exclude_column_names)
    for name in tuple(column_accessors.keys()):
        is_indexed = exclude_indexed_columns and is_indexed_column(
            name, column_accessors.keys()
        )
        is_excluded = exclude_column_names is not None and name in exclude_column_names
        if is_indexed or is_excluded:
            column_accessors.pop(name, None)
            column_accessors.pop(f"{name}_index", None)
            column_accessors.pop(name.removesuffix("_index"), None)

    # indexed columns (columns containing lists) need to be handled differently:
    indexed_column_names: set[str] = get_indexed_column_names(column_accessors.keys())
    non_indexed_column_names = column_accessors.keys() - indexed_column_names
    # some columns have >2 dims but no index - they also need to be handled differently
    multi_dim_column_names = []

    column_data: dict[str, npt.NDArray | list[list]] = {}
    logger.debug(
        f"materializing non-indexed columns for {file._path}/{table_path}: {non_indexed_column_names}"
    )
    for column_name in non_indexed_column_names:
        if (ndim := column_accessors[column_name].ndim) >= 2:
            logger.warning(
                f"non-indexed column {column_name!r} has {ndim=}: will be treated as an indexed column"
            )
            multi_dim_column_names.append(column_name)
            continue
        if column_accessors[column_name].dtype.kind in ("S", "O"):
            column_data[column_name] = column_accessors[column_name][:].astype(str)
        else:
            column_data[column_name] = column_accessors[column_name][:]

    if indexed_column_names and not exclude_indexed_columns:
        data_column_names = {
            name for name in indexed_column_names if not name.endswith("_index")
        }
        logger.debug(
            f"materializing indexed columns for {file._path}/{table_path}: {data_column_names}"
        )
        for column_name in data_column_names:
            column_data[column_name] = get_indexed_column_data(
                data_column_accessor=column_accessors[column_name],
                index_column_accessor=column_accessors[f"{column_name}_index"],
            )
        for column_name in multi_dim_column_names:
            column_data[column_name] = column_accessors[column_name][:].tolist()

    column_length = len(next(iter(column_data.values())))

    # add identifiers to each row, so they can be linked back their source at a later time:
    identifier_column_data = {
        NWB_PATH_COLUMN_NAME: [file._path.as_posix()] * column_length,
        TABLE_PATH_COLUMN_NAME: [normalize_internal_file_path(table_path)] * column_length,
        TABLE_INDEX_COLUMN_NAME: np.arange(column_length),
    }
    df = pd.DataFrame(data=column_data | identifier_column_data)
    logger.debug(
        f"initialized DataFrame for {file._path}/{table_path} in {time.time() - t0:.2f} s"
    )
    return df


def get_indexed_column_data(
    data_column_accessor: zarr.Array | h5py.Dataset,
    index_column_accessor: zarr.Array | h5py.Dataset,
    table_row_indices: Sequence[int] | None = None,
) -> list[list[np.float64 | list[np.float64]]]:
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
    index_array: npt.NDArray[np.int32] = np.concatenate(([0], index_column_accessor[:]))  # small enough to read in one go
    if table_row_indices is None:
        table_row_indices = list(range(len(index_array) - 1)) # -1 because of the inserted 0 above
    data_indices: list[int] = []
    for i in table_row_indices:
        data_indices.extend(range(index_array[i], index_array[i + 1]))
    assert len(data_indices) == np.sum(np.diff(index_array)[table_row_indices]), "length of data_indices is incorrect"

    # read actual data and split into sub-vectors for each row of the table:
    data_array: list[np.float64 | list[np.float64]] = data_column_accessor[data_indices]
    column_data = []
    start_idx = 0
    for run_length in np.diff(index_array)[table_row_indices]: 
        end_idx = start_idx + run_length
        column_data.append(data_array[start_idx:end_idx])
        start_idx = end_idx
    return column_data


def is_indexed_column(column_name: str, all_column_names: Iterable[str]) -> bool:
    """
    >>> is_indexed_column('spike_times', ['spike_times', 'spike_times_index'])
    True
    >>> is_indexed_column('spike_times_index', ['spike_times', 'spike_times_index'])
    True
    >>> is_indexed_column('spike_times', ['spike_times'])
    False
    >>> is_indexed_column('unit_index', ['unit_index'])
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
    return {k for k in column_names if is_indexed_column(k, column_names)}

def _indexed_column_helper(
    nwb_path: npc_io.PathLike,
    table_path: str,
    column_name: str,
    table_row_indices: Sequence[int],
) -> pd.DataFrame:
    with LazyFile(nwb_path) as file:
        data_column_accessor = file[table_path][column_name]
        if data_column_accessor.ndim >= 2:
            column_data = data_column_accessor[table_row_indices].tolist()
        else:
            column_data = get_indexed_column_data(
                data_column_accessor=data_column_accessor,
                index_column_accessor=file[table_path][f"{column_name}_index"],
                table_row_indices=table_row_indices,
            )
    return pd.DataFrame(
        {
            column_name: column_data,
            TABLE_INDEX_COLUMN_NAME: table_row_indices,
            NWB_PATH_COLUMN_NAME: [nwb_path] * len(table_row_indices),
        },
    )

def merge_array_column(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    column_data: list[pd.DataFrame] = []
    future_to_path = {}
    for nwb_path, session_df in df.groupby(NWB_PATH_COLUMN_NAME):
        future = get_threadpool_executor().submit(
            _indexed_column_helper,
            nwb_path=nwb_path,
            table_path=session_df[TABLE_PATH_COLUMN_NAME].iloc[0],
            column_name=column_name,
            table_row_indices=session_df[TABLE_INDEX_COLUMN_NAME].values,
        )
        future_to_path[future] = nwb_path
    for future in concurrent.futures.as_completed(future_to_path):
        try:
            column_data.append(future.result())
        except:
            logger.error(
                f"Error getting indexed column data for {npc_io.from_pathlike(future_to_path[future])}:"
            )
            raise
    return df.merge(
        pd.concat(column_data), 
        how="left",
        on=[TABLE_INDEX_COLUMN_NAME, NWB_PATH_COLUMN_NAME],
    ).set_index(df[TABLE_INDEX_COLUMN_NAME].values)


def _get_table_column_accessors(
    file: LazyFile,
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
            get_threadpool_executor().submit(
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


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
