from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import time
from collections.abc import Generator, Iterable
from typing import Any
import typing

import h5py
import npc_io
import numpy as np
import numpy.typing as npt
import polars as pl
import zarr
from polars.type_aliases import PolarsDataType, PythonDataType

from lazynwb.file_io import LazyFile

logger = logging.getLogger(__name__)

def get_df(nwb: LazyFile, table_path: str) -> pl.DataFrame:
    t0 = time.time()
    data = _get_table_data(nwb, table_path=table_path, use_thread_pool=isinstance(nwb.units, zarr.Group))
    data = {k: data[k] for k in _get_filtered_units_column_names(data.keys())}
    # generator_data = {k: _get_data_generator(v) for k, v in data.items()}
    non_generator_data = {k: v[:] for k, v in data.items()}
    _overrides = {k: _get_polars_schema_override(data[k]) for k in data}
    schema_overrides = {k: v for k, v in _overrides.items() if v is not None}
    lf = pl.DataFrame(data=non_generator_data, schema_overrides=schema_overrides)
    logger.debug(f"initialized {table_path!r} DataFrame from {nwb._data} in {time.time() - t0:.2f} s")
    return lf
    
def get_units(nwb: LazyFile) -> pl.DataFrame:
    """
    Get the units table as a polars LazyFrame. Excludes the spike_times and obs_intervals columns.

    Examples:
        >>> nwb = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> lf = get_units(nwb) 
        >>> lf                          # doctest: +SKIP
        <LazyFrame [41 cols, {"amplitude_cutoff": Float64 … "waveform_sd": List(Float64)}] at 0x220B71F4750>        
        
        >>> nwb = LazyFile('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> lf = get_units(nwb) 
        >>> lf                          # doctest: +SKIP
        <LazyFrame [38 cols, {"amplitude_cutoff": Float64 … "waveform_sd": List(Float64)}] at 0x7FC93DB97490>
        
    """
    t0 = time.time()
    data = _get_table_data(nwb, table_path='units', use_thread_pool=isinstance(nwb.units, zarr.Group))
    data = {k: data[k] for k in _get_filtered_units_column_names(data.keys())}
    # generator_data = {k: _get_data_generator(v) for k, v in data.items()}
    non_generator_data = {k: v[:] for k, v in data.items()}
    _overrides = {k: _get_polars_schema_override(data[k]) for k in data}
    schema_overrides = {k: v for k, v in _overrides.items() if v is not None}
    lf = pl.DataFrame(data=non_generator_data, schema_overrides=schema_overrides)
    logger.debug(f"initialized units DataFrame from {nwb._data} in {time.time() - t0:.2f} s")
    return lf

def _get_filtered_units_column_names(names: Iterable[str]) -> Generator[str, None, None]:
    names = tuple(names)
    for name in names:
        if name in ("spike_times", "waveform_mean", "waveform_sd"):
            continue
        # skip indexed columns:
        if name.endswith("_index") or f"{name}_index" in names:
            continue
        yield name

def _get_table_data(nwb: LazyFile, table_path: str, use_thread_pool: bool = False) -> dict[str, zarr.Array | h5py.Dataset]:
    """Get the units table as a dict of zarr.Array or h5py.Dataset objects. 
    
    Optionally use a thread pool to speed up retrieval of the columns - faster for zarr files.
    """
    data: dict[str, zarr.Array | h5py.Dataset] = {}
    t0 = time.time()
    if use_thread_pool:
        future_to_column = {}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future_to_column = {pool.submit(nwb[table_path].get, column_name): column_name for column_name in nwb.units}
        for future in concurrent.futures.as_completed(future_to_column):
            column_name = future_to_column[future]
            data[column_name] = future.result()
    else:
        data = {column_name: nwb[table_path].get(column_name) for column_name in nwb[table_path]}
    logger.debug(f"retrieved units columns from {nwb._data} in {time.time() - t0:.2f} s ({use_thread_pool=})")
    return data

def _get_polars_schema_override(data: zarr.Array | h5py.Dataset) -> pl.DataType | None:
    if data.dtype.kind == 'O' or 'name' in data.name.split('/')[-1]:
        return pl.String()
    return None

def _get_data_generator(data: h5py.Dataset | zarr.Array) -> Generator[PolarsDataType | PythonDataType, None, None]:
    yield from iter(data)


def get_spike_times(nwb: LazyFile, unit_idx: Iterable[int]) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Get the spike times for a single unit, from its index in the units table.

    Examples:
        >>> nwb = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_spike_times(nwb, 0)
        (array([2925.85956667, 2931.19676667, 2944.81003333, ...,
           6696.99163333, 6700.57663333, 6700.89296667]),)
    """
    return get_indexed_table_column(nwb, table_path="units", column="spike_times", idx=unit_idx)

def get_obs_intervals(nwb: LazyFile, unit_idx: Iterable[int]) -> tuple[tuple[float, float], ...] | None:
    """Get the observation intervals for a single unit, from its index in the
    units table.
    
    Examples:
    
        >>> nwb = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_obs_intervals(nwb, 0) # column does not exist: returns None
        
    """
    with contextlib.suppress(KeyError):
        return get_indexed_table_column(nwb, table_path="units", column="obs_intervals", idx=unit_idx)
    return None

def get_indexed_table_column(nwb: LazyFile, table_path: str, column: str, idx: int | Iterable[int]) -> tuple[npt.NDArray, ...]:
    table = nwb[table_path]
    if column not in table:
        raise KeyError(f"Column {column!r} not found in table {table_path!r}")
    index_column_name = f"{column}_index"
    if isinstance(idx, int):
        idx = (idx, )
    else:
        idx = tuple(idx)
    logger.debug(f"getting {table_path}.{column} for {len(idx)} indices")
    t0 = time.time()
    index_array = table.get(index_column_name)[:]
    data_column = table.get(column)
    if len(idx) > 1000:
        t1 = time.time()
        data_array = data_column[:]
        logger.warning(f"read {table_path}.{column} from disk (len={len(data_column)}) for {len(idx)} indices in {time.time() - t1:.1f} s")
    else:
        data_array = data_column
    def _get_sub_array(i):
        if i == 0:
            start_idx = 0
        else:
            start_idx = index_array[i - 1].item()
        end_idx = index_array[i].item()
        assert start_idx < end_idx, f"{start_idx=} >= {end_idx=}"
        return data_array[start_idx:end_idx]
    sub_arrays = []
    with_threadpool = False # data_array is data_column # if we already read in the data we don't need a threadpool (doesn't make much difference anyway)
    if with_threadpool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [pool.submit(_get_sub_array, i) for i in idx]
        sub_arrays = [future.result() for future in futures]
    else:
        sub_arrays = [_get_sub_array(i) for i in idx]
    logger.warning(f"got {table_path}.{column} for {len(idx)} entries in {time.time() - t0:.1f} s total")
    return tuple(sub_arrays)

if __name__ == "__main__":
    from npc_io import testmod

    testmod()
