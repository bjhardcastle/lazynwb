from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import time
from typing import Any, Generator, Iterable

import h5py
import polars as pl
from polars.type_aliases import PolarsDataType, PythonDataType
import zarr

from lazynwb.base import LazyNWB

logger = logging.getLogger(__name__)

def get_units(nwb: LazyNWB) -> pl.LazyFrame:
    """
    Get the units table as a polars LazyFrame. Excludes the spike_times and obs_intervals columns.

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> lf = get_units(nwb) 
        >>> lf                          # doctest: +SKIP
        <LazyFrame [41 cols, {"amplitude_cutoff": Float64 … "waveform_sd": List(Float64)}] at 0x220B71F4750>        
        
        >>> nwb = LazyNWB('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> lf = get_units(nwb) 
        >>> lf                          # doctest: +SKIP
        <LazyFrame [38 cols, {"amplitude_cutoff": Float64 … "waveform_sd": List(Float64)}] at 0x7FC93DB97490>
        
    """
    data = _get_units_column_data(nwb, use_thread_pool=isinstance(nwb.units, zarr.Group)) 
    data = {k: data[k] for k in _get_filtered_units_column_names(data.keys())}
    generator_data = {k: _get_data_generator(v) for k, v in data.items()}
    _overrides = {k: _get_polars_schema_override(data[k]) for k in data}
    schema_overrides = {k: v for k, v in _overrides.items() if v is not None}
    t0 = time.time()
    lf = pl.LazyFrame(data=generator_data, infer_schema_length=10, schema_overrides=schema_overrides)
    logger.warning(f"initialized units LazyFrame from {nwb._nwb} in {time.time() - t0:.2f} s")
    return lf

def _get_filtered_units_column_names(names: Iterable[str]) -> Generator[str, None, None]:
    for name in names:
        if name not in ("spike_times", "obs_intervals") and not name.endswith("_index"):
            yield name
            
def _get_units_column_data(nwb: LazyNWB, use_thread_pool: bool = False) -> dict[str, zarr.Array | h5py.Dataset]:
    """Get the units table as a dict of zarr.Array or h5py.Dataset objects. 
    
    Optionally use a thread pool to speed up retrieval of the columns - faster for zarr files.
    """
    data: dict[str, zarr.Array | h5py.Dataset] = {}
    t0 = time.time()
    if use_thread_pool:
        future_to_column = {}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future_to_column = {pool.submit(nwb.units.get, column_name): column_name for column_name in nwb.units}
        for future in concurrent.futures.as_completed(future_to_column):
            column_name = future_to_column[future]
            data[column_name] = future.result()
    else:
        data = {column_name: nwb.units.get(column_name) for column_name in nwb.units}
    logger.warning(f"retrieved units columns from {nwb._nwb} in {time.time() - t0:.2f} s ({use_thread_pool=})")
    return data

def _get_polars_schema_override(data: zarr.Array | h5py.Dataset) -> pl.DataType | None:
    if data.dtype.kind == 'O' or 'name' in data.name.split('/')[-1]:
        return pl.String()
    return None
    
def _get_data_generator(data: h5py.Dataset | zarr.Array) -> Generator[PolarsDataType | PythonDataType, None, None]:
    yield from iter(data)

    
def get_spike_times(nwb: LazyNWB, unit_idx: int) -> Any:
    """
    Get the spike times for a single unit, from its index in the units table.

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_spike_times(nwb, 0)
        array([2925.85956667, 2931.19676667, 2944.81003333, ...,
           6696.99163333, 6700.57663333, 6700.89296667])
    """
    return get_indexed_units_column(nwb, "spike_times", unit_idx)

def get_obs_intervals(nwb: LazyNWB, unit_idx: int) -> tuple[tuple[float, float], ...] | None:
    """Get the observation intervals for a single unit, from its index in the
    units table.
    
    Examples:
    
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_obs_intervals(nwb, 0) # column does not exist: returns None
        
    """
    with contextlib.suppress(KeyError):
        return get_indexed_units_column(nwb, "obs_intervals", unit_idx)
    return None

def get_indexed_units_column(nwb: LazyNWB, column: str, unit_idx: int) -> Any:
    if column not in nwb.units:
        raise KeyError(f"Column {column!r} not found in units table")
    index_column = f"{column}_index"
    if unit_idx == 0:
        start_idx = 0
    else:
        start_idx = nwb.units.get(index_column)[unit_idx - 1].item()
    end_idx = nwb.units.get(index_column)[unit_idx].item()
    assert start_idx < end_idx, f"{start_idx=} >= {end_idx=}"
    return nwb.units.get(column)[start_idx:end_idx]

if __name__ == "__main__":
    from npc_io import testmod

    testmod()
