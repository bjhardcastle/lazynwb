import concurrent.futures
import contextlib
from typing import Any
from lazynwb.base import LazyNWB
import polars as pl

def get_spike_times(nwb: LazyNWB, unit_idx: int) -> Any:
    """
    Get the spike times for a single unit, from its index in the units table.

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_spike_times(nwb, 0)
        array([2925.85956667, 2931.19676667, 2944.81003333, ...,
           6696.99163333, 6700.57663333, 6700.89296667])
    """
    return _get_indexed_units_column(nwb, "spike_times", unit_idx)

def get_obs_intervals(nwb: LazyNWB, unit_idx: int) -> tuple[tuple[float, float], ...] | None:
    """Get the observation intervals for a single unit, from its index in the
    units table.
    
    Examples:
    
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> get_obs_intervals(nwb, 0) # column does not exist: returns None
        
    """
    with contextlib.suppress(KeyError):
        return _get_indexed_units_column(nwb, "obs_intervals", unit_idx)
    return None

def _get_indexed_units_column(nwb: LazyNWB, column: str, unit_idx: int) -> Any:
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

def get_units(nwb: LazyNWB) -> pl.LazyFrame:
    """
    Get the units table as a polars LazyFrame. Excludes the spike_times and obs_intervals columns.

    Examples:
        # >>> nwb = LazyNWB('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> print(get_units(nwb))
        <LazyFrame [41 cols, {"amplitude_cutoff": Float64 … "waveform_sd": List(Float64)}] at 0x220B71F4750>
    """
    data = {}
    future_to_column = {}
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future_to_column = {pool.submit(nwb.units.get, column_name): column_name for column_name in nwb.units if column_name not in ("spike_times", "obs_intervals")}
    for future in concurrent.futures.as_completed(future_to_column):
        column_name = future_to_column[future]
        data[column_name] = future.result()
    return pl.LazyFrame(data=data, schema_overrides=_infer_polars_schema_overrides(data))

def _infer_polars_schema_overrides(data: dict[str, Any]) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for column_name, dtype in data.items():
        if dtype.kind == "O":
            schema[column_name] = pl.String()
    return schema

if __name__ == "__main__":
    from npc_io import testmod

    testmod()
