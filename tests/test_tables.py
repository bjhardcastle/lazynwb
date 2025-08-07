import logging

import pytest

import pandas as pd
import polars as pl
import lazynwb
import pynwb


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_hdf5_paths",
        "local_zarr_path",
        "local_zarr_paths",
    ],
)
def test_sources(nwb_fixture_name, request):
    """Test get_df with various NWB file/store inputs."""
    # Resolve the fixture name to its value (the path or list of paths)
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)

    df = lazynwb.get_df(nwb_path_or_paths, "/intervals/trials", as_polars=True)
    assert not df.is_empty(), f"DataFrame is empty for {nwb_fixture_name}"


def test_internal_column_names(local_hdf5_path): 
    df = lazynwb.get_df(
        local_hdf5_path, "/intervals/trials"
    )
    for col in lazynwb.INTERNAL_COLUMN_NAMES:
        assert col in df.columns, f"Internal column {col!r} not found"

@pytest.mark.parametrize("table_name", ["trials", "units"])
def test_contents(local_hdf5_path, table_name):
    """Validate contents of dataframes against those obtained via pynwb"""
    exact_table_paths = {
        'trials': "/intervals/trials",
        'units': '/units',
    }
    df = (
        lazynwb.get_df(
            local_hdf5_path,
            exact_table_paths[table_name],
            exact_path=True,
            exclude_array_columns=False,
        )
        # we add internal columns for identifying source of rows when concatenating across files: 
        # drop them for comparison
        .drop(columns=lazynwb.INTERNAL_COLUMN_NAMES)
        .set_index('id')
    )
    nwb = pynwb.read_nwb(local_hdf5_path)
    reference_df = getattr(nwb, table_name).to_dataframe()
    pd.testing.assert_frame_equal(
        df,
        reference_df,
        check_dtype=True,
        check_exact=False,
        check_like=True,
    )

@pytest.mark.parametrize("table_shortcut", ['trials', 'epochs', 'session'])
def test_shortcuts(local_hdf5_path, table_shortcut: str):
    """Test that table shortcuts work as expected."""
    expected_path = lazynwb.TABLE_SHORTCUTS[table_shortcut]
    df = lazynwb.get_df(local_hdf5_path, table_shortcut, as_polars=True)
    assert not df.is_empty(), f"DataFrame fetched with {table_shortcut=} should not be empty"
    assert df['_table_path'].first() == expected_path, f"Table path should be full path, not {table_shortcut=}"

def test_general(local_hdf5_path):
    df = lazynwb.get_df(local_hdf5_path, "/general", as_polars=True)
    assert not df.is_empty(), f"'general' table should provide metadata from /general and top-level of file"
    assert 'session_start_time' in df.columns, f"'general' table should provide metadata from /general and top-level of file"

@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_hdf5_paths",
    ],
)
def test_timeseries_with_rate(nwb_fixture_name, request):
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)
    # without timestamps, the default TimeSeries object has two keys: 'data' and
    # 'starting_time' which is another Group.
    # get_df() interprets them as 'data': List[float], 'starting_time': float
    # it needs to be aware of this possibility and generate a timestamps column
    df = lazynwb.get_df(nwb_path_or_paths, "processing/behavior/running_speed_with_rate", as_polars=True)
    assert 'timestamps' in df.columns, f"'trials' table should provide a 'timestamps' column"
    assert isinstance(df.schema['timestamps'], pl.Float64), f"'timestamps' column should be a float type, not {df.schema['timestamps']}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
