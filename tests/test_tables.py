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
        "local_zarr_path",
        "local_zarr_paths",
    ],
)
def test_zarr_object_arrays(nwb_fixture_name, request):
    """Regression test: zarr v2 object arrays can be read without ValueError.

    zarr.Array.astype(str) creates a view with dtype='<U0' (zero-length unicode);
    when zarr decodes a chunk it calls chunk.view('<U0') which raises
    ValueError: When changing to a smaller dtype, its size must be a divisor of
    the size of original dtype.

    The fix is to use the raw zarr Array directly instead of wrapping it with
    astype(str) for indexed string columns.
    """
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)

    # Indexed string column: epochs 'tags' (ragged array of strings).
    # This exercises the code path that was previously broken for zarr v2.
    df = lazynwb.get_df(
        nwb_path_or_paths,
        "/intervals/epochs",
        exclude_array_columns=False,
    )
    assert "tags" in df.columns, "epochs table should have a 'tags' column"
    assert all(
        isinstance(row, list) for row in df["tags"].tolist()
    ), "each epoch's tags should be a list"
    assert all(
        isinstance(tag, str)
        for row in df["tags"].tolist()
        for tag in row
    ), "each tag should be a string"

    # Non-indexed string column: units 'structure'.
    df = lazynwb.get_df(
        nwb_path_or_paths,
        "/units",
        exclude_array_columns=True,
    )
    assert "structure" in df.columns, "units table should have a 'structure' column"
    assert all(
        isinstance(v, str) for v in df["structure"].tolist()
    ), "each structure value should be a string"


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
