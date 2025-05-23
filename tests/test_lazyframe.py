import logging

import polars as pl
import pytest

import h5py
import lazynwb
import lazynwb.tables

def test_polars_dtype_inference(local_hdf5_path):
    schema = lazynwb.tables._get_table_schema(
        lazynwb.FileAccessor(local_hdf5_path),
        table_path='units',
    )
    assert schema['obs_intervals'] == pl.List(pl.Array(pl.Float64, shape=(2,))), f"Expected polars dtype for obs_intervals to be List[List[Float64]]: {schema['obs_intervals']=}"

@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_hdf5_paths",
        "local_zarr_path",
        "local_zarr_paths",
    ],
)
def test_scan_nwb_sources(nwb_fixture_name, request):
    """Test scan_nwb with various NWB file/store inputs."""
    # Resolve the fixture name to its value (the path or list of paths)
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)

    # Test with trials table
    lf = lazynwb.scan_nwb(
        source=nwb_path_or_paths, table_path="/intervals/trials", disable_progress=True
    )

    # Check that we got a LazyFrame
    assert isinstance(lf, pl.LazyFrame), "scan_nwb should return a LazyFrame"

    # Execute the lazy frame and check we have data
    df = lf.collect()
    assert not df.is_empty(), f"DataFrame is empty for {nwb_fixture_name}"

    # Test that all expected internal columns are present
    for col in lazynwb.INTERNAL_COLUMN_NAMES:
        assert col in df.columns, f"Internal column {col!r} not found"


def test_scan_nwb_exclude_array_columns(local_hdf5_path):
    """Test scan_nwb with exclude_array_columns option."""
    # First get all columns including arrays
    lf_with_arrays = lazynwb.scan_nwb(
        source=local_hdf5_path,
        table_path="/units",
        exclude_array_columns=False,
        disable_progress=True,
    )
    df_with_arrays = lf_with_arrays.collect()

    # Now exclude array columns
    lf_no_arrays = lazynwb.scan_nwb(
        source=local_hdf5_path,
        table_path="/units",
        exclude_array_columns=True,
        disable_progress=True,
    )
    df_no_arrays = lf_no_arrays.collect()

    # Check that the version with arrays has more columns
    assert (
        df_with_arrays.width > df_no_arrays.width
    ), "DataFrame with arrays should have more columns"

    # Check for array columns like spike_times, waveform_mean, obs_intervals
    array_columns = {"spike_times", "waveform_mean", "obs_intervals"}
    for col in array_columns:
        assert (
            col in df_with_arrays.columns
        ), f"Array column {col!r} should be present when exclude_array_columns=False"
        assert (
            col not in df_no_arrays.columns
        ), f"Array column {col!r} should be excluded when exclude_array_columns=True"


def test_scan_nwb_schema_override(local_hdf5_path):
    """Test scan_nwb with schema overrides."""
    # Define schema overrides for specific columns
    schema_overrides = {"condition": pl.Categorical()}

    df = lazynwb.scan_nwb(
        source=local_hdf5_path,
        table_path="/intervals/trials",
        schema_overrides=schema_overrides,
        disable_progress=True,
    ).collect()
    # Check that the schema override was applied
    assert (
        df.schema["condition"] == pl.Categorical
    ), f"Schema override for 'condition' column was not applied: {df.schema=}"
    
    # verify this works with predicate pushdown
    df = lazynwb.scan_nwb(
        source=local_hdf5_path,
        table_path="/intervals/trials",
        schema_overrides=schema_overrides,
        disable_progress=True,
    ).filter(pl.col("start_time") > 2.0).collect()
    assert (
        df.schema["condition"] == pl.Categorical
    ), f"Schema override for 'condition' column was not applied when combined with filtering: {df.schema=}"

    # verify this works with projection 
    df = lazynwb.scan_nwb(
        source=local_hdf5_path,
        table_path="/intervals/trials",
        schema_overrides=schema_overrides,
        disable_progress=True,
    ).select("start_time", "stop_time", "condition").collect()
    assert (
        df.schema["condition"] == pl.Categorical
    ), f"Schema override for 'condition' column was not applied when combined with projection: {df.schema=}"
    
    
def test_scan_nwb_predicate_pushdown(local_hdf5_path):
    """Test that scan_nwb correctly handles predicate pushdown optimizations."""
    # Create a query with a predicate
    lf = lazynwb.scan_nwb(
        source=local_hdf5_path, table_path="/intervals/trials", disable_progress=True
    )

    # Apply a predicate filter
    expr = pl.col("start_time") > 2.0
    filtered_lf = lf.filter(expr)

    # Execute and check results
    filtered_df = filtered_lf.collect()

    # Verify the filter was applied correctly
    assert (
        filtered_df["start_time"] > 2.0
    ).all(), "Predicate pushdown filter was not applied correctly"
    assert len(filtered_df) == len(lf.collect().filter(expr)), "Filtered DataFrame length does not match length when collecting and filtering separately"


def test_scan_nwb_raises_on_missing(local_hdf5_path):
    """Test that scan_nwb raises an error when the table is not found and raise_on_missing=True."""
    # Try with a non-existent table path
    with pytest.raises(KeyError):
        lf = lazynwb.scan_nwb(
            source=local_hdf5_path,
            table_path="/non_existent_table",
            raise_on_missing=True,
            disable_progress=True,
        )
        # Force execution to trigger the error
        lf.collect()


def test_scan_nwb_projection_pushdown(local_hdf5_path):
    """Test that scan_nwb correctly handles column projection optimizations."""
    # Create a query with column selection
    lf = lazynwb.scan_nwb(
        source=local_hdf5_path, table_path="/intervals/trials", disable_progress=True
    )

    # Select only specific columns
    selected_columns = ["start_time", "stop_time", "condition"]
    projected_lf = lf.select(selected_columns)

    # Execute and check results
    projected_df = projected_lf.collect()

    # Check that only the requested columns are present
    expected_columns = set(selected_columns)
    assert (
        set(projected_df.columns) == expected_columns
    ), f"LazyFrame Projection did not work correctly: {projected_df.columns=}, {expected_columns=}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
