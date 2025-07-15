from __future__ import annotations

import contextlib
import pathlib
import tempfile

import polars as pl
import pytest

import lazynwb

CLEANUP_FILES = True
OVERRIDE_DIR: None | pathlib.Path = (
    None
    if CLEANUP_FILES
    else pathlib.Path(__file__).parent / "files" / "converted_tables"
)

@pytest.mark.xfail(reason="Fails with zarr v2: zarr v3 not supported by hdmf-zarr")
@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_zarr_paths", # will not pass until hdmf-zarr supports zarr v3
        "local_hdf5_paths",
    ],
)
def test_convert_nwb_tables(nwb_fixture_name, request):
    """Test basic NWB to Parquet conversion using convert_nwb_tables."""
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)
    if CLEANUP_FILES:
        context = tempfile.TemporaryDirectory()
    else:
        context = contextlib.nullcontext()

    with context as temp_dir:
        output_dir = pathlib.Path(temp_dir) if CLEANUP_FILES else OVERRIDE_DIR

        # Convert with minimal settings
        output_paths = lazynwb.convert_nwb_tables(
            nwb_sources=nwb_path_or_paths[:2],  # Use first 2 files
            output_dir=output_dir,
            output_format="parquet",
            min_file_count=1,
            disable_progress=True,
        )

        # Check that we got some outputs
        assert len(output_paths) > 0, "Should have converted at least one table"

        # Check that output files exist
        for _table_path, parquet_path in output_paths.items():
            assert parquet_path.exists(), f"Parquet file should exist: {parquet_path}"
            assert (
                parquet_path.suffix == ".parquet"
            ), f"Should have .parquet extension: {parquet_path}"

            # Verify we can read the Parquet file
            df = pl.read_parquet(parquet_path)
            assert (
                not df.is_empty()
            ), f"Parquet file should not be empty: {parquet_path}"

            # Check that internal columns are present
            assert (
                lazynwb.NWB_PATH_COLUMN_NAME in df.columns
            ), "Should contain NWB path column"
            assert (
                lazynwb.TABLE_PATH_COLUMN_NAME in df.columns
            ), "Should contain table path column"

            # Check that there are two sessions
            assert (
                df[lazynwb.NWB_PATH_COLUMN_NAME].n_unique() == 2
            ), "Should contain two unique NWB paths"

            if "units" in parquet_path.name:
                # Check that units table has expected columns
                assert (
                    "spike_times" in df.columns
                ), "Units table should contain spike_times column"


def test_table_path_to_output_path():
    """Test conversion of table paths to output filenames."""
    from lazynwb.conversion import _table_path_to_output_path

    output_dir = pathlib.Path("/tmp")

    # Test standard table paths
    result = _table_path_to_output_path(output_dir, "/intervals/trials", ".parquet")
    assert result == output_dir / "intervals_trials.parquet"

    result = _table_path_to_output_path(output_dir, "/units", ".csv")
    assert result == output_dir / "units.csv"

    result = _table_path_to_output_path(
        output_dir, "/processing/behavior/running_speed", ".json"
    )
    assert result == output_dir / "processing_behavior_running_speed.json"

    # Test delta format (no extension)
    result = _table_path_to_output_path(output_dir, "/units", "")
    assert result == output_dir / "units"

    # Test with full_path=False
    result = _table_path_to_output_path(
        output_dir, "/intervals/trials", ".parquet", full_path=False
    )
    assert result == output_dir / "trials.parquet"
    result = _table_path_to_output_path(
        output_dir, "/processing/behavior/running_speed", ".json", full_path=False
    )
    assert result == output_dir / "running_speed.json"
    
@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_paths",
    ],
)
def test_sql_context(nwb_fixture_name, request):
    """Test polars SQLContext with NWB files."""
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)

    sql_context = lazynwb.get_sql_context(
        nwb_sources=nwb_path_or_paths,
        full_path=True,
        min_file_count=1,
        exclude_array_columns=False,
        ignore_errors=True,
        disable_progress=False,
    )

    # Check that the SQL context is not None
    assert sql_context is not None

    # Check that the expected tables are registered
    expected_tables = [
        "intervals/trials",
        "intervals/epochs",
        "units",
        "processing/behavior/running_speed_with_timestamps", # rate not currently supported 
    ]
    tables = sql_context.tables()
    for table in expected_tables:
        assert table in tables, f"Table not found: {table}"
    # Check that we can query a table
    trials_df = sql_context.execute("SELECT * FROM `intervals/trials`", eager=True)
    assert not trials_df.is_empty(), "Trials table should not be empty"
    
    # Check that we can query a timeseries converted to table
    ts_df = sql_context.execute("SELECT * FROM `processing/behavior/running_speed_with_timestamps` LIMIT 10", eager=True)
    assert not ts_df.is_empty(), "Timestamps table should not be empty"
    
    # Check that table filtering works
    sql_context = lazynwb.get_sql_context(
            nwb_sources=nwb_path_or_paths,
            full_path=False,
            min_file_count=1,
            exclude_array_columns=False,
            ignore_errors=True,
            disable_progress=False,
            table_names=['units'],
        )
    assert sql_context.tables() == ["units"], "Only units table should be registered when using table_names filter"

    # Check that inferred schema length works
    sql_context = lazynwb.get_sql_context(
        nwb_sources=nwb_path_or_paths,
        full_path=False,
        min_file_count=1,
        exclude_array_columns=False,
        ignore_errors=True,
        disable_progress=False,
        table_names=['units'],
        infer_schema_length=1,
    )
    assert (actual := sql_context.execute("SELECT COUNT(DISTINCT _nwb_path) FROM units", eager=True).height) == 1, f"infer_schema_length should limit to one unique path, got {actual}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", '--pdb'])
