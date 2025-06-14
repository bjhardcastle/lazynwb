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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    # lazynwb.convert_nwb_tables(
    #     (OVERRIDE_DIR.parent / "nwb_files").iterdir(),
    #     output_dir=OVERRIDE_DIR,
    #     output_format="parquet",
    #     min_file_count=1,
    #     disable_progress=False,
    #     ignore_errors=False,
    # )
