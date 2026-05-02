import polars as pl
import pytest

import lazynwb
import lazynwb.table_metadata
import lazynwb.tables


@pytest.mark.parametrize(
    ("nwb_fixture_name", "expected_backend"),
    [
        ("local_hdf5_path", "hdf5"),
        ("local_zarr_path", "zarr"),
    ],
)
@pytest.mark.parametrize(
    "table_path",
    [
        "/units",
        "/intervals/trials",
        "/general",
        "processing/behavior/running_speed_with_rate",
    ],
)
def test_raw_table_column_metadata_records_backend_facts(
    nwb_fixture_name, expected_backend, table_path, request
):
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    columns = lazynwb.get_table_column_metadata(nwb_path, table_path)

    assert columns
    assert {column.backend for column in columns} == {expected_backend}
    assert {column.table_path for column in columns} == {
        lazynwb.normalize_internal_file_path(table_path)
    }
    assert all(isinstance(column.attrs, dict) for column in columns)
    assert all(
        column.shape is None or isinstance(column.shape, tuple) for column in columns
    )
    assert any(column.is_dataset for column in columns)


@pytest.mark.parametrize("nwb_fixture_name", ["local_hdf5_path", "local_zarr_path"])
def test_raw_units_metadata_preserves_index_and_array_columns(
    nwb_fixture_name, request
):
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    columns = lazynwb.get_table_column_metadata(nwb_path, "/units")
    column_by_name = {column.name: column for column in columns}

    assert "spike_times" in column_by_name
    assert "spike_times_index" in column_by_name
    assert column_by_name["spike_times"].is_nominally_indexed
    assert column_by_name["spike_times_index"].is_index_column
    assert column_by_name["spike_times"].index_column_name == "spike_times_index"
    assert column_by_name["waveform_mean"].is_multidimensional


@pytest.mark.parametrize("nwb_fixture_name", ["local_hdf5_path", "local_zarr_path"])
def test_raw_metadata_marks_general_as_metadata_table(nwb_fixture_name, request):
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    columns = lazynwb.get_table_column_metadata(nwb_path, "/general")
    column_by_name = {column.name: column for column in columns}

    assert all(column.is_metadata_table for column in columns)
    assert "session_start_time" in column_by_name
    assert lazynwb.table_metadata.get_table_length_from_metadata(columns) == 1


@pytest.mark.parametrize("nwb_fixture_name", ["local_hdf5_path", "local_zarr_path"])
def test_metadata_table_materializes_one_row_from_metadata(nwb_fixture_name, request):
    nwb_path = request.getfixturevalue(nwb_fixture_name)

    df = lazynwb.get_df(nwb_path, "/general", as_polars=True)

    assert df.height == 1
    assert "session_start_time" in df.columns


@pytest.mark.parametrize("nwb_fixture_name", ["local_hdf5_path", "local_zarr_path"])
def test_raw_metadata_marks_timeseries_with_rate(nwb_fixture_name, request):
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    columns = lazynwb.get_table_column_metadata(
        nwb_path, "processing/behavior/running_speed_with_rate"
    )
    column_by_name = {column.name: column for column in columns}

    assert "timestamps" not in column_by_name
    assert {"data", "starting_time"}.issubset(column_by_name)
    assert all(column.is_timeseries for column in columns)
    assert all(column.is_timeseries_with_rate for column in columns)
    assert "rate" in column_by_name["starting_time"].attrs
    assert (
        lazynwb.table_metadata.get_table_length_from_metadata(columns)
        == column_by_name["data"].shape[0]
    )


def test_get_table_schema_from_raw_metadata(local_hdf5_path):
    columns = lazynwb.get_table_column_metadata(local_hdf5_path, "/units")
    schema = lazynwb.tables.get_table_schema_from_metadata(columns)

    assert "spike_times_index" not in schema
    assert schema["spike_times"] == pl.List(pl.Float64)
    assert schema["obs_intervals"] == pl.List(pl.Array(pl.Float64, shape=(2,)))
    assert isinstance(schema["waveform_mean"], pl.Array)


def test_get_timeseries_schema_from_raw_metadata(local_hdf5_path):
    columns = lazynwb.get_table_column_metadata(
        local_hdf5_path, "processing/behavior/running_speed_with_rate"
    )
    schema = lazynwb.tables.get_table_schema_from_metadata(columns)

    assert "starting_time" not in schema
    assert schema["timestamps"] == pl.Float64
    assert schema["data"] == pl.Float64


def test_timeseries_timestamp_projection_materializes_from_metadata(local_hdf5_path):
    df = lazynwb.get_df(
        local_hdf5_path,
        "processing/behavior/running_speed_with_rate",
        include_column_names="timestamps",
        as_polars=True,
    )

    assert "timestamps" in df.columns
    assert "data" not in df.columns
    assert isinstance(df.schema["timestamps"], pl.Float64)


def test_timeseries_timestamp_predicate_pushdown_uses_metadata(local_hdf5_path):
    lf = lazynwb.scan_nwb(
        local_hdf5_path,
        "processing/behavior/running_speed_with_rate",
        disable_progress=True,
    )

    projected = lf.select("timestamps").head(5).collect()
    assert projected.columns == ["timestamps"]
    assert len(projected) == 5

    filtered = (
        lf.filter(pl.col("timestamps") > 2.1).select("timestamps").head(3).collect()
    )
    assert filtered.columns == ["timestamps"]
    assert (filtered["timestamps"] > 2.1).all()


def test_scan_nwb_n_rows_uses_metadata_lengths(local_hdf5_paths):
    df = (
        lazynwb.scan_nwb(
            local_hdf5_paths,
            "/intervals/trials",
            disable_progress=True,
        )
        .head(7)
        .collect()
    )

    assert df.height == 7
