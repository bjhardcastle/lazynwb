from __future__ import annotations

import asyncio
import logging
import os
import pathlib

import h5py
import numpy as np
import polars as pl
import pytest

import lazynwb
import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.models as catalog_models
import lazynwb._catalog.polars as catalog_polars
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb.conversion as conversion
import lazynwb.table_metadata
import lazynwb.tables
import lazynwb.utils


def test_hdf5_backend_reader_parses_scalar_table_from_byte_buffer(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    snapshot = asyncio.run(reader.read_table_schema_snapshot("intervals/trials"))
    schema = catalog_polars._snapshot_to_polars_schema(snapshot)
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert snapshot.backend == "hdf5"
    assert snapshot.table_path == "intervals/trials"
    assert {"start_time", "stop_time", "condition"}.issubset(schema)
    assert schema["start_time"] == pl.Float64
    assert columns_by_name["start_time"].dataset.hdf5_data_offset is not None
    assert columns_by_name["start_time"].dataset.hdf5_storage_size is not None
    assert "direct_contiguous" in columns_by_name["start_time"].dataset.read_capabilities


def test_hdf5_backend_reader_preserves_units_array_catalog_facts(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert columns_by_name["spike_times"].dataset.path == "units/spike_times"
    assert columns_by_name["spike_times"].dataset.shape is not None
    assert columns_by_name["spike_times"].dataset.hdf5_data_offset is not None
    assert columns_by_name["spike_times"].dataset.hdf5_storage_size is not None
    assert "shape" in columns_by_name["spike_times"].dataset.read_capabilities
    assert columns_by_name["obs_intervals"].dataset.shape is not None
    assert columns_by_name["obs_intervals"].dataset.shape[1:] == (2,)
    assert columns_by_name["waveform_mean"].dataset.shape is not None
    assert columns_by_name["waveform_mean"].dataset.shape[1:] == (
        25,
        384,
    )
    assert columns_by_name["waveform_mean"].dataset.storage_layout is not None


def test_public_exclude_array_columns_filters_after_cached_hdf5_snapshot(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "catalog.sqlite"
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(cache_path))
    source_url = local_hdf5_path.as_uri()

    schema = lazynwb.tables.get_table_schema(
        source_url,
        "/units",
        exclude_array_columns=True,
        exclude_internal_columns=True,
    )
    warm_reader = hdf5_reader._default_hdf5_backend_reader(source_url)
    warm_snapshot = asyncio.run(warm_reader.read_table_schema_snapshot("units"))
    warm_column_names = {column.name for column in warm_snapshot.columns}

    assert "spike_times" not in schema
    assert "waveform_mean" not in schema
    assert {"spike_times", "waveform_mean", "obs_intervals"}.issubset(warm_column_names)
    assert warm_reader._range_reader.request_count == 0


def test_hdf5_backend_reader_reuses_sqlite_snapshot_without_range_reads(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    cache_path = tmp_path / "catalog.sqlite"
    cold_reader = _buffer_hdf5_reader(local_hdf5_path, cache_path)

    cold_snapshot = asyncio.run(
        cold_reader.read_table_schema_snapshot("intervals/trials")
    )
    cold_requests = cold_reader._range_reader.request_count

    warm_reader = _buffer_hdf5_reader(local_hdf5_path, cache_path)
    warm_snapshot = asyncio.run(
        warm_reader.read_table_schema_snapshot("intervals/trials")
    )

    assert cold_requests > 0
    assert warm_reader._range_reader.request_count == 0
    assert warm_snapshot == cold_snapshot


def test_hdf5_backend_reader_does_not_call_h5py_file_for_schema(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    def _fail_h5py_file(*args: object, **kwargs: object) -> None:
        raise AssertionError("fast HDF5 schema path must not call h5py.File")

    monkeypatch.setattr(h5py, "File", _fail_h5py_file)

    snapshot = asyncio.run(reader.read_table_schema_snapshot("intervals/trials"))
    schema = catalog_polars._snapshot_to_polars_schema(snapshot)

    assert {"start_time", "stop_time", "condition"}.issubset(schema)


def test_hdf5_backend_reader_single_table_schema_does_not_warm_followup_cache(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    parsed_cache_path = tmp_path / "parsed-cache.sqlite"
    trials_reader = _buffer_hdf5_reader(local_hdf5_path, parsed_cache_path)
    asyncio.run(trials_reader.read_table_schema_snapshot("intervals/trials"))
    source_identity = asyncio.run(trials_reader.get_source_identity())

    cache = cache_sqlite._SQLiteSnapshotCache(parsed_cache_path)
    followup_cache_lookup = asyncio.run(
        cache.get_table_schema_snapshot(source_identity, "units")
    )
    followup_units_reader = _buffer_hdf5_reader(local_hdf5_path, parsed_cache_path)
    followup_snapshot = asyncio.run(
        followup_units_reader.read_table_schema_snapshot("units")
    )
    followup_requests = followup_units_reader._range_reader.request_count

    assert not followup_cache_lookup.hit
    assert followup_snapshot.table_path == "units"
    assert followup_requests > 0


def test_hdf5_backend_reader_logs_single_and_multi_table_scan_modes(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="lazynwb._hdf5.reader")
    single_reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "single.sqlite")
    asyncio.run(single_reader.read_table_schema_snapshot("intervals/trials"))

    multi_reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "multi.sqlite")
    asyncio.run(
        multi_reader._read_table_schema_snapshots(("intervals/trials", "units"))
    )

    assert "single-table HDF5 schema scan" in caplog.text
    assert "explicit multi-table HDF5 schema scan" in caplog.text


def test_hdf5_backend_reader_builds_path_summary(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    summary = asyncio.run(reader.read_path_summary())
    paths = {entry.path for entry in summary}
    entries_by_path = {entry.path: entry for entry in summary}

    assert "/intervals/trials" in paths
    assert "/units" in paths
    assert "/general" in paths
    assert "/general/subject" in paths
    assert "/processing/behavior/running_speed_with_rate" in paths
    assert entries_by_path["/units"].is_group
    assert entries_by_path["/units"].attrs["colnames"] is True


def test_catalog_path_summary_filters_metadata_timeseries_and_specifications(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.file_io")
    source_url = local_hdf5_path.as_uri()

    summary = lazynwb.file_io._get_catalog_path_summary_if_available(
        source_url,
        include_arrays=True,
        include_table_columns=False,
        include_metadata=True,
        include_specifications=False,
        parents=True,
    )

    assert summary is not None
    assert "/intervals/trials" in summary
    assert "/units" in summary
    assert "/general" in summary
    assert "/general/subject" in summary
    assert "/processing/behavior/running_speed_with_rate" in summary
    assert "/processing/behavior/running_speed_with_rate/data" in summary
    assert not any(path.startswith("/specifications") for path in summary)
    assert "catalog summary" in caplog.text


def test_common_path_discovery_uses_catalog_summary_without_accessor(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = local_hdf5_path.as_uri()

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("common path discovery should use the catalog summary")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    paths = conversion._find_common_paths(
        (source_url,),
        min_file_count=1,
        disable_progress=True,
        include_timeseries=True,
        include_metadata=True,
    )

    assert "/intervals/trials" in paths
    assert "/units" in paths
    assert "/general" in paths
    assert "/general/subject" in paths
    assert "/processing/behavior/running_speed_with_rate" in paths


def test_sql_context_uses_explicit_hdf5_multi_table_schema_scan(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.conversion")
    source_url = local_hdf5_path.as_uri()
    calls: list[tuple[str, ...]] = []
    original_scan = hdf5_reader._HDF5BackendReader._read_table_schema_snapshots

    async def _spy_multi_table_scan(
        self: hdf5_reader._HDF5BackendReader,
        exact_table_paths: tuple[str, ...],
    ) -> dict[str, hdf5_reader._HDF5TableSchemaScanResult]:
        calls.append(exact_table_paths)
        return await original_scan(self, exact_table_paths)

    monkeypatch.setattr(
        hdf5_reader._HDF5BackendReader,
        "_read_table_schema_snapshots",
        _spy_multi_table_scan,
    )

    sql_context = lazynwb.get_sql_context(
        source_url,
        full_path=True,
        min_file_count=1,
        exclude_array_columns=False,
        exclude_timeseries=False,
        ignore_errors=True,
        disable_progress=True,
    )

    assert any(
        {
            "intervals/trials",
            "units",
            "general",
            "processing/behavior/running_speed_with_rate",
        }.issubset(call)
        for call in calls
    )
    for table_name in (
        "intervals/trials",
        "units",
        "session",
        "processing/behavior/running_speed_with_rate",
    ):
        assert table_name in sql_context.tables()
        result = sql_context.execute(
            f"SELECT COUNT(*) FROM `{table_name}`",
            eager=True,
        )
        assert result.item() > 0
    assert "SQL context HDF5 multi-table scan" in caplog.text
    assert "cache_writes=" in caplog.text


def test_hdf5_backend_reader_scans_multiple_tables_in_one_lifecycle(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    shared_reader = _buffer_hdf5_reader(
        local_hdf5_path,
        tmp_path / "shared-catalog.sqlite",
    )

    results = asyncio.run(
        shared_reader._read_table_schema_snapshots(("intervals/trials", "units"))
    )
    shared_requests = shared_reader._range_reader.request_count

    cold_trials_reader = _buffer_hdf5_reader(
        local_hdf5_path,
        tmp_path / "cold-trials.sqlite",
    )
    asyncio.run(cold_trials_reader.read_table_schema_snapshot("intervals/trials"))
    cold_units_reader = _buffer_hdf5_reader(
        local_hdf5_path,
        tmp_path / "cold-units.sqlite",
    )
    asyncio.run(cold_units_reader.read_table_schema_snapshot("units"))
    independent_requests = (
        cold_trials_reader._range_reader.request_count
        + cold_units_reader._range_reader.request_count
    )

    assert set(results) == {"intervals/trials", "units"}
    assert results["intervals/trials"].ok
    assert results["units"].ok
    assert results["intervals/trials"].snapshot is not None
    assert results["units"].snapshot is not None
    assert results["intervals/trials"].request_count > 0
    assert (
        results["units"].request_count < cold_units_reader._range_reader.request_count
    )
    assert shared_requests < independent_requests


def test_hdf5_backend_multi_table_helper_writes_individual_schema_cache_entries(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    cache_path = tmp_path / "shared-catalog.sqlite"
    shared_reader = _buffer_hdf5_reader(local_hdf5_path, cache_path)

    asyncio.run(
        shared_reader._read_table_schema_snapshots(("intervals/trials", "units"))
    )
    warm_units_reader = _buffer_hdf5_reader(local_hdf5_path, cache_path)
    warm_units_snapshot = asyncio.run(
        warm_units_reader.read_table_schema_snapshot("units")
    )

    assert warm_units_snapshot.table_path == "units"
    assert warm_units_reader._range_reader.request_count == 0


def test_hdf5_backend_multi_table_helper_reports_missing_table_per_result(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    results = asyncio.run(
        reader._read_table_schema_snapshots(("intervals/trials", "not_a_table"))
    )

    assert results["intervals/trials"].ok
    assert results["not_a_table"].snapshot is None
    assert isinstance(
        results["not_a_table"].error,
        hdf5_reader._HDF5TableSchemaScanError,
    )
    assert results["not_a_table"].error is not None
    assert results["not_a_table"].error.source_url == local_hdf5_path.as_uri()
    assert results["not_a_table"].error.table_path == "not_a_table"
    assert results["not_a_table"].error.feature == "missing_table"


def test_public_get_table_schema_uses_fast_hdf5_for_obstore_file_url(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LAZYNWB_CATALOG_CACHE_PATH",
        str(tmp_path / "catalog.sqlite"),
    )

    schema = lazynwb.tables.get_table_schema(
        local_hdf5_path.as_uri(),
        "/intervals/trials",
        exclude_internal_columns=True,
    )

    assert {"start_time", "stop_time", "condition"}.issubset(schema)
    assert schema["start_time"] == pl.Float64


def test_public_get_df_uses_fast_hdf5_catalog_for_materialization(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = local_hdf5_path.as_uri()

    def _fail_metadata_traversal(*args: object, **kwargs: object) -> None:
        raise AssertionError("get_df should plan exact HDF5 reads from the catalog")

    monkeypatch.setattr(
        lazynwb.table_metadata,
        "get_table_column_metadata",
        _fail_metadata_traversal,
    )

    df = lazynwb.get_df(
        source_url,
        "/intervals/trials",
        exact_path=True,
        include_column_names=("start_time",),
        as_polars=True,
    )

    assert df.height > 0
    assert {"start_time", lazynwb.NWB_PATH_COLUMN_NAME}.issubset(df.columns)


def test_public_get_df_materializes_supported_hdf5_scalars_without_accessor(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.tables")
    source_url = local_hdf5_path.as_uri()
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("supported scalar HDF5 columns should use range reads")

    monkeypatch.setattr(hdf5_reader, "_default_hdf5_backend_reader", lambda _: reader)
    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    df = lazynwb.get_df(
        source_url,
        "/intervals/trials",
        exact_path=True,
        include_column_names=("start_time", "stop_time"),
        as_polars=True,
    )

    assert df.select("start_time", "stop_time").height > 0
    assert 0 < reader._range_reader.request_count < 1000
    assert reader._range_reader.bytes_fetched < local_hdf5_path.stat().st_size
    assert "direct HDF5 materialization" in caplog.text
    assert "fallback columns=[]" in caplog.text


def test_hdf5_direct_scalar_materialization_matches_h5py_reference(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = local_hdf5_path.as_uri()

    trials_df = lazynwb.get_df(
        source_url,
        "/intervals/trials",
        exact_path=True,
        include_column_names=("start_time", "stop_time"),
        as_polars=True,
    )
    units_df = lazynwb.get_df(
        source_url,
        "/units",
        exact_path=True,
        include_column_names=("id",),
        as_polars=True,
    )

    with h5py.File(local_hdf5_path, "r") as h5_file:
        expected_start = h5_file["intervals/trials/start_time"][:].tolist()
        expected_stop = h5_file["intervals/trials/stop_time"][:].tolist()
        expected_unit_id = h5_file["units/id"][:].tolist()

    assert trials_df["start_time"].to_list() == expected_start
    assert trials_df["stop_time"].to_list() == expected_stop
    assert units_df["id"].to_list() == expected_unit_id


def test_hdf5_direct_scalar_materializes_bool_column_from_ranges(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nwb_path = tmp_path / "bool-table.nwb"
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("table")
        group.create_dataset("flag", data=np.array([True, False, True], dtype=np.bool_))

    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = nwb_path.as_uri()

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("supported bool HDF5 columns should use range reads")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    df = lazynwb.get_df(
        source_url,
        "/table",
        exact_path=True,
        include_column_names=("flag",),
        as_polars=True,
    )

    assert df["flag"].to_list() == [True, False, True]


def test_hdf5_direct_indexed_materialization_matches_h5py_reference(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.tables")
    source_url = local_hdf5_path.as_uri()

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("supported indexed HDF5 columns should use range reads")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    df = lazynwb.get_df(
        source_url,
        "/units",
        exact_path=True,
        include_column_names=("spike_times",),
        as_polars=True,
    )

    with h5py.File(local_hdf5_path, "r") as h5_file:
        expected = _h5py_indexed_column(
            h5_file["units/spike_times"][:],
            h5_file["units/spike_times_index"][:],
        )

    assert df["spike_times"].to_list() == expected
    assert "direct HDF5 indexed materialization" in caplog.text
    assert "direct indexed columns=['spike_times']" in caplog.text


def test_hdf5_direct_indexed_materializes_selected_rows_and_empty_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nwb_path = tmp_path / "ragged-table.nwb"
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("units")
        group.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.3]))
        group.create_dataset(
            "spike_times_index",
            data=np.array([0, 2, 2, 3], dtype=np.uint8),
        )

    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = nwb_path.as_uri()

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("supported indexed HDF5 columns should use range reads")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    df = lazynwb.get_df(
        source_url,
        "/units",
        exact_path=True,
        include_column_names=("spike_times",),
        nwb_path_to_row_indices={source_url: [0, 1, 3]},
        as_polars=True,
    )

    assert df["spike_times"].to_list() == [[], [0.1, 0.2], [0.3]]


def test_scan_nwb_predicate_projection_uses_direct_indexed_ranges(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    nwb_path = tmp_path / "filtered-ragged-table.nwb"
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("units")
        group.create_dataset("id", data=np.arange(4, dtype=np.int64))
        group.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.3, 0.4]))
        group.create_dataset(
            "spike_times_index",
            data=np.array([1, 3, 3, 4], dtype=np.uint8),
        )

    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.tables")
    source_url = nwb_path.as_uri()
    lf = (
        lazynwb.scan_nwb(source_url, "/units", disable_progress=True)
        .filter(pl.col("id") == 1)
        .select("spike_times")
    )

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("filtered indexed projection should use range reads")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    df = lf.collect()

    assert df["spike_times"].to_list() == [[0.2, 0.3]]
    assert "direct HDF5 indexed materialization" in caplog.text
    assert "requested_elements=2" in caplog.text


def test_scan_nwb_uses_fast_hdf5_catalog_for_materialization(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = local_hdf5_path.as_uri()

    def _fail_metadata_traversal(*args: object, **kwargs: object) -> None:
        raise AssertionError("scan_nwb should plan exact HDF5 reads from the catalog")

    monkeypatch.setattr(
        lazynwb.table_metadata,
        "get_table_column_metadata",
        _fail_metadata_traversal,
    )

    df = (
        lazynwb.scan_nwb(source_url, "/intervals/trials", disable_progress=True)
        .select("start_time")
        .head(1)
        .collect()
    )

    assert df.shape == (1, 1)
    assert df.schema["start_time"] == pl.Float64


def test_scan_nwb_select_reuses_scan_carried_hdf5_snapshot(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.tables")
    source_url = local_hdf5_path.as_uri()

    lf = lazynwb.scan_nwb(
        source_url,
        "/intervals/trials",
        disable_progress=True,
    ).select("start_time")

    def _fail_duplicate_snapshot_lookup(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialization should use the scan-carried snapshot")

    monkeypatch.setattr(
        lazynwb.tables,
        "_get_fast_catalog_snapshot_if_available",
        _fail_duplicate_snapshot_lookup,
    )

    df = lf.collect()

    assert df.height > 0
    assert df.schema["start_time"] == pl.Float64
    assert "using scan-carried fast catalog snapshot for materialization" in caplog.text


def test_scan_nwb_head_reuses_scan_carried_hdf5_snapshot_for_length(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.tables")
    source_url = local_hdf5_path.as_uri()

    lf = lazynwb.scan_nwb(
        source_url,
        "/intervals/trials",
        disable_progress=True,
    ).head(1)

    def _fail_duplicate_snapshot_lookup(*args: object, **kwargs: object) -> None:
        raise AssertionError("head planning should use the scan-carried snapshot")

    monkeypatch.setattr(
        lazynwb.tables,
        "_get_fast_catalog_snapshot_if_available",
        _fail_duplicate_snapshot_lookup,
    )

    df = lf.collect()

    assert df.height == 1
    assert "resolved table length" in caplog.text
    assert "scan-carried catalog snapshot" in caplog.text


def test_hdf5_backend_reader_preserves_metadata_and_timeseries_backend_facts(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="lazynwb._hdf5.parser")
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    general_snapshot = asyncio.run(reader.read_table_schema_snapshot("general"))
    rate_snapshot = asyncio.run(
        reader.read_table_schema_snapshot("processing/behavior/running_speed_with_rate")
    )
    timestamps_snapshot = asyncio.run(
        reader.read_table_schema_snapshot(
            "processing/behavior/running_speed_with_timestamps"
        )
    )
    general_columns = {column.name: column for column in general_snapshot.columns}
    rate_columns = {column.name: column for column in rate_snapshot.columns}
    timestamps_columns = {column.name: column for column in timestamps_snapshot.columns}

    assert general_snapshot.backend == "hdf5"
    assert general_columns["session_start_time"].dataset.path == "session_start_time"
    assert general_columns["session_start_time"].dataset.hdf5_storage_size is not None
    assert rate_columns["starting_time"].attrs["rate"] > 0
    assert rate_columns["starting_time"].dataset.path.endswith("starting_time")
    assert timestamps_columns["data"].dataset.hdf5_data_offset is not None
    assert timestamps_columns["timestamps"].dataset.hdf5_data_offset is not None
    assert "using metadata-only schema facts for tiny metadata column" in caplog.text
    assert "resolved selected TimeSeries rate attr" in caplog.text


def test_hdf5_backend_reader_preserves_unaligned_timeseries_sidecar_facts(
    tmp_path: pathlib.Path,
) -> None:
    nwb_path = tmp_path / "unaligned_timeseries.nwb"
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("acquisition/run")
        group.attrs["neurodata_type"] = "TimeSeries"
        group.create_dataset("data", data=np.arange(4, dtype=np.float64))
        group.create_dataset("timestamps", data=np.arange(4, dtype=np.float64))
        group.create_dataset("short_sidecar", data=np.arange(3, dtype=np.float64))

    reader = _buffer_hdf5_reader(nwb_path, tmp_path / "catalog.sqlite")

    snapshot = asyncio.run(reader.read_table_schema_snapshot("acquisition/run"))
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert (
        columns_by_name["short_sidecar"].dataset.path == "acquisition/run/short_sidecar"
    )
    assert columns_by_name["short_sidecar"].dataset.shape == (3,)
    assert columns_by_name["short_sidecar"].dataset.hdf5_data_offset is not None


def test_public_multifile_fast_hdf5_preserves_schema_merge_semantics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.WARNING, logger="lazynwb.tables")
    float_file = _write_schema_table(tmp_path / "float.nwb", np.float64)
    int_file_0 = _write_schema_table(tmp_path / "int-0.nwb", np.int64)
    int_file_1 = _write_schema_table(tmp_path / "int-1.nwb", np.int64)
    source_urls = tuple(path.as_uri() for path in (float_file, int_file_0, int_file_1))

    schema = lazynwb.tables.get_table_schema(source_urls, "/table")
    first_only_schema = lazynwb.tables.get_table_schema(
        source_urls,
        "/table",
        first_n_files_to_infer_schema=1,
        exclude_internal_columns=True,
    )

    assert schema["value"] == pl.Int64
    assert schema[lazynwb.tables.NWB_PATH_COLUMN_NAME] == pl.String
    assert schema[lazynwb.tables.TABLE_PATH_COLUMN_NAME] == pl.String
    assert schema[lazynwb.tables.TABLE_INDEX_COLUMN_NAME] == pl.UInt32
    assert first_only_schema == pl.Schema({"value": pl.Float64})
    assert "Column 'value' has inconsistent types across files" in caplog.text


def test_public_multifile_fast_hdf5_schema_batch_avoids_threadpool(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_urls = tuple(
        path.as_uri()
        for path in (
            _write_schema_table(tmp_path / "left.nwb", np.float64),
            _write_schema_table(tmp_path / "right.nwb", np.float64),
        )
    )

    def _fail_threadpool() -> None:
        raise AssertionError("fast HDF5 schema batch should not use the threadpool")

    monkeypatch.setattr(lazynwb.utils, "get_threadpool_executor", _fail_threadpool)

    schema = lazynwb.tables.get_table_schema(
        source_urls,
        "/table",
        exclude_internal_columns=True,
    )

    assert schema == pl.Schema({"value": pl.Float64})


def test_public_fast_hdf5_missing_table_preserves_raise_on_missing(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_url = local_hdf5_path.as_uri()

    with pytest.raises(lazynwb.exceptions.InternalPathError, match="not found in"):
        lazynwb.tables.get_table_schema(
            source_url,
            "/not_a_table",
            raise_on_missing=False,
        )
    with pytest.raises(lazynwb.exceptions.InternalPathError, match="not found in"):
        lazynwb.tables.get_table_schema(
            source_url,
            "/not_a_table",
            raise_on_missing=True,
        )


def test_public_multifile_fast_hdf5_reuses_per_source_cache_entries(
    local_hdf5_paths: list[pathlib.Path],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    source_urls = tuple(path.as_uri() for path in local_hdf5_paths)

    lazynwb.tables.get_table_schema(
        source_urls,
        "/intervals/trials",
        exclude_internal_columns=True,
    )
    lazynwb.tables.get_table_schema(
        source_urls[1:],
        "/intervals/trials",
        exclude_internal_columns=True,
    )
    warm_reader = hdf5_reader._default_hdf5_backend_reader(source_urls[1])
    warm_snapshot = asyncio.run(
        warm_reader.read_table_schema_snapshot("intervals/trials")
    )

    assert warm_snapshot.table_path == "intervals/trials"
    assert warm_reader._range_reader.request_count == 0


def test_hdf5_backend_reader_fails_fast_with_structured_parser_error(
    tmp_path: pathlib.Path,
) -> None:
    identity = catalog_models._SourceIdentity(
        source_url="memory://broken-hdf5",
        content_length=128,
        version_id="broken-v1",
    )
    range_reader = hdf5_range_reader._BufferRangeReader(
        hdf5_range_reader._HDF5_SIGNATURE + (b"\x00" * 120),
        source_identity=identity,
    )
    reader = hdf5_reader._HDF5BackendReader(
        "memory://broken-hdf5",
        range_reader=range_reader,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )

    with pytest.raises(hdf5_reader._HDF5ParserError) as exc_info:
        asyncio.run(reader.read_table_schema_snapshot("intervals/trials"))

    assert exc_info.value.source_url == "memory://broken-hdf5"
    assert exc_info.value.table_path == "intervals/trials"
    assert exc_info.value.feature == "hdf5_metadata_parser"


@pytest.mark.skipif(
    os.environ.get("LAZYNWB_REMOTE_SCHEMA_TESTS") != "1",
    reason="set LAZYNWB_REMOTE_SCHEMA_TESTS=1 to run remote schema integration test",
)
def test_remote_hdf5_scalar_schema_integration(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LAZYNWB_CATALOG_CACHE_PATH",
        str(tmp_path / "catalog.sqlite"),
    )
    url = "https://dandiarchive.s3.amazonaws.com/blobs/56c/31a/56c31a1f-a6fb-4b73-ab7d-98fb5ef9a553"

    schema = lazynwb.tables.get_table_schema(
        url,
        "/intervals/trials",
        exclude_internal_columns=True,
    )

    assert {"start_time", "stop_time"}.issubset(schema)


def _buffer_hdf5_reader(
    path: pathlib.Path,
    cache_path: pathlib.Path,
) -> hdf5_reader._HDF5BackendReader:
    data = path.read_bytes()
    identity = catalog_models._SourceIdentity(
        source_url=path.as_uri(),
        content_length=len(data),
        version_id=f"test-{path.stat().st_mtime_ns}",
    )
    range_reader = hdf5_range_reader._BufferRangeReader(
        data,
        source_identity=identity,
    )
    return hdf5_reader._HDF5BackendReader(
        path.as_uri(),
        range_reader=range_reader,
        cache=cache_sqlite._SQLiteSnapshotCache(cache_path),
    )


def _snapshot_schema(snapshot: catalog_models._TableSchemaSnapshot) -> pl.Schema:
    return catalog_polars._snapshot_to_polars_schema(snapshot)


def _existing_schema(path: pathlib.Path, table_path: str) -> pl.Schema:
    raw_columns = lazynwb.get_table_column_metadata(path, table_path)
    return lazynwb.tables.get_table_schema_from_metadata(raw_columns)


def _h5py_indexed_column(
    data: np.ndarray,
    index_values: np.ndarray,
) -> list[list[float]]:
    starts = np.empty(index_values.size + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = index_values
    return [
        data[int(start) : int(end)].tolist()
        for start, end in zip(starts[:-1], starts[1:])
    ]


def _write_schema_table(path: pathlib.Path, dtype: type[np.generic]) -> pathlib.Path:
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("table")
        group.create_dataset("value", data=np.array([1, 2], dtype=dtype))
    return path
