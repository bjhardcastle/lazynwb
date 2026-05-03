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
import lazynwb.tables


def test_hdf5_backend_reader_parses_scalar_table_from_byte_buffer(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    snapshot = asyncio.run(reader.read_table_schema_snapshot("intervals/trials"))
    schema = catalog_polars._snapshot_to_polars_schema(snapshot)

    assert snapshot.backend == "hdf5"
    assert snapshot.table_path == "intervals/trials"
    assert {"start_time", "stop_time", "condition"}.issubset(schema)
    assert schema["start_time"] == pl.Float64


def test_hdf5_backend_reader_preserves_units_array_catalog_facts(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = _buffer_hdf5_reader(local_hdf5_path, tmp_path / "catalog.sqlite")

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))
    schema = catalog_polars._snapshot_to_polars_schema(snapshot)
    raw_columns = lazynwb.get_table_column_metadata(local_hdf5_path, "/units")
    existing_schema = lazynwb.tables.get_table_schema_from_metadata(raw_columns)
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert schema == existing_schema
    assert columns_by_name["spike_times"].is_nominally_indexed
    assert columns_by_name["spike_times"].index_column_name == "spike_times_index"
    assert columns_by_name["spike_times"].dataset.path == "units/spike_times"
    assert columns_by_name["spike_times"].row_element_shape == ()
    assert "shape" in columns_by_name["spike_times"].dataset.read_capabilities
    assert columns_by_name["obs_intervals"].row_element_shape == (2,)
    assert columns_by_name["waveform_mean"].is_multidimensional
    assert columns_by_name["waveform_mean"].row_element_shape == (
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
    assert {"spike_times", "waveform_mean", "obs_intervals"}.issubset(
        warm_column_names
    )
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


def test_hdf5_backend_reader_matches_metadata_and_timeseries_schema_parity(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="lazynwb.table_metadata")
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

    assert general_snapshot.table_length == 1
    assert _snapshot_schema(general_snapshot) == _existing_schema(
        local_hdf5_path, "/general"
    )
    assert "session_start_time" in _snapshot_schema(general_snapshot)
    rate_schema = _snapshot_schema(rate_snapshot)
    assert rate_schema == _existing_schema(
        local_hdf5_path,
        "processing/behavior/running_speed_with_rate",
    )
    assert "starting_time" not in rate_schema
    assert rate_schema["timestamps"] == pl.Float64
    timestamps_schema = _snapshot_schema(timestamps_snapshot)
    assert timestamps_schema == _existing_schema(
        local_hdf5_path,
        "processing/behavior/running_speed_with_timestamps",
    )
    assert {"data", "timestamps"}.issubset(timestamps_schema)
    assert "using metadata-only schema facts for tiny metadata column" in caplog.text
    assert "resolved selected TimeSeries rate attr" in caplog.text


def test_hdf5_backend_timeseries_schema_skips_unaligned_columns(
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
    schema = _snapshot_schema(snapshot)

    assert {"data", "timestamps"}.issubset(schema)
    assert "short_sidecar" not in schema


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
