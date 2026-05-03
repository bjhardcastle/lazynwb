from __future__ import annotations

import asyncio
import os
import pathlib

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
