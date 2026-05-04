from __future__ import annotations

import asyncio
import dataclasses
import logging
import pathlib
import shutil

import h5py
import numpy as np
import polars as pl
import pytest
import zarr

import lazynwb
import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.accessor as catalog_accessor
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._catalog.polars as catalog_polars
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb._zarr.reader as zarr_reader
import lazynwb.conversion as conversion
import lazynwb.file_io
import lazynwb.table_metadata
import lazynwb.tables
import lazynwb.types_


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
def test_accessor_backend_reader_builds_accessor_free_catalog_snapshots(
    nwb_fixture_name: str,
    expected_backend: str,
    table_path: str,
    request: pytest.FixtureRequest,
) -> None:
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    exact_table_path = lazynwb.normalize_internal_file_path(table_path)
    reader = catalog_accessor._AccessorBackendReader(nwb_path)

    snapshot = asyncio.run(reader.read_table_schema_snapshot(exact_table_path))

    assert isinstance(reader, catalog_backend._BackendReader)
    assert snapshot.backend == expected_backend
    assert snapshot.table_path == exact_table_path
    assert snapshot.columns
    assert all(dataclasses.is_dataclass(column) for column in snapshot.columns)
    assert not _contains_live_accessor(snapshot)
    assert {column.backend for column in snapshot.columns} == {expected_backend}
    assert all(column.table_path == exact_table_path for column in snapshot.columns)


@pytest.mark.parametrize("nwb_fixture_name", ["local_hdf5_path", "local_zarr_path"])
@pytest.mark.parametrize(
    "table_path",
    [
        "/units",
        "/intervals/trials",
        "/general",
        "processing/behavior/running_speed_with_rate",
    ],
)
def test_accessor_catalog_schema_matches_existing_raw_metadata_schema(
    nwb_fixture_name: str,
    table_path: str,
    request: pytest.FixtureRequest,
) -> None:
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    exact_table_path = lazynwb.normalize_internal_file_path(table_path)
    reader = catalog_accessor._AccessorBackendReader(nwb_path)

    snapshot = asyncio.run(reader.read_table_schema_snapshot(exact_table_path))
    catalog_schema = catalog_polars._snapshot_to_polars_schema(snapshot)
    raw_columns = lazynwb.table_metadata.get_table_column_metadata(
        nwb_path,
        exact_table_path,
    )
    existing_schema = lazynwb.tables.get_table_schema_from_metadata(raw_columns)
    existing_table_length = lazynwb.table_metadata.get_table_length_from_metadata(
        raw_columns
    )

    assert catalog_schema == existing_schema
    assert snapshot.table_length == existing_table_length


def test_accessor_catalog_schema_preserves_units_array_facts(
    local_hdf5_path: pathlib.Path,
) -> None:
    reader = catalog_accessor._AccessorBackendReader(local_hdf5_path)

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))
    columns_by_name = {column.name: column for column in snapshot.columns}
    schema = catalog_polars._snapshot_to_polars_schema(snapshot)

    assert columns_by_name["spike_times"].is_nominally_indexed
    assert columns_by_name["spike_times_index"].is_index_column
    assert columns_by_name["waveform_mean"].is_multidimensional
    assert schema["spike_times"] == pl.List(pl.Float64)
    assert isinstance(schema["waveform_mean"], pl.Array)


def test_catalog_polars_schema_converts_reference_dtype_to_string() -> None:
    dtype = catalog_models._NeutralDType(kind="reference", numpy_dtype="|O")

    assert catalog_polars._neutral_dtype_to_polars_base(dtype) == pl.String


def test_neutral_dtype_model_classifies_supported_schema_dtypes() -> None:
    cases = {
        "numeric": np.dtype("int64"),
        "bool": np.dtype("bool"),
        "string": np.dtype("S8"),
        "vlen_string": h5py.string_dtype(encoding="utf-8"),
        "enum": h5py.enum_dtype({"a": 1, "b": 2}, basetype="i"),
        "array": np.dtype((np.float64, (2,))),
        "reference": h5py.ref_dtype,
        "compound": np.dtype([("x", "i4"), ("y", "f8")]),
        "opaque": np.dtype("V8"),
    }

    classified = {
        expected_kind: catalog_models._NeutralDType.from_backend_dtype(dtype)
        for expected_kind, dtype in cases.items()
    }

    assert {kind: dtype.kind for kind, dtype in classified.items()} == {
        kind: kind for kind in cases
    }
    assert classified["array"].element_numpy_dtype == "<f8"
    assert classified["array"].element_shape == (2,)
    assert catalog_models._NeutralDType.from_backend_dtype(None).kind == "unknown"


def test_accessor_backend_reader_requires_exact_normalized_paths(
    local_hdf5_path: pathlib.Path,
) -> None:
    reader = catalog_accessor._AccessorBackendReader(local_hdf5_path)

    with pytest.raises(ValueError, match="exact normalized"):
        asyncio.run(reader.read_table_schema_snapshot("/units"))


@pytest.mark.parametrize(
    "table_path",
    [
        "/units",
        "/intervals/trials",
        "/general",
        "processing/behavior/running_speed_with_rate",
    ],
)
def test_zarr_backend_reader_builds_table_schema_snapshots(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    table_path: str,
) -> None:
    exact_table_path = lazynwb.normalize_internal_file_path(table_path)
    reader = zarr_reader._ZarrBackendReader(
        local_zarr_path,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )

    snapshot = asyncio.run(reader.read_table_schema_snapshot(exact_table_path))
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert isinstance(reader, catalog_backend._BackendReader)
    assert reader.used_consolidated_metadata
    assert reader.metadata_read_count == 1
    assert snapshot.backend == "zarr"
    assert snapshot.table_path == exact_table_path
    assert snapshot.columns
    assert {column.backend for column in snapshot.columns} == {"zarr"}
    assert all(
        "metadata" in column.dataset.read_capabilities for column in snapshot.columns
    )
    assert any(
        column.dataset.storage_layout == "chunked"
        for column in columns_by_name.values()
        if column.is_dataset
    )


def test_zarr_backend_reader_reuses_sqlite_snapshot(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    cache_path = tmp_path / "catalog.sqlite"
    cold_reader = zarr_reader._ZarrBackendReader(
        local_zarr_path,
        cache=cache_sqlite._SQLiteSnapshotCache(cache_path),
    )

    cold_snapshot = asyncio.run(cold_reader.read_table_schema_snapshot("units"))
    warm_reader = zarr_reader._ZarrBackendReader(
        local_zarr_path,
        cache=cache_sqlite._SQLiteSnapshotCache(cache_path),
    )
    warm_snapshot = asyncio.run(warm_reader.read_table_schema_snapshot("units"))

    assert cold_reader.metadata_read_count == 1
    assert warm_reader.metadata_read_count == 0
    assert warm_snapshot == cold_snapshot


def test_zarr_backend_reader_uses_targeted_metadata_without_consolidated(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    copied_zarr = tmp_path / "without-consolidated.nwb.zarr"
    shutil.copytree(local_zarr_path, copied_zarr)
    (copied_zarr / ".zmetadata").unlink()
    reader = zarr_reader._ZarrBackendReader(
        copied_zarr,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))
    columns_by_name = {column.name: column for column in snapshot.columns}

    assert not reader.used_consolidated_metadata
    assert reader.metadata_read_count > 1
    assert columns_by_name["spike_times"].dataset.path == "units/spike_times"
    assert columns_by_name["spike_times"].dataset.chunks is not None
    assert columns_by_name["spike_times"].dataset.storage_layout == "chunked"
    assert "chunked" in columns_by_name["spike_times"].dataset.read_capabilities


def test_zarr_backend_reader_builds_path_summary(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    reader = zarr_reader._ZarrBackendReader(
        local_zarr_path,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )

    summary = asyncio.run(reader.read_path_summary())
    entries_by_path = {entry.path: entry for entry in summary}

    assert reader.used_consolidated_metadata
    assert reader.metadata_read_count == 1
    assert "/intervals/trials" in entries_by_path
    assert "/units" in entries_by_path
    assert "/units/spike_times" in entries_by_path
    assert "/general" in entries_by_path
    assert "/general/subject" in entries_by_path
    assert "/processing/behavior/running_speed_with_rate/data" in entries_by_path
    assert entries_by_path["/units"].is_group
    assert entries_by_path["/units/spike_times"].is_dataset
    assert entries_by_path["/units/spike_times"].shape is not None


def test_zarr_catalog_path_summary_filters_metadata_timeseries_and_specifications(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.file_io")

    summary = lazynwb.file_io._get_catalog_path_summary_if_available(
        local_zarr_path,
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
    assert "used zarr catalog summary" in caplog.text


def test_zarr_common_path_discovery_matches_accessor_and_uses_catalog_summary(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    with monkeypatch.context() as fallback_patch:
        fallback_patch.setattr(
            lazynwb.file_io,
            "_get_catalog_path_summary_if_available",
            lambda *args, **kwargs: None,
        )
        accessor_paths = conversion._find_common_paths(
            (local_zarr_path,),
            min_file_count=1,
            disable_progress=True,
            include_timeseries=True,
            include_metadata=True,
        )

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("Zarr discovery should use the catalog path summary")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    catalog_paths = conversion._find_common_paths(
        (local_zarr_path,),
        min_file_count=1,
        disable_progress=True,
        include_timeseries=True,
        include_metadata=True,
    )

    assert catalog_paths == accessor_paths
    assert "/intervals/trials" in catalog_paths
    assert "/units" in catalog_paths
    assert "/general" in catalog_paths
    assert "/general/subject" in catalog_paths
    assert "/processing/behavior/running_speed_with_rate" in catalog_paths


def test_zarr_sql_context_discovery_uses_catalog_summary_without_accessor(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    caplog.set_level(logging.DEBUG, logger="lazynwb.conversion")
    caplog.set_level(logging.DEBUG, logger="lazynwb.file_io")

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("SQL context discovery should use Zarr catalog summary")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)

    sql_context = conversion.get_sql_context(
        (local_zarr_path,),
        table_names=("trials",),
        disable_progress=True,
        exclude_timeseries=True,
    )

    assert "trials" in sql_context.tables()
    assert "path discovery used catalog summary" in caplog.text


def test_zarr_path_summary_uses_targeted_metadata_without_consolidated(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    copied_zarr = tmp_path / "path-summary-without-consolidated.nwb.zarr"
    shutil.copytree(local_zarr_path, copied_zarr)
    (copied_zarr / ".zmetadata").unlink()
    caplog.set_level(logging.DEBUG, logger="lazynwb._zarr.reader")
    reader = zarr_reader._ZarrBackendReader(
        copied_zarr,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )

    summary = asyncio.run(reader.read_path_summary())
    paths = {entry.path for entry in summary}

    assert not reader.used_consolidated_metadata
    assert reader.metadata_read_count > 1
    assert "/units" in paths
    assert "/units/spike_times" in paths
    assert "/processing/behavior/running_speed_with_rate/data" in paths
    assert "read targeted Zarr metadata file" in caplog.text


def test_public_get_table_schema_uses_zarr_backend_for_local_store(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))

    schema = lazynwb.tables.get_table_schema(
        local_zarr_path,
        "/units",
        exclude_internal_columns=True,
    )
    warm_reader = zarr_reader._ZarrBackendReader(
        local_zarr_path,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )
    warm_snapshot = asyncio.run(warm_reader.read_table_schema_snapshot("units"))
    raw_columns = lazynwb.table_metadata.get_table_column_metadata(
        local_zarr_path,
        "/units",
    )
    existing_schema = lazynwb.tables.get_table_schema_from_metadata(raw_columns)

    assert isinstance(warm_reader, catalog_backend._BackendReader)
    assert schema == existing_schema
    assert warm_snapshot.table_path == "units"
    assert warm_reader.metadata_read_count == 0


def test_fast_catalog_snapshot_skips_local_zarr_probe_for_remote_hdf5_upath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = lazynwb.file_io.from_pathlike(
        "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
        "620263_2022-07-26.nwb"
    )

    def _not_hdf5_candidate(candidate_source: lazynwb.types_.PathLike) -> bool:
        assert candidate_source == source
        return False

    monkeypatch.setattr(
        hdf5_reader,
        "_is_fast_hdf5_candidate",
        _not_hdf5_candidate,
    )

    snapshot = lazynwb.tables._get_fast_catalog_snapshot_if_available(
        source,
        "intervals/trials",
    )

    assert snapshot is None


def test_fast_catalog_snapshot_prefers_hdf5_for_non_zarr_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = lazynwb.file_io.from_pathlike(
        "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
        "620263_2022-07-26.nwb"
    )
    expected_snapshot = catalog_models._TableSchemaSnapshot(
        source_identity=catalog_models._SourceIdentity(
            source_url=str(source),
            in_process_token="test-hdf5-first",
        ),
        table_path="intervals/trials",
        backend="hdf5",
        columns=(),
    )
    calls: list[str] = []

    class _HDF5Reader:
        async def read_table_schema_snapshot(
            self,
            exact_table_path: str,
        ) -> catalog_models._TableSchemaSnapshot:
            calls.append(f"hdf5:{exact_table_path}")
            return expected_snapshot

        async def close(self) -> None:
            calls.append("hdf5:close")

    def _unexpected_zarr_probe(candidate_source: lazynwb.types_.PathLike) -> bool:
        raise AssertionError(
            f"Zarr should not be probed first for {candidate_source!r}"
        )

    monkeypatch.setattr(
        hdf5_reader,
        "_default_hdf5_backend_reader",
        lambda candidate_source: _HDF5Reader(),
    )
    monkeypatch.setattr(
        zarr_reader,
        "_is_fast_zarr_candidate",
        _unexpected_zarr_probe,
    )

    snapshot = lazynwb.tables._get_fast_catalog_snapshot_if_available(
        source,
        "intervals/trials",
    )

    assert snapshot == expected_snapshot
    assert calls == ["hdf5:intervals/trials", "hdf5:close"]


def test_fast_catalog_snapshot_falls_back_to_zarr_after_hdf5_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "file:///tmp/remote-style-store-without-zarr-suffix"
    expected_snapshot = catalog_models._TableSchemaSnapshot(
        source_identity=catalog_models._SourceIdentity(
            source_url=source,
            in_process_token="test-zarr-fallback",
        ),
        table_path="intervals/trials",
        backend="zarr",
        columns=(),
    )
    calls: list[str] = []

    class _RejectingHDF5Reader:
        async def read_table_schema_snapshot(
            self,
            exact_table_path: str,
        ) -> catalog_models._TableSchemaSnapshot:
            calls.append(f"hdf5:{exact_table_path}")
            raise hdf5_reader._NotHDF5Error(
                source_url=source,
                table_path=exact_table_path,
                feature="signature",
                detail="not an HDF5 file",
            )

        async def close(self) -> None:
            calls.append("hdf5:close")

    class _ZarrReader:
        async def read_table_schema_snapshot(
            self,
            exact_table_path: str,
        ) -> catalog_models._TableSchemaSnapshot:
            calls.append(f"zarr:{exact_table_path}")
            return expected_snapshot

        async def close(self) -> None:
            calls.append("zarr:close")

    monkeypatch.setattr(
        hdf5_reader,
        "_default_hdf5_backend_reader",
        lambda candidate_source: _RejectingHDF5Reader(),
    )
    monkeypatch.setattr(
        zarr_reader,
        "_is_fast_zarr_candidate",
        lambda candidate_source: True,
    )
    monkeypatch.setattr(
        zarr_reader,
        "_default_zarr_backend_reader",
        lambda candidate_source: _ZarrReader(),
    )

    snapshot = lazynwb.tables._get_fast_catalog_snapshot_if_available(
        source,
        "intervals/trials",
    )

    assert snapshot == expected_snapshot
    assert calls == [
        "hdf5:intervals/trials",
        "hdf5:close",
        "zarr:intervals/trials",
        "zarr:close",
    ]


def test_fast_catalog_snapshot_prefers_explicit_zarr_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "file:///tmp/example.nwb.zarr"
    expected_snapshot = catalog_models._TableSchemaSnapshot(
        source_identity=catalog_models._SourceIdentity(
            source_url=source,
            in_process_token="test-zarr-first",
        ),
        table_path="intervals/trials",
        backend="zarr",
        columns=(),
    )
    calls: list[str] = []

    class _ZarrReader:
        async def read_table_schema_snapshot(
            self,
            exact_table_path: str,
        ) -> catalog_models._TableSchemaSnapshot:
            calls.append(f"zarr:{exact_table_path}")
            return expected_snapshot

        async def close(self) -> None:
            calls.append("zarr:close")

    def _unexpected_hdf5_reader(candidate_source: lazynwb.types_.PathLike) -> object:
        raise AssertionError(
            f"HDF5 should not be probed first for {candidate_source!r}"
        )

    monkeypatch.setattr(
        zarr_reader,
        "_is_fast_zarr_candidate",
        lambda candidate_source: True,
    )
    monkeypatch.setattr(
        zarr_reader,
        "_default_zarr_backend_reader",
        lambda candidate_source: _ZarrReader(),
    )
    monkeypatch.setattr(
        hdf5_reader,
        "_default_hdf5_backend_reader",
        _unexpected_hdf5_reader,
    )

    snapshot = lazynwb.tables._get_fast_catalog_snapshot_if_available(
        source,
        "intervals/trials",
    )

    assert snapshot == expected_snapshot
    assert calls == ["zarr:intervals/trials", "zarr:close"]


def _contains_live_accessor(value: object) -> bool:
    if isinstance(value, (h5py.Dataset, h5py.Group, h5py.File, zarr.Array, zarr.Group)):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_live_accessor(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, catalog_models._DatasetSchema):
        return False
    if isinstance(value, dict):
        return any(
            _contains_live_accessor(key) or _contains_live_accessor(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set)):
        return any(_contains_live_accessor(item) for item in value)
    return False
