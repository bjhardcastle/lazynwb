from __future__ import annotations

import asyncio
import dataclasses
import pathlib

import h5py
import polars as pl
import pytest
import zarr

import lazynwb
import lazynwb._catalog.accessor as catalog_accessor
import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb._catalog.polars as catalog_polars
import lazynwb._zarr.reader as zarr_reader
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

    assert catalog_schema == existing_schema


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


def test_accessor_backend_reader_requires_exact_normalized_paths(
    local_hdf5_path: pathlib.Path,
) -> None:
    reader = catalog_accessor._AccessorBackendReader(local_hdf5_path)

    with pytest.raises(ValueError, match="exact normalized"):
        asyncio.run(reader.read_table_schema_snapshot("/units"))


def test_zarr_backend_reader_is_private_skeleton(local_zarr_path: pathlib.Path) -> None:
    reader = zarr_reader._ZarrBackendReader(local_zarr_path)

    assert "TODO" in (zarr_reader._ZarrBackendReader.__doc__ or "")
    assert isinstance(reader, catalog_backend._BackendReader)
    with pytest.raises(NotImplementedError, match="private skeleton"):
        asyncio.run(reader.read_table_schema_snapshot("units"))


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
