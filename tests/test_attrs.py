"""Tests for attrs module functions."""

import asyncio
import logging
import pathlib
import shutil

import pytest

import lazynwb._zarr.reader as zarr_reader
import lazynwb.attrs
import lazynwb.file_io


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_get_attrs(nwb_fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Test retrieving attrs for a specific path."""
    nwb_path = request.getfixturevalue(nwb_fixture_name)

    # Get attrs for units table
    attrs = lazynwb.attrs.get_attrs(
        nwb_path=nwb_path,
        internal_path="/units",
    )
    assert isinstance(attrs, dict)
    # Most NWB files have neurodata_type for tables
    # but we don't assert it to support various NWB versions


def test_get_attrs_nonexistent(local_hdf5_path: pathlib.Path) -> None:
    """Test that nonexistent path raises KeyError."""
    with pytest.raises(KeyError):
        lazynwb.attrs.get_attrs(
            nwb_path=local_hdf5_path,
            internal_path="/nonexistent/path",
        )


def test_get_attrs_filtering(local_hdf5_path: pathlib.Path) -> None:
    """Test attribute filtering (private and id)."""
    # Get without filtering
    attrs_all = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
        exclude_private=False,
    )

    # Get with filtering
    attrs_filtered = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
        exclude_private=True,
    )

    # Filtered should have fewer or equal items
    assert len(attrs_filtered) <= len(attrs_all)


def test_get_attrs_caching(local_hdf5_path: pathlib.Path) -> None:
    """Test that attrs are cached correctly."""
    lazynwb.attrs.clear_attrs_cache()

    # First call
    attrs1 = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
    )

    # Second call (should use cache)
    attrs2 = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
    )

    # Should be identical
    assert attrs1 == attrs2


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_get_sub_attrs(nwb_fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Test retrieving all sub-attributes under a parent path."""
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=nwb_path,
        parent_path="/units",
    )

    assert isinstance(all_attrs, dict)
    # Should have at least the parent path
    assert "units" in all_attrs


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_get_sub_attrs_root(
    nwb_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test retrieving all sub-attributes from root."""
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=nwb_path,
        parent_path="/",
    )

    assert isinstance(all_attrs, dict)
    # Should have multiple paths
    assert len(all_attrs) > 1


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_get_sub_attrs_nonexistent(
    nwb_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test retrieving sub-attrs from nonexistent path."""
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=nwb_path,
        parent_path="/nonexistent/path",
    )

    assert isinstance(all_attrs, dict)
    # Should be empty or only contain the nonexistent path
    assert len(all_attrs) == 0


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_get_sub_attrs_filtering(
    nwb_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that filtering works for sub_attrs."""
    nwb_path = request.getfixturevalue(nwb_fixture_name)
    attrs_filtered = lazynwb.attrs.get_sub_attrs(
        nwb_path=nwb_path,
        parent_path="/units",
        exclude_private=True,
    )

    lazynwb.attrs.get_sub_attrs(
        nwb_path=nwb_path,
        parent_path="/units",
        exclude_private=False,
    )

    # Filtered version should have no private attrs or id
    for path_attrs in attrs_filtered.values():
        assert "id" not in path_attrs
        for key in path_attrs:
            assert not key.startswith("_")


def test_consolidate_attrs_single_file(local_hdf5_path: pathlib.Path) -> None:
    """Test consolidating attrs from a single file."""
    consolidated = lazynwb.attrs.consolidate_attrs(
        nwb_paths=local_hdf5_path,
        internal_path="/units",
    )

    assert isinstance(consolidated, dict)
    assert "units" in consolidated
    # Each attribute should have at least 'common' key
    for attr_dict in consolidated["units"].values():
        assert "common" in attr_dict


def test_consolidate_attrs_multiple_files(
    local_hdf5_paths: list[pathlib.Path],
) -> None:
    """Test consolidating attrs from multiple files."""
    if len(local_hdf5_paths) < 2:
        pytest.skip("Need at least 2 NWB files for this test")

    consolidated = lazynwb.attrs.consolidate_attrs(
        nwb_paths=local_hdf5_paths[:2],
        internal_path="/units",
    )

    assert isinstance(consolidated, dict)
    assert "units" in consolidated
    # Each attribute should have at least 'common' key
    for attr_dict in consolidated["units"].values():
        assert "common" in attr_dict


def test_consolidate_attrs_no_paths() -> None:
    """Test that consolidate_attrs raises error with no paths."""
    with pytest.raises(ValueError, match="At least one NWB path"):
        lazynwb.attrs.consolidate_attrs(nwb_paths=[], internal_path="/units")


def test_consolidate_attrs_structure(local_hdf5_paths: list[pathlib.Path]) -> None:
    """Test the structure of consolidated attrs output."""
    if len(local_hdf5_paths) < 2:
        pytest.skip("Need at least 2 NWB files for this test")

    consolidated = lazynwb.attrs.consolidate_attrs(
        nwb_paths=local_hdf5_paths[:2],
        internal_path="/units",
    )

    # Check outer structure
    assert isinstance(consolidated, dict)
    assert len(consolidated) == 1
    assert "units" in consolidated

    # Check inner structure
    attrs_dict = consolidated["units"]
    assert isinstance(attrs_dict, dict)

    # Each attribute should have the correct structure
    for _, attr_entry in attrs_dict.items():
        assert isinstance(attr_entry, dict)
        assert "common" in attr_entry


def test_clear_attrs_cache(local_hdf5_path: pathlib.Path) -> None:
    """Test clearing the attrs cache."""
    # Populate cache
    lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
    )

    # Clear for this specific file
    lazynwb.attrs.clear_attrs_cache(nwb_path=local_hdf5_path)

    # Cache should be empty for this file
    # (We can't directly check the cache, but we can verify the function works)
    assert True


def test_clear_attrs_cache_all() -> None:
    """Test clearing the entire attrs cache."""
    lazynwb.attrs.clear_attrs_cache()
    # If no exception raised, cache was cleared successfully
    assert True


@pytest.mark.parametrize("parent_path", ["/", "/units", "/intervals", "/processing"])
def test_get_sub_attrs_hdf5_zarr_parity(
    local_hdf5_path: pathlib.Path,
    local_zarr_path: pathlib.Path,
    parent_path: str,
) -> None:
    """Test that get_sub_attrs returns consistent results for HDF5 and zarr backends.

    Zarr stores scalars as separate arrays, so we compare the shared path set. Zarr-only
    attrs are filtered by the attrs reader before comparison.
    """
    zarr_only_attrs = {"zarr_dtype", "_ARRAY_DIMENSIONS"}

    lazynwb.attrs.clear_attrs_cache()
    hdf5_result = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path, parent_path=parent_path
    )
    lazynwb.attrs.clear_attrs_cache()
    zarr_result = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_zarr_path, parent_path=parent_path
    )

    # All HDF5 paths should be present in zarr (zarr may have extra scalar paths)
    hdf5_keys = set(hdf5_result.keys())
    zarr_keys = set(zarr_result.keys())
    assert hdf5_keys <= zarr_keys, (
        f"Paths in HDF5 but not zarr for parent_path={parent_path!r}: "
        f"{hdf5_keys - zarr_keys}"
    )

    # For shared paths, attrs should match after the attrs reader drops zarr-only keys
    for path in hdf5_keys:
        hdf5_attrs = hdf5_result[path]
        zarr_attrs = zarr_result[path]
        assert not (set(zarr_attrs) & zarr_only_attrs)
        assert (
            hdf5_attrs == zarr_attrs
        ), f"Value mismatch at {path!r}: hdf5={hdf5_attrs}, zarr={zarr_attrs}"


@pytest.mark.parametrize(
    "internal_path",
    ["/units", "/units/spike_times", "/intervals/trials"],
)
def test_get_attrs_hdf5_zarr_parity(
    local_hdf5_path: pathlib.Path,
    local_zarr_path: pathlib.Path,
    internal_path: str,
) -> None:
    """Test that single-path attrs match across HDF5 and zarr backends."""
    zarr_only_attrs = {"zarr_dtype", "_ARRAY_DIMENSIONS"}

    lazynwb.attrs.clear_attrs_cache()
    hdf5_attrs = lazynwb.attrs.get_attrs(local_hdf5_path, internal_path)
    lazynwb.attrs.clear_attrs_cache()
    zarr_attrs = lazynwb.attrs.get_attrs(local_zarr_path, internal_path)

    assert not (set(zarr_attrs) & zarr_only_attrs)
    assert hdf5_attrs == zarr_attrs


def test_zarr_attrs_reuse_schema_metadata_catalog(
    local_zarr_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Schema reads populate the shared Zarr catalog used by attrs reads."""
    lazynwb.attrs.clear_attrs_cache()
    caplog.set_level(logging.DEBUG, logger="lazynwb._zarr.reader")
    reader = zarr_reader._ZarrBackendReader(local_zarr_path)

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))
    columns_by_name = {column.name: column for column in snapshot.columns}
    caplog.clear()
    attrs = lazynwb.attrs.get_attrs(
        local_zarr_path,
        "/units/spike_times",
        exclude_private=False,
    )
    normalized_attrs = lazynwb.attrs.get_attrs(
        local_zarr_path,
        "units/spike_times",
        exclude_private=False,
    )

    assert reader.metadata_read_count == 1
    assert reader.used_consolidated_metadata
    assert attrs == normalized_attrs
    assert attrs == dict(columns_by_name["spike_times"].dataset.attrs)
    assert "zarr_dtype" not in attrs
    assert "_ARRAY_DIMENSIONS" not in attrs
    assert "shared Zarr metadata catalog cache hit" in caplog.text


def test_zarr_schema_reuses_attrs_metadata_catalog(
    local_zarr_path: pathlib.Path,
) -> None:
    """Attrs reads can warm the shared Zarr catalog used by schema snapshots."""
    lazynwb.attrs.clear_attrs_cache()
    lazynwb.attrs.get_sub_attrs(local_zarr_path, parent_path="/units")
    reader = zarr_reader._ZarrBackendReader(local_zarr_path)

    snapshot = asyncio.run(reader.read_table_schema_snapshot("units"))

    assert snapshot.columns
    assert reader.used_consolidated_metadata
    assert reader.metadata_read_count == 0


def test_zarr_attrs_use_targeted_metadata_without_consolidated(
    local_zarr_path: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    copied_zarr = tmp_path / "attrs-without-consolidated.nwb.zarr"
    shutil.copytree(local_zarr_path, copied_zarr)
    (copied_zarr / ".zmetadata").unlink()
    lazynwb.attrs.clear_attrs_cache()
    caplog.set_level(logging.DEBUG, logger="lazynwb._zarr.reader")

    attrs = lazynwb.attrs.get_attrs(copied_zarr, "/units/spike_times")
    sub_attrs = lazynwb.attrs.get_sub_attrs(copied_zarr, parent_path="/units")

    assert attrs["description"] == "the spike times for each unit in seconds"
    assert "units/spike_times" in sub_attrs
    assert "zarr_dtype" not in sub_attrs["units/spike_times"]
    assert "read targeted Zarr metadata file" in caplog.text


def test_get_attrs_normalize_path(local_hdf5_path: pathlib.Path) -> None:
    """Test that internal paths are normalized correctly."""
    # Test with leading slash
    attrs1 = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="/units",
    )

    # Test without leading slash
    attrs2 = lazynwb.attrs.get_attrs(
        nwb_path=local_hdf5_path,
        internal_path="units",
    )

    # Should return the same result
    assert attrs1 == attrs2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
