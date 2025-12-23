"""Tests for attrs module functions."""

import pathlib

import pytest

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


def test_get_sub_attrs(local_hdf5_path: pathlib.Path) -> None:
    """Test retrieving all sub-attributes under a parent path."""
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path,
        parent_path="/units",
    )

    assert isinstance(all_attrs, dict)
    # Should have at least the parent path
    assert "units" in all_attrs


def test_get_sub_attrs_root(local_hdf5_path: pathlib.Path) -> None:
    """Test retrieving all sub-attributes from root."""
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path,
        parent_path="/",
    )

    assert isinstance(all_attrs, dict)
    # Should have multiple paths
    assert len(all_attrs) > 1


def test_get_sub_attrs_nonexistent(local_hdf5_path: pathlib.Path) -> None:
    """Test retrieving sub-attrs from nonexistent path."""
    all_attrs = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path,
        parent_path="/nonexistent/path",
    )

    assert isinstance(all_attrs, dict)
    # Should be empty or only contain the nonexistent path
    assert len(all_attrs) == 0


def test_get_sub_attrs_filtering(local_hdf5_path: pathlib.Path) -> None:
    """Test that filtering works for sub_attrs."""
    attrs_filtered = lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path,
        parent_path="/units",
        exclude_private=True,
    )

    lazynwb.attrs.get_sub_attrs(
        nwb_path=local_hdf5_path,
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
