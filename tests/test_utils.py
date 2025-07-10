import logging

import pytest

import lazynwb

expected_paths = (
    "/processing/behavior/running_speed_with_timestamps/data",
    "/processing/behavior/running_speed_with_timestamps/timestamps",
    "/processing/behavior/running_speed_with_rate/data",
    "/units",
    "/intervals/trials",
    "/intervals/epochs",
) # excluding metadata, specifications and table columns


def test_get_nwb_file_structure_hdf5(local_hdf5_path):
    """Test get_nwb_file_structure with HDF5 file."""
    structure = lazynwb.get_internal_paths(local_hdf5_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    assert (
        extras := set(structure.keys()) - set(expected_paths)
    ) == set(), f"Additional unexpected paths found: {extras}"

    # Check that we can inspect the datasets
    units_group = structure["/units"]
    assert hasattr(units_group, "keys"), "Units should be a group with keys"


def test_get_nwb_file_structure_zarr(local_zarr_path):
    """Test get_nwb_file_structure with Zarr file."""
    structure = lazynwb.get_internal_paths(local_zarr_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    assert (
        extras := set(structure.keys()) - set(expected_paths)
    ) == set(), f"Additional unexpected paths found: {extras}"

    # Check that we can inspect the datasets
    units_group = structure["/units"]
    assert hasattr(units_group, "keys"), "Units should be a group with keys"


def test_get_nwb_file_structure_filtering(local_hdf5_path):
    """Test get_nwb_file_structure with different filtering options."""
    # Test with all filtering disabled
    structure_all = lazynwb.get_internal_paths(
        local_hdf5_path,
        include_arrays=True,
        include_specifications=True,
        include_table_columns=True,
        include_metadata=True,
    )

    # Test with default filtering
    structure_filtered = lazynwb.get_internal_paths(local_hdf5_path)

    # Filtered structure should have fewer or equal entries
    assert len(structure_filtered) <= len(structure_all)

    # Table columns should now be present
    for path in (
        # check table columns:
        "/intervals/trials/condition",
        "/intervals/epochs/tags",
        # check specifications:
        "/specifications/core/",
        # check metadata:
        "/general/subject",
    ):
        assert path in structure_all or any(
            p.startswith(path) for p in structure_all
        ), f"Expected paths starting with {path} not found in filtered structure"


def test_normalize_internal_file_path():
    """Test normalize_internal_file_path function."""
    # Test path without leading slash
    assert (
        lazynwb.normalize_internal_file_path("units/spike_times")
        == "units/spike_times"
    )

    # Test path with leading slash
    assert (
        lazynwb.normalize_internal_file_path("/units/spike_times")
    == "units/spike_times"
    )

    # Test empty path
    assert lazynwb.normalize_internal_file_path("") == "/"

    # Test root path
    assert lazynwb.normalize_internal_file_path("/") == "/"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
