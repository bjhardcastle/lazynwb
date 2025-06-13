import logging

import pytest
import lazynwb


expected_paths = [
    '/processing/behavior/running_speed_with_timestamps/data',
    '/processing/behavior/running_speed_with_timestamps/timestamps',
    '/processing/behavior/running_speed_with_rate/data',
    '/processing/behavior/running_speed_with_rate/starting_time',
    '/units',
    '/intervals/trials',
    '/intervals/epochs'
]

def test_get_nwb_file_structure_hdf5(local_hdf5_path):
    """Test get_nwb_file_structure with HDF5 file."""
    structure = lazynwb.get_nwb_file_structure(local_hdf5_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    
    # Check that we can inspect the datasets
    units_group = structure['/units']
    assert hasattr(units_group, 'keys'), "Units should be a group with keys"


def test_get_nwb_file_structure_zarr(local_zarr_path):
    """Test get_nwb_file_structure with Zarr file."""
    structure = lazynwb.get_nwb_file_structure(local_zarr_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    
    # Check that we can inspect the datasets
    units_group = structure['/units']
    assert hasattr(units_group, 'keys'), "Units should be a group with keys"


def test_get_nwb_file_structure_filtering(local_hdf5_path):
    """Test get_nwb_file_structure with different filtering options."""
    # Test with all filtering disabled
    structure_all = lazynwb.get_nwb_file_structure(
        local_hdf5_path,
        exclude_specifications=False,
        exclude_table_columns=False,
        exclude_metadata=False
    )
    
    # Test with default filtering
    structure_filtered = lazynwb.get_nwb_file_structure(local_hdf5_path)
    
    # Filtered structure should have fewer or equal entries
    assert len(structure_filtered) <= len(structure_all)
    
    # Both should contain main data paths
    for path in ['/units', '/intervals/trials']:
        assert path in structure_all
        assert path in structure_filtered


def test_normalize_internal_file_path():
    """Test normalize_internal_file_path function."""
    # Test path without leading slash
    assert lazynwb.utils.normalize_internal_file_path('units/spike_times') == '/units/spike_times'
    
    # Test path with leading slash
    assert lazynwb.utils.normalize_internal_file_path('/units/spike_times') == '/units/spike_times'
    
    # Test empty path
    assert lazynwb.utils.normalize_internal_file_path('') == '/'
    
    # Test root path
    assert lazynwb.utils.normalize_internal_file_path('/') == '/'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
