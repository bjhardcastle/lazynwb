import logging

import numpy as np
import pytest

import pandas as pd
import lazynwb
import pynwb


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_sources(nwb_fixture_name, request):
    """Test get_timeseries with various NWB file inputs."""
    # Resolve the fixture name to its value (the path to a single NWB file)
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)
    _ = lazynwb.get_timeseries(nwb_path_or_paths, "/processing/behavior/", exact_path=True)

@pytest.mark.parametrize(
    "table_name",
    [
        "running_speed_with_timestamps",
        "running_speed_with_rate",
    ],
)
def test_properties(local_hdf5_path, table_name):
    """Test get_timeseries properties"""
    ts = lazynwb.get_timeseries(local_hdf5_path, f"/processing/behavior/{table_name}", exact_path=True,  match_all=False)
    assert ts.timestamps.shape == (120,)
    assert ts.timestamps_unit == "seconds"
    assert ts.unit == "m/s"

def test_contents(local_hdf5_path):
    """Validate contents of timeseries against those obtained via pynwb"""
    test = (
        lazynwb.get_timeseries(
            local_hdf5_path,
            "/processing/behavior/running_speed_with_timestamps",
            exact_path=True,
            match_all=False,
        )
    ).data[:]
    nwb = pynwb.read_nwb(local_hdf5_path)
    reference = nwb.processing['behavior']['running_speed_with_timestamps'].data[:]
    assert test.shape == reference.shape, f"Timeseries data shape mismatch: {test.shape} vs {reference.shape}"
    assert np.array_equal(test, reference), "Timeseries data mismatch"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
