import logging
import pathlib

import numpy as np
import pynwb
import pytest

import lazynwb
import lazynwb.exceptions


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_sources(nwb_fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Test get_timeseries with various NWB file inputs."""
    # Resolve the fixture name to its value (the path to a single NWB file)
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)
    _ = lazynwb.get_timeseries(
        nwb_path_or_paths, "/processing/behavior/", exact_path=True
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "running_speed_with_timestamps",
        "running_speed_with_rate",
    ],
)
def test_properties(local_hdf5_path: pathlib.Path, table_name: str) -> None:
    """Test get_timeseries properties"""
    ts = lazynwb.get_timeseries(
        local_hdf5_path,
        f"/processing/behavior/{table_name}",
        exact_path=True,
        match_all=False,
    )
    assert len(ts.timestamps) > 0, "timestamps should not be empty"
    assert len(ts.timestamps.shape) == 1, "timestamps should be exploded to 1D"
    assert ts.timestamps_unit == "seconds"
    assert ts.unit == "m/s"


def test_contents(local_hdf5_path: pathlib.Path) -> None:
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
    reference = nwb.processing["behavior"]["running_speed_with_timestamps"].data[:]
    assert (
        test.shape == reference.shape
    ), f"Timeseries data shape mismatch: {test.shape} vs {reference.shape}"
    assert np.array_equal(test, reference), "Timeseries data mismatch"


def test_getattr_missing_attribute(local_hdf5_path: pathlib.Path) -> None:
    """Test that accessing a nonexistent attribute raises AttributeError."""
    ts = lazynwb.get_timeseries(
        local_hdf5_path,
        "/processing/behavior/running_speed_with_timestamps",
        exact_path=True,
    )
    with pytest.raises(AttributeError):
        _ = ts.nonexistent_attribute


def test_getattr_private_attribute(local_hdf5_path: pathlib.Path) -> None:
    """Test that accessing an undefined private attribute raises AttributeError immediately."""
    ts = lazynwb.get_timeseries(
        local_hdf5_path,
        "/processing/behavior/running_speed_with_timestamps",
        exact_path=True,
    )
    with pytest.raises(AttributeError):
        _ = ts._nonexistent


def test_get_timeseries_missing_search_term_has_clear_error(
    local_hdf5_path: pathlib.Path,
) -> None:
    with pytest.raises(
        lazynwb.exceptions.InternalPathError,
        match="No TimeSeries matching",
    ):
        lazynwb.get_timeseries(local_hdf5_path, "definitely_missing_timeseries")


def test_get_timeseries_ambiguous_search_warns(
    local_hdf5_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="lazynwb.timeseries")

    ts = lazynwb.get_timeseries(local_hdf5_path, "running_speed")

    assert isinstance(ts, lazynwb.TimeSeries)
    assert "Found multiple timeseries matching 'running_speed'" in caplog.text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
