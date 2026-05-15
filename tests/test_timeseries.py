import logging
import pathlib

import numpy as np
import pynwb
import pytest

import lazynwb
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.timeseries as lazynwb_timeseries


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


def test_timeseries_native_zarr_data_accessor_reads_exact_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array_path = "processing/behavior/running_speed_with_timestamps/data"
    fallback = _FakeArray(np.asarray([1.0, 2.0, 3.0, 4.0]))
    native_reader = _FakeNativeZarrReader(
        {
            array_path: np.asarray([10.0, 20.0, 30.0, 40.0]),
        }
    )
    monkeypatch.setattr(
        lazynwb_timeseries.zarr_reader,
        "_default_zarr_backend_reader",
        lambda source: native_reader,
    )
    file = _FakeZarrFile("s3://example-bucket/test.nwb.zarr")

    accessor = lazynwb_timeseries._native_zarr_array_accessor_if_available(
        file=file,
        array_path=array_path,
        accessor=fallback,
    )
    result = accessor[slice(1, 3)]

    np.testing.assert_array_equal(result, np.asarray([20.0, 30.0]))
    assert native_reader.calls == [(array_path, slice(1, 3))]
    assert fallback.keys == []
    assert native_reader.closed


class _FakePath:
    def __init__(self, value: str) -> None:
        self._value = value

    def as_posix(self) -> str:
        return self._value


class _FakeZarrFile:
    _hdmf_backend = lazynwb.file_io.FileAccessor.HDMFBackend.ZARR

    def __init__(self, path: str) -> None:
        self._path = _FakePath(path)


class _FakeArray:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.attrs: dict[str, object] = {}
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype
        self.chunks = (2,)
        self.keys: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.keys.append(key)
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values)


class _FakeNativeZarrReader:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays
        self._remote_metadata_client = object()
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    async def read_array_selection(
        self,
        exact_array_path: str,
        selection: object = None,
    ) -> np.ndarray:
        self.calls.append((exact_array_path, selection))
        return self._arrays[exact_array_path][selection]

    async def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
