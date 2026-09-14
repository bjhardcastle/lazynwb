import logging
import pathlib

import h5py
import numpy as np
import pynwb
import pytest

import lazynwb
import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.timeseries


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


def test_hdf5_timeseries_dataset_properties_and_slices_are_range_backed(
    local_hdf5_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    timeseries_path = "/processing/behavior/running_speed_with_timestamps"
    with h5py.File(local_hdf5_path, "r") as h5_file:
        expected = h5_file[f"{timeseries_path}/data"][:]

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("HDF5 TimeSeries reads must not instantiate FileAccessor")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)
    caplog.set_level(logging.DEBUG, logger="lazynwb.timeseries")
    timeseries = lazynwb.get_timeseries(
        local_hdf5_path.as_uri(),
        timeseries_path,
        exact_path=True,
    )

    data = timeseries.data
    assert isinstance(data, lazynwb.timeseries._RangeBackedHDF5Dataset)
    assert data.shape == expected.shape
    assert data.ndim == expected.ndim
    assert data.size == expected.size
    assert data.dtype == expected.dtype
    assert data.name == f"{timeseries_path}/data"
    assert data.attrs["unit"] == "m/s"
    assert timeseries.unit == "m/s"
    assert timeseries.conversion == 1.0
    assert timeseries.offset == 0.0
    assert timeseries.resolution == -1.0
    assert timeseries.timestamps_unit == "seconds"
    assert timeseries.description == "forward running speed on wheel"
    assert data[5:11].tolist() == expected[5:11].tolist()
    assert data[-1] == expected[-1]
    assert data[[7, 1, 7]].tolist() == expected[[7, 1, 7]].tolist()
    assert np.array_equal(np.asarray(data), expected)
    assert "range-backed TimeSeries dataset read" in caplog.text


def test_rate_based_timeseries_timestamps_are_lazy_and_bounded(
    local_hdf5_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    timeseries = lazynwb.get_timeseries(
        local_hdf5_path.as_uri(),
        "/processing/behavior/running_speed_with_rate",
        exact_path=True,
    )
    arange_calls: list[tuple[object, ...]] = []
    original_arange = np.arange

    def _record_arange(*args: object, **kwargs: object) -> np.ndarray:
        arange_calls.append(args)
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(lazynwb.timeseries.np, "arange", _record_arange)
    caplog.set_level(logging.DEBUG, logger="lazynwb.timeseries")

    timestamps = timeseries.timestamps

    assert isinstance(timestamps, lazynwb.timeseries._RateBasedTimestamps)
    assert timestamps.shape == (len(timeseries.data),)
    assert timestamps.dtype == np.dtype(np.float64)
    assert arange_calls == []
    assert timestamps[2:5].tolist() == pytest.approx(
        [
            timestamps._starting_time + 2 / timeseries.rate,
            timestamps._starting_time + 3 / timeseries.rate,
            timestamps._starting_time + 4 / timeseries.rate,
        ]
    )
    assert arange_calls == []
    assert "selected_samples=3" in caplog.text


def test_chunked_multidimensional_timeseries_and_child_dataset_are_range_backed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nwb_path = tmp_path / "chunked-timeseries.nwb"
    data_values = np.arange(120, dtype=np.int32).reshape(20, 6)
    quality_values = np.linspace(0.0, 1.0, 20)
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("acquisition/recording")
        group.attrs["neurodata_type"] = "TimeSeries"
        data = group.create_dataset(
            "data",
            data=data_values,
            chunks=(4, 3),
            compression="gzip",
            shuffle=True,
        )
        data.attrs["unit"] = "volts"
        group.create_dataset("timestamps", data=np.arange(20, dtype=np.float64))
        group.create_dataset("quality", data=quality_values)
        group.create_dataset(
            "labels",
            data=np.asarray(["good", "bad"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("range-backed TimeSeries child reads must avoid FileAccessor")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)
    timeseries = lazynwb.get_timeseries(
        nwb_path.as_uri(),
        "/acquisition/recording",
        exact_path=True,
    )

    assert timeseries.data[3, 1:5].tolist() == data_values[3, 1:5].tolist()
    assert timeseries.data[2:8:2, 2:4].tolist() == data_values[2:8:2, 2:4].tolist()
    assert timeseries.data[[7, 1, 7], 2].tolist() == data_values[[7, 1, 7], 2].tolist()
    assert timeseries.quality[4:9].tolist() == quality_values[4:9].tolist()
    assert h5py.check_string_dtype(timeseries.labels.dtype).encoding == "utf-8"
    assert timeseries.labels[:].tolist() == [b"good", b"bad"]


def test_sparse_timeseries_slice_reads_only_selected_contiguous_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nwb_path = tmp_path / "large-timeseries.nwb"
    row_count = 100_000
    values = np.arange(row_count, dtype=np.float64)
    with h5py.File(nwb_path, "w") as h5_file:
        group = h5_file.create_group("acquisition/recording")
        group.attrs["neurodata_type"] = "TimeSeries"
        data = group.create_dataset("data", data=values)
        data.attrs["unit"] = "volts"
        group.create_dataset("timestamps", data=values / 1_000.0)

    file_bytes = nwb_path.read_bytes()
    identity = catalog_models._SourceIdentity(
        source_url=nwb_path.as_uri(),
        content_length=len(file_bytes),
        version_id="large-timeseries-v1",
    )
    readers: list[hdf5_reader._HDF5BackendReader] = []

    def _reader_factory(
        source: object,
        *,
        resolve_vlen_attributes: bool = False,
    ) -> hdf5_reader._HDF5BackendReader:
        range_reader = hdf5_range_reader._BufferRangeReader(
            file_bytes,
            source_identity=identity,
            config=hdf5_range_reader._RangeReaderConfig(
                range_alignment=1,
                coalesce_gap_bytes=0,
            ),
        )
        reader = hdf5_reader._HDF5BackendReader(
            str(source),
            range_reader=range_reader,
            cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
            resolve_vlen_attributes=resolve_vlen_attributes,
        )
        readers.append(reader)
        return reader

    monkeypatch.setattr(
        hdf5_reader,
        "_default_hdf5_backend_reader",
        _reader_factory,
    )
    timeseries = lazynwb.get_timeseries(
        nwb_path.as_uri(),
        "/acquisition/recording",
        exact_path=True,
    )

    selected = timeseries.data[row_count - 1]

    data_reader = readers[-1]._range_reader
    assert selected == values[-1]
    assert data_reader.request_count == 1
    assert data_reader.bytes_fetched == values.dtype.itemsize


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
