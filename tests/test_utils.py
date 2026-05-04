import logging
import pathlib

import h5py
import pytest

import lazynwb

expected_paths = (
    "/processing/behavior/running_speed_with_timestamps/data",
    "/processing/behavior/running_speed_with_timestamps/timestamps",
    "/processing/behavior/running_speed_with_rate/data",
    "/units",
    "/intervals/trials",
    "/intervals/epochs",
)  # excluding metadata, specifications and table columns


def test_get_nwb_file_structure_hdf5(local_hdf5_path):
    """Test get_nwb_file_structure with HDF5 file."""
    structure = lazynwb.get_internal_paths(local_hdf5_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    assert (
        extras := set(structure.keys()) - set(expected_paths)
    ) == set(), f"Additional unexpected paths found: {extras}"

    # Check that metadata is available without returning live accessors.
    units_metadata = structure["/units"]
    assert units_metadata["is_group"] is True
    assert units_metadata["is_dataset"] is False
    assert isinstance(units_metadata["attrs"], dict)


def test_get_nwb_file_structure_zarr(local_zarr_path):
    """Test get_nwb_file_structure with Zarr file."""
    structure = lazynwb.get_internal_paths(local_zarr_path)

    for path in expected_paths:
        assert path in structure, f"Expected path {path} not found in structure"
    assert (
        extras := set(structure.keys()) - set(expected_paths)
    ) == set(), f"Additional unexpected paths found: {extras}"

    # Check that metadata is available without returning live accessors.
    units_metadata = structure["/units"]
    assert units_metadata["is_group"] is True
    assert units_metadata["is_dataset"] is False
    assert isinstance(units_metadata["attrs"], dict)


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


def test_get_internal_paths_warns_for_remote_hdf5_accessor_traversal(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdf5_path = tmp_path / "remote-shaped.nwb"
    with h5py.File(hdf5_path, "w") as h5_file:
        h5_file.create_group("units").attrs["neurodata_type"] = "Units"

    h5_file = h5py.File(hdf5_path, "r")

    class _FakePath:
        protocol = "s3"

        def as_posix(self) -> str:
            return "s3://bucket/remote-shaped.nwb"

    class _FakeAccessor:
        _accessor = h5_file
        _hdmf_backend = lazynwb.file_io.FileAccessor.HDMFBackend.HDF5
        _path = _FakePath()

    monkeypatch.setattr(
        lazynwb.file_io,
        "_get_catalog_path_summary_if_available",
        lambda **_: None,
    )
    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", lambda _: _FakeAccessor())

    try:
        with pytest.warns(RuntimeWarning, match="remote HDF5"):
            structure = lazynwb.get_internal_paths("s3://bucket/remote-shaped.nwb")
    finally:
        h5_file.close()

    assert "/units" in structure
    assert structure["/units"]["is_group"] is True


def test_get_internal_paths_uses_hdf5_catalog_cache(
    local_hdf5_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(
        "LAZYNWB_CATALOG_CACHE_PATH",
        str(tmp_path / "catalog.sqlite"),
    )
    caplog.set_level(logging.DEBUG, logger="lazynwb._hdf5.reader")

    cold_structure = lazynwb.get_internal_paths(
        local_hdf5_path,
        include_metadata=True,
        parents=True,
    )

    assert cold_structure["/units"]["is_group"] is True
    assert isinstance(cold_structure["/units"]["attrs"], dict)

    def _fail_accessor(*args: object, **kwargs: object) -> None:
        raise AssertionError("get_internal_paths should use catalog-backed metadata")

    monkeypatch.setattr(lazynwb.file_io, "_get_accessor", _fail_accessor)
    caplog.clear()

    warm_structure = lazynwb.get_internal_paths(
        local_hdf5_path,
        include_metadata=True,
        parents=True,
    )

    assert warm_structure.keys() == cold_structure.keys()
    assert "parsed HDF5 metadata cache lookup" in caplog.text
    assert ": hit" in caplog.text


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
