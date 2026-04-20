import pathlib
from collections.abc import Iterable

import pytest

import lazynwb.file_io


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_zarr_path",
    ],
)
def test_file_accessor(nwb_fixture_name, request):
    """Test FileAccessor with various NWB file/store inputs."""
    path = request.getfixturevalue(nwb_fixture_name)
    accessor = lazynwb.file_io._get_accessor(path)
    assert isinstance(accessor, lazynwb.file_io.FileAccessor)
    assert "units" in accessor, "__contains__() failing, or NWB fixture has changed"
    assert "/units" in accessor, "__contains__() failling to normalize path"
    assert accessor.get("units") is not None, "get() should return an object"
    assert (
        next(iter(accessor), None) is not None
    ), "Accessor should be iterable and yield at least one item"


def test_file_accessor_caching(local_hdf5_path: pathlib.Path) -> None:
    """Test that FileAccessor instances are cached and reused."""
    file_path = local_hdf5_path

    # Initial access
    accessor1 = lazynwb.file_io.FileAccessor(file_path)
    accessor1_id = id(accessor1)

    # Access again, should return the same instance
    accessor2 = lazynwb.file_io.FileAccessor(file_path)
    accessor2_id = id(accessor2)

    assert accessor1_id == accessor2_id


def test_file_accessor_reinstantiation_after_close(
    local_hdf5_path: pathlib.Path,
) -> None:
    """Test that FileAccessor can be reinstantiated after the underlying HDF5 file is closed."""
    file_path = local_hdf5_path

    # Initial access
    accessor1 = lazynwb.file_io.FileAccessor(file_path)
    accessor1_id = id(accessor1)

    # Verify it's working initially
    assert "units" in accessor1
    assert accessor1._hdmf_backend == lazynwb.file_io.FileAccessor.HDMFBackend.HDF5

    # Close the underlying HDF5 file
    accessor1._accessor.close()

    # Verify the file is closed
    assert not bool(accessor1._accessor)

    # Access again - should detect stale cache and return same instance with new accessor
    accessor2 = lazynwb.file_io.FileAccessor(file_path)
    accessor2_id = id(accessor2)

    # Should be the same cached instance
    assert accessor1_id == accessor2_id

    # But should have a fresh, working accessor
    assert bool(accessor2._accessor)
    assert "units" in accessor2
    assert accessor2._hdmf_backend == lazynwb.file_io.FileAccessor.HDMFBackend.HDF5


def test_file_accessor_clearing(local_hdf5_path: pathlib.Path) -> None:
    """Test that FileAccessor cache can be cleared."""
    file_path = local_hdf5_path

    # Initial access
    accessor1 = lazynwb.file_io.FileAccessor(file_path)
    accessor1_id = id(accessor1)

    # Clear the cache
    lazynwb.file_io.clear_cache()

    # Access again, should return a new instance
    accessor2 = lazynwb.file_io.FileAccessor(file_path)
    accessor2_id = id(accessor2)

    assert accessor1_id != accessor2_id


def test_open_single_and_multiple(local_hdf5_paths: list[pathlib.Path]) -> None:
    """Test lazynwb.file_io.open with single and multiple paths.

    Ensures correct return type and access for both single and iterable input.
    """

    # Single path
    accessor = lazynwb.file_io._get_accessor(local_hdf5_paths[0])
    assert isinstance(accessor, lazynwb.file_io.FileAccessor)
    assert "units" in accessor

    # Multiple paths
    accessors = lazynwb.file_io._get_accessors(local_hdf5_paths)
    assert isinstance(accessors, Iterable)
    assert all(isinstance(a, lazynwb.file_io.FileAccessor) for a in accessors)
    assert "units" in accessors[0]
    assert "units" in accessors[1]


def test_config_anon_translates_for_fsspec_and_obstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_anon = lazynwb.file_io.config.anon
    original_storage_options = dict(lazynwb.file_io.config.fsspec_storage_options)
    try:
        lazynwb.file_io.config.anon = True
        lazynwb.file_io.config.fsspec_storage_options = {}
        path = lazynwb.file_io.from_pathlike("s3://test-bucket/example.nwb")

        assert lazynwb.file_io._get_fsspec_storage_options(path)["anon"] is True

        monkeypatch.setattr(
            lazynwb.file_io,
            "_infer_s3_bucket_region",
            lambda _path: "us-west-2",
        )
        obstore_options = lazynwb.file_io._get_obstore_storage_options(path)
        assert obstore_options["skip_signature"] is True
        assert obstore_options["region"] == "us-west-2"
        assert "anon" not in obstore_options
    finally:
        lazynwb.file_io.config.anon = original_anon
        lazynwb.file_io.config.fsspec_storage_options = original_storage_options


def test_legacy_fsspec_anon_option_still_translates_for_obstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_anon = lazynwb.file_io.config.anon
    original_storage_options = dict(lazynwb.file_io.config.fsspec_storage_options)
    try:
        lazynwb.file_io.config.anon = None
        lazynwb.file_io.config.fsspec_storage_options = {"anon": True}
        path = lazynwb.file_io.from_pathlike("s3://test-bucket/example.nwb")

        monkeypatch.setattr(
            lazynwb.file_io,
            "_infer_s3_bucket_region",
            lambda _path: "us-west-2",
        )
        obstore_options = lazynwb.file_io._get_obstore_storage_options(path)
        assert obstore_options["skip_signature"] is True
        assert obstore_options["region"] == "us-west-2"
        assert "anon" not in obstore_options
    finally:
        lazynwb.file_io.config.anon = original_anon
        lazynwb.file_io.config.fsspec_storage_options = original_storage_options


if __name__ == "__main__":
    pytest.main([__file__])
