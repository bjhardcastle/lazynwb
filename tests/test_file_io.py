import pathlib
import types
from collections.abc import Iterable
from unittest import mock

import pytest
import requests
import upath

import lazynwb.file_io


def test_open_hdf5_uses_obstore_for_s3_https_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3 HTTPS URLs are normalized and opened through obstore."""
    url = "https://test-bucket.s3.amazonaws.com/path/session.nwb"
    hdf5_file = object()
    buffered_file = object()
    fsspec_store = object()
    region_response = types.SimpleNamespace(
        headers={"x-amz-bucket-region": "us-west-2"}
    )

    monkeypatch.setattr(
        lazynwb.file_io.config,
        "fsspec_storage_options",
        {},
    )
    monkeypatch.setattr(lazynwb.file_io, "_S3_REGION_CACHE", {})
    head = mock.Mock(return_value=region_response)
    monkeypatch.setattr(requests, "head", head)
    store = mock.Mock(return_value=fsspec_store)
    monkeypatch.setattr(lazynwb.file_io.obstore.fsspec, "FsspecStore", store)
    buffered = mock.Mock(return_value=buffered_file)
    monkeypatch.setattr(lazynwb.file_io.obstore.fsspec, "BufferedFile", buffered)
    h5py_file = mock.Mock(return_value=hdf5_file)
    monkeypatch.setattr(lazynwb.file_io.h5py, "File", h5py_file)
    remfile = mock.Mock(side_effect=AssertionError("remfile should not be used"))
    monkeypatch.setattr(lazynwb.file_io.remfile, "File", remfile)

    result = lazynwb.file_io._open_hdf5(
        upath.UPath(url), use_obstore=True, use_remfile=False
    )

    assert result is hdf5_file
    head.assert_called_once_with(
        "https://s3.amazonaws.com/test-bucket",
        allow_redirects=False,
        timeout=10,
    )
    store.assert_called_once_with("s3", skip_signature=True, region="us-west-2")
    buffered.assert_called_once_with(
        fs=fsspec_store,
        path="s3://test-bucket/path/session.nwb",
    )
    h5py_file.assert_called_once_with(buffered_file, mode="r")


def test_s3_region_discovery_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated Dandiset assets share one region lookup per bucket."""
    response = types.SimpleNamespace(headers={"x-amz-bucket-region": "us-east-2"})
    head = mock.Mock(return_value=response)
    monkeypatch.setattr(requests, "head", head)
    monkeypatch.setattr(lazynwb.file_io, "_S3_REGION_CACHE", {})

    assert lazynwb.file_io._discover_s3_region("cached-test-bucket") == "us-east-2"
    assert lazynwb.file_io._discover_s3_region("cached-test-bucket") == "us-east-2"

    head.assert_called_once()


def test_open_hdf5_translates_fsspec_options_for_obstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common s3fs names are translated to their obstore equivalents."""
    url = "https://test-bucket.s3.us-east-2.amazonaws.com/path/session.nwb"
    fsspec_store = object()

    monkeypatch.setattr(
        lazynwb.file_io.config,
        "fsspec_storage_options",
        {
            "anon": False,
            "key": "access-key",
            "secret": "secret-key",
            "token": "session-token",
            "requester_pays": True,
            "client_kwargs": {
                "endpoint_url": "https://s3.us-east-2.amazonaws.com",
                "region_name": "us-east-2",
            },
        },
    )
    head = mock.Mock(side_effect=AssertionError("region is present in the URL"))
    monkeypatch.setattr(requests, "head", head)
    store = mock.Mock(return_value=fsspec_store)
    monkeypatch.setattr(lazynwb.file_io.obstore.fsspec, "FsspecStore", store)
    monkeypatch.setattr(
        lazynwb.file_io.obstore.fsspec,
        "BufferedFile",
        mock.Mock(return_value=object()),
    )
    monkeypatch.setattr(lazynwb.file_io.h5py, "File", mock.Mock(return_value=object()))

    lazynwb.file_io._open_hdf5(upath.UPath(url), use_obstore=True, use_remfile=False)

    head.assert_not_called()
    store.assert_called_once_with(
        "s3",
        skip_signature=False,
        access_key_id="access-key",
        secret_access_key="secret-key",
        session_token="session-token",
        request_payer=True,
        endpoint="https://s3.us-east-2.amazonaws.com",
        region="us-east-2",
    )


def test_open_hdf5_uses_obstore_for_generic_https_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-S3 HTTPS URLs use obstore's HTTP store with translated headers."""
    url = "https://example.com/path/session.nwb"
    fsspec_store = object()

    monkeypatch.setattr(
        lazynwb.file_io.config,
        "fsspec_storage_options",
        {
            "anon": False,
            "headers": {"Authorization": "Bearer test-token"},
        },
    )
    store = mock.Mock(return_value=fsspec_store)
    monkeypatch.setattr(lazynwb.file_io.obstore.fsspec, "FsspecStore", store)
    buffered = mock.Mock(return_value=object())
    monkeypatch.setattr(lazynwb.file_io.obstore.fsspec, "BufferedFile", buffered)
    monkeypatch.setattr(lazynwb.file_io.h5py, "File", mock.Mock(return_value=object()))
    monkeypatch.setattr(
        lazynwb.file_io.remfile,
        "File",
        mock.Mock(side_effect=AssertionError("remfile should not be used")),
    )

    lazynwb.file_io._open_hdf5(upath.UPath(url), use_obstore=True, use_remfile=False)

    store.assert_called_once_with(
        "https",
        client_options={"default_headers": {"Authorization": "Bearer test-token"}},
    )
    buffered.assert_called_once_with(fs=fsspec_store, path=url)


def test_open_hdf5_falls_back_to_remfile_when_obstore_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An obstore setup failure does not make an HTTP NWB inaccessible."""
    url = "https://example.com/path/session.nwb"
    remfile_file = object()
    hdf5_file = object()

    monkeypatch.setattr(
        lazynwb.file_io.obstore.fsspec,
        "FsspecStore",
        mock.Mock(side_effect=RuntimeError("obstore failed")),
    )
    remfile = mock.Mock(return_value=remfile_file)
    monkeypatch.setattr(lazynwb.file_io.remfile, "File", remfile)
    h5py_file = mock.Mock(return_value=hdf5_file)
    monkeypatch.setattr(lazynwb.file_io.h5py, "File", h5py_file)

    result = lazynwb.file_io._open_hdf5(
        upath.UPath(url), use_obstore=True, use_remfile=False
    )

    assert result is hdf5_file
    remfile.assert_called_once_with(url=url)
    h5py_file.assert_called_once_with(remfile_file, mode="r")


def test_open_hdf5_falls_back_when_h5py_cannot_read_obstore_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback covers h5py's initial reads as well as obstore construction."""
    url = "https://example.com/path/session.nwb"
    buffered_file = mock.Mock()
    remfile_file = object()
    hdf5_file = object()

    monkeypatch.setattr(
        lazynwb.file_io.obstore.fsspec,
        "FsspecStore",
        mock.Mock(return_value=object()),
    )
    monkeypatch.setattr(
        lazynwb.file_io.obstore.fsspec,
        "BufferedFile",
        mock.Mock(return_value=buffered_file),
    )
    remfile = mock.Mock(return_value=remfile_file)
    monkeypatch.setattr(lazynwb.file_io.remfile, "File", remfile)
    h5py_file = mock.Mock(side_effect=[OSError("bad superblock read"), hdf5_file])
    monkeypatch.setattr(lazynwb.file_io.h5py, "File", h5py_file)

    result = lazynwb.file_io._open_hdf5(
        upath.UPath(url), use_obstore=True, use_remfile=False
    )

    assert result is hdf5_file
    buffered_file.close.assert_called_once_with()
    remfile.assert_called_once_with(url=url)


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


if __name__ == "__main__":
    pytest.main([__file__])
