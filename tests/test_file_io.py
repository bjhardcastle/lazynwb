import pathlib
import typing
from collections.abc import Iterable

import pytest

import lazynwb.file_io


@pytest.fixture(autouse=True)
def reset_file_io_config():
    """Keep file I/O config isolated between tests."""
    original_use_polars = lazynwb.file_io.config.use_polars
    original_anon = lazynwb.file_io.config.anon
    original_storage_options = dict(lazynwb.file_io.config.fsspec_storage_options)
    try:
        yield
    finally:
        lazynwb.file_io.config.use_polars = original_use_polars
        lazynwb.file_io.config.anon = original_anon
        lazynwb.file_io.config.fsspec_storage_options = original_storage_options


def test_file_io_config_aliases_global_config() -> None:
    assert lazynwb.file_io.config is lazynwb.config


def test_config_reads_global_and_legacy_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYNWB_USE_POLARS", "true")
    monkeypatch.setenv("LAZYNWB_FILE_IO_USE_REMFILE", "false")

    config = lazynwb.Config()

    assert config.use_polars is True
    assert config.use_remfile is False


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


def test_open_file_prefers_zarr_when_zarr_word_is_in_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = object()

    def _unexpected_hdf5_open(
        *args: object,
        **kwargs: object,
    ) -> typing.NoReturn:
        raise AssertionError("HDF5 should not be opened before a Zarr-looking URI")

    def _zarr_open(path: object, mode: str) -> object:
        calls.append(f"zarr:{path}:{mode}")
        return sentinel

    monkeypatch.setattr(lazynwb.file_io, "_open_hdf5", _unexpected_hdf5_open)
    monkeypatch.setattr(lazynwb.file_io.zarr, "open", _zarr_open)

    path = "file:///tmp/zarr-benchmark.nwb"
    result = lazynwb.file_io._open_file(path)

    assert result is sentinel
    assert calls == [_expected_local_zarr_open_call(path)]


def test_open_file_prefers_zarr_for_explicit_zarr_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = object()

    def _unexpected_hdf5_open(
        *args: object,
        **kwargs: object,
    ) -> typing.NoReturn:
        raise AssertionError("HDF5 should not be opened before explicit Zarr")

    def _zarr_open(path: object, mode: str) -> object:
        calls.append(f"zarr:{path}:{mode}")
        return sentinel

    monkeypatch.setattr(lazynwb.file_io, "_open_hdf5", _unexpected_hdf5_open)
    monkeypatch.setattr(lazynwb.file_io.zarr, "open", _zarr_open)

    path = "file:///tmp/example.nwb.zarr"
    result = lazynwb.file_io._open_file(path)

    assert result is sentinel
    assert calls == [_expected_local_zarr_open_call(path)]


def test_open_file_uses_fsspec_mapper_for_remote_zarr_without_zarr_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, object]] = []
    mapper = object()
    sentinel = object()
    lazynwb.file_io.config.anon = True
    lazynwb.file_io.config.fsspec_storage_options = {
        "anon": False,
        "client": "value",
    }

    def _reject_hdf5(
        *args: object,
        **kwargs: object,
    ) -> typing.NoReturn:
        calls.append(("hdf5", args, kwargs))
        raise OSError("not a single-object HDF5 file")

    def _get_mapper(path: object, **storage_options: object) -> object:
        calls.append(("mapper", path, storage_options))
        return mapper

    def _open_consolidated(store: object, mode: str) -> object:
        calls.append(("open_consolidated", store, mode))
        return sentinel

    def _unexpected_zarr_open(*args: object, **kwargs: object) -> typing.NoReturn:
        raise AssertionError("remote Zarr stores should open through an fsspec mapper")

    monkeypatch.setattr(lazynwb.file_io, "_open_hdf5", _reject_hdf5)
    monkeypatch.setattr(lazynwb.file_io.fsspec, "get_mapper", _get_mapper)
    monkeypatch.setattr(
        lazynwb.file_io.zarr,
        "open_consolidated",
        _open_consolidated,
    )
    monkeypatch.setattr(lazynwb.file_io.zarr, "open", _unexpected_zarr_open)

    result = lazynwb.file_io._open_file("s3://bucket/example.nwb")

    assert result is sentinel
    assert len(calls) == 3
    assert calls[0][0] == "hdf5"
    hdf5_args = calls[0][1]
    assert isinstance(hdf5_args, tuple)
    assert hdf5_args[0].as_posix() == "s3://bucket/example.nwb"
    assert calls[0][2] == {"use_remfile": True, "use_obstore": False}
    assert calls[1:] == [
        ("mapper", "s3://bucket/example.nwb", {"anon": True, "client": "value"}),
        ("open_consolidated", mapper, "r"),
    ]


def test_fsspec_storage_options_use_top_level_anon() -> None:
    lazynwb.file_io.config.anon = True
    lazynwb.file_io.config.fsspec_storage_options = {"anon": False, "custom": "value"}

    assert lazynwb.file_io._get_fsspec_storage_options() == {
        "anon": True,
        "custom": "value",
    }


def test_obstore_storage_options_translate_anon_to_skip_signature() -> None:
    lazynwb.file_io.config.anon = True
    lazynwb.file_io.config.fsspec_storage_options = {
        "anon": False,
        "region": "us-west-2",
    }

    assert lazynwb.file_io._get_obstore_storage_options() == {
        "region": "us-west-2",
        "skip_signature": True,
    }


def test_obstore_storage_options_use_aws_region_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    lazynwb.file_io.config.anon = True
    lazynwb.file_io.config.fsspec_storage_options = {"anon": False}

    assert lazynwb.file_io._get_obstore_storage_options() == {
        "region": "us-west-2",
        "skip_signature": True,
    }


def test_storage_options_fall_back_to_legacy_anon_setting() -> None:
    lazynwb.file_io.config.anon = None
    lazynwb.file_io.config.fsspec_storage_options = {"anon": True}

    assert lazynwb.file_io._get_fsspec_storage_options()["anon"] is True
    assert lazynwb.file_io._get_obstore_storage_options()["skip_signature"] is True


def _expected_local_zarr_open_call(path: str) -> str:
    normalized_path = lazynwb.file_io.from_pathlike(path).as_posix()
    return f"zarr:{normalized_path}:r"


if __name__ == "__main__":
    pytest.main([__file__])
