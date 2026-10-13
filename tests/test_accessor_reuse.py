"""Regression coverage for internal accessor reuse during metadata reads."""

import asyncio
import collections.abc
import pathlib

import pytest

import lazynwb._catalog.accessor as catalog_accessor
import lazynwb.base as base
import lazynwb.file_io as file_io
import lazynwb.types_ as types_


@pytest.fixture(autouse=True)
def _isolated_accessor_cache() -> collections.abc.Iterator[None]:
    """Keep cache ownership assertions independent of test ordering."""
    file_io._clear_accessor_cache()
    yield
    file_io._clear_accessor_cache()


def test_metadata_properties_reuse_one_open_handle(
    local_hdf5_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session and subject properties must not reopen a file for every field."""
    opens = []
    original = file_io._open_file

    def _counted_open(path: types_.PathLike) -> object:
        """Count real backend opens while preserving fixture reads."""
        opens.append(path)
        return original(path)

    monkeypatch.setattr(file_io, "_open_file", _counted_open)
    nwb = base.LazyNWB(local_hdf5_path)
    assert nwb._to_dict()["session_description"]
    assert nwb.subject._to_dict()["subject_id"]
    assert len(opens) == 1


def test_internal_accessor_reopens_closed_handle(local_hdf5_path: pathlib.Path) -> None:
    """Closing a borrowed accessor must not poison subsequent lookups."""
    first = file_io._get_accessor(local_hdf5_path)
    first._close()
    second = file_io._get_accessor(local_hdf5_path)
    assert second is not first
    assert "units" in second


def test_internal_accessor_respects_disabled_cache(
    local_hdf5_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled cache must bypass even an already cached handle."""
    first = file_io._get_accessor(local_hdf5_path)
    monkeypatch.setattr(file_io.config, "disable_cache", True)
    second = file_io._get_accessor(local_hdf5_path)
    assert first is not second
    second._close()


def test_catalog_close_does_not_close_borrowed_dataset(
    local_hdf5_path: pathlib.Path,
) -> None:
    """Schema readers must not invalidate datasets retained by other callers."""
    borrowed = file_io._get_accessor(local_hdf5_path)
    dataset = borrowed["units/id"]
    reader = catalog_accessor._AccessorBackendReader(local_hdf5_path)
    asyncio.run(reader.close())
    assert dataset[0] >= 0


def test_eviction_and_clear_leave_borrowed_handles_open(
    local_hdf5_paths: list[pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound cached ownership without closing accessors still in use."""
    monkeypatch.setattr(file_io, "_ACCESSOR_CACHE_LIMIT", 1)
    first = file_io._get_accessor(local_hdf5_paths[0])
    second = file_io._get_accessor(local_hdf5_paths[1])
    assert len(file_io._accessor_cache) == 1
    assert "units" in first
    file_io._clear_accessor_cache()
    assert "units" in second
    second._close()


def test_internal_accessor_separates_storage_configuration(
    local_hdf5_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing authentication settings must not reuse the previous handle."""
    first = file_io._get_accessor(local_hdf5_path)
    monkeypatch.setattr(file_io.config, "anon", not file_io.config.anon)
    second = file_io._get_accessor(local_hdf5_path)
    assert first is not second
