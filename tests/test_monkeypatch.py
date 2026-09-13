from __future__ import annotations

import pathlib
import uuid
from datetime import datetime, timezone

import polars as pl
import pynwb
import pytest

import lazynwb.file_io
import lazynwb.lazyframe
import lazynwb.monkeypatch


def test_nwbfile_to_lazyframe_uses_container_source(local_hdf5_path: pathlib.Path) -> None:
    with pynwb.NWBHDF5IO(str(local_hdf5_path), "r") as io:
        nwbfile = io.read()

        lazyframe = nwbfile.to_lazyframe("/units", disable_progress=True)

        assert isinstance(lazyframe, pl.LazyFrame)
        assert not lazyframe.collect().is_empty()


def test_dynamic_table_to_lazyframe_infers_table_path(local_hdf5_path: pathlib.Path) -> None:
    with pynwb.NWBHDF5IO(str(local_hdf5_path), "r") as io:
        nwbfile = io.read()

        lazyframe = nwbfile.trials.to_lazyframe(disable_progress=True)

        assert isinstance(lazyframe, pl.LazyFrame)
        assert not lazyframe.collect().is_empty()


def test_dynamic_table_to_lazyframe_prefers_lazynwb_accessor_source(
    local_hdf5_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str]] = []

    def _scan_nwb(source: object, table_path: str, **scan_kwargs: object) -> pl.LazyFrame:
        calls.append((source, table_path))
        return pl.DataFrame({"x": [1]}).lazy()

    monkeypatch.setattr(lazynwb.lazyframe, "scan_nwb", _scan_nwb)

    try:
        accessor = lazynwb.file_io.FileAccessor(local_hdf5_path)
        io = pynwb.NWBHDF5IO(file=accessor._accessor, load_namespaces=True)
        nwbfile = io.read()

        lazyframe = nwbfile.units.to_lazyframe(disable_progress=True)

        assert isinstance(lazyframe, pl.LazyFrame)
        assert calls == [(lazynwb.file_io.from_pathlike(local_hdf5_path).as_posix(), "/units")]
    finally:
        lazynwb.file_io.clear_cache()


def test_container_source_prefers_lazynwb_read_io_source() -> None:
    class _FakeFile:
        pass

    class _FakeReadIO:
        def __init__(self, file: object) -> None:
            self._file = file

    class _FakeContainer:
        container_source = "<remfile.RemFile.RemFile object at 0x123>"

        def __init__(self, read_io: object) -> None:
            self._read_io = read_io

        def get_read_io(self) -> object:
            return self._read_io

    fake_file = _FakeFile()
    source = "s3://bucket/subdir/file.nwb"
    setattr(fake_file, lazynwb.file_io._LAZYNWB_SOURCE_PATH_ATTR, source)

    assert (
        lazynwb.monkeypatch._get_container_source(_FakeContainer(_FakeReadIO(fake_file)))
        == source
    )


def test_to_lazyframe_raises_when_container_source_is_missing() -> None:
    nwbfile = pynwb.NWBFile(
        session_description="not written",
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.now(tz=timezone.utc),
    )

    with pytest.raises(ValueError, match="Could not find the NWB file path"):
        nwbfile.to_lazyframe("/units", disable_progress=True)


def test_patch_is_idempotent() -> None:
    lazynwb.monkeypatch.patch()
    lazynwb.monkeypatch.patch()

    assert vars(pynwb.NWBFile.to_lazyframe)["_lazynwb_monkeypatch"] is True
