from __future__ import annotations

import asyncio
import datetime
import pathlib

import pytest

import lazynwb._hdf5.range_reader as hdf5_range_reader


def test_buffer_range_reader_raises_on_short_response() -> None:
    reader = hdf5_range_reader._BufferRangeReader(b"abc")

    with pytest.raises(hdf5_range_reader._RangeReadError, match="short range"):
        asyncio.run(reader.read_range(1, length=8))


def test_buffer_range_reader_coalesces_aligned_ranges() -> None:
    reader = hdf5_range_reader._BufferRangeReader(
        bytes(range(64)),
        config=hdf5_range_reader._RangeReaderConfig(range_alignment=16),
    )
    ranges = (
        hdf5_range_reader._ByteRange(1, 4),
        hdf5_range_reader._ByteRange(14, 16),
        hdf5_range_reader._ByteRange(33, 35),
    )

    payloads = asyncio.run(reader.read_ranges(ranges))

    assert payloads[hdf5_range_reader._ByteRange(1, 4)] == bytes([1, 2, 3])
    assert payloads[hdf5_range_reader._ByteRange(14, 16)] == bytes([14, 15])
    assert payloads[hdf5_range_reader._ByteRange(33, 35)] == bytes([33, 34])
    assert reader.request_count == 2
    assert reader.bytes_fetched == 32


def test_source_identity_from_obstore_metadata() -> None:
    metadata = {
        "size": 123,
        "version": "version-1",
        "e_tag": "etag-1",
        "last_modified": datetime.datetime(
            2026,
            5,
            3,
            tzinfo=datetime.timezone.utc,
        ),
    }

    identity = hdf5_range_reader._source_identity_from_metadata(
        "s3://bucket/file.nwb",
        metadata,
    )

    assert identity.source_url == "s3://bucket/file.nwb"
    assert identity.content_length == 123
    assert identity.version_id == "version-1"
    assert identity.etag == "etag-1"
    assert identity.validator_kind == "version_id"


def test_s3_region_discovery_adds_region_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hdf5_range_reader,
        "_discover_s3_bucket_region",
        lambda bucket: "us-west-2",
    )

    options = hdf5_range_reader._add_discovered_s3_region(
        bucket="aind-scratch-data",
        storage_options={"skip_signature": True},
    )

    assert options == {"skip_signature": True, "region": "us-west-2"}


def test_s3_region_discovery_preserves_explicit_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hdf5_range_reader,
        "_discover_s3_bucket_region",
        lambda bucket: "us-west-2",
    )

    options = hdf5_range_reader._add_discovered_s3_region(
        bucket="aind-scratch-data",
        storage_options={"skip_signature": True, "region": "eu-west-1"},
    )

    assert options == {"skip_signature": True, "region": "eu-west-1"}


def test_obstore_store_cache_reuses_store_for_same_bucket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = [object()]
    calls: list[tuple[str, dict[str, object]]] = []
    hdf5_range_reader._OBSTORE_STORE_CACHE.clear()

    def _fake_from_url(store_url: str, **kwargs: object) -> object:
        calls.append((store_url, kwargs))
        return stores[0]

    monkeypatch.setattr(hdf5_range_reader.obstore.store, "from_url", _fake_from_url)
    config = hdf5_range_reader._RangeReaderConfig(
        storage_options={"region": "us-west-2", "skip_signature": True},
    )

    try:
        first_store, first_path = hdf5_range_reader._store_and_path_from_url(
            "s3://aind-scratch-data/first.nwb",
            config,
        )
        second_store, second_path = hdf5_range_reader._store_and_path_from_url(
            "s3://aind-scratch-data/second.nwb",
            config,
        )

        assert first_store is second_store
        assert first_path == "first.nwb"
        assert second_path == "second.nwb"
        assert [call[0] for call in calls] == ["s3://aind-scratch-data"]
    finally:
        hdf5_range_reader._OBSTORE_STORE_CACHE.clear()


def test_probe_hdf5_signature_at_zero() -> None:
    reader = hdf5_range_reader._BufferRangeReader(
        hdf5_range_reader._HDF5_SIGNATURE + b"\x00" * 32
    )

    result = asyncio.run(hdf5_range_reader._probe_hdf5_signature(reader))

    assert result.is_hdf5
    assert result.signature_offset == 0
    assert result.checked_offsets == (0,)


def test_probe_hdf5_signature_at_later_valid_superblock_offset() -> None:
    reader = hdf5_range_reader._BufferRangeReader(
        (b"\x00" * 512) + hdf5_range_reader._HDF5_SIGNATURE + b"\x00" * 32
    )

    result = asyncio.run(hdf5_range_reader._probe_hdf5_signature(reader))

    assert result.is_hdf5
    assert result.signature_offset == 512
    assert result.checked_offsets == (0, 512)


def test_probe_hdf5_signature_rejects_non_hdf5_content() -> None:
    reader = hdf5_range_reader._BufferRangeReader(b"not an hdf5 file" + b"\x00" * 600)

    result = asyncio.run(
        hdf5_range_reader._probe_hdf5_signature(reader, max_probe_offset=512)
    )

    assert not result.is_hdf5
    assert result.signature_offset is None
    assert result.checked_offsets == (0, 512)


def test_obstore_range_reader_reads_local_file_url(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "object.nwb"
    path.write_bytes(b"0123456789abcdef")
    reader = hdf5_range_reader._ObstoreRangeReader(path.as_uri())

    identity = asyncio.run(reader.get_source_identity())
    payload = asyncio.run(reader.read_range(2, length=4))

    assert identity.source_url == path.as_uri()
    assert identity.content_length == 16
    assert identity.validator_kind in {"etag", "last_modified_content_length"}
    assert payload == b"2345"
