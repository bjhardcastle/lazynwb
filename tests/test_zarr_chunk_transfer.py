from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence

import numcodecs
import numcodecs.registry as numcodecs_registry
import numpy as np
import pytest

import lazynwb._zarr.chunk_transfer as chunk_transfer


def test_reads_uncompressed_1d_selection_across_chunks_with_read_many(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="lazynwb._zarr.chunk_transfer")
    zarray = _zarray(shape=(10,), chunks=(4,), dtype="<i4", compressor=None)
    objects = {
        f"data/{chunk_coord}": _encode_chunk(
            np.arange(chunk_coord * 4, chunk_coord * 4 + 4, dtype="<i4"),
            zarray=zarray,
        )
        for chunk_coord in range(3)
    }
    client = _FakeManyClient(objects)

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="/data",
            zarray=zarray,
            selection=slice(3, 9),
        )
    )

    np.testing.assert_array_equal(result, np.arange(3, 9, dtype="<i4"))
    assert client.read_many_calls == [("data/0", "data/1", "data/2")]
    assert "chunk_count=3" in caplog.text
    assert "fetched_bytes=48" in caplog.text
    assert "decode=" in caplog.text
    assert "assemble=" in caplog.text


def test_reads_blosc_compressed_2d_bounded_selection() -> None:
    compressor = numcodecs.Blosc(
        cname="lz4",
        clevel=1,
        shuffle=numcodecs.Blosc.NOSHUFFLE,
    )
    zarray = _zarray(
        shape=(5, 7),
        chunks=(2, 3),
        dtype="<i4",
        compressor=compressor.get_config(),
        fill_value=-1,
    )
    source = np.arange(35, dtype="<i4").reshape(5, 7)
    objects = _chunk_objects_from_array("matrix", source, zarray)
    client = _FakeManyClient(objects)

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="matrix",
            zarray=zarray,
            selection=(slice(1, 5), slice(2, 6)),
        )
    )

    np.testing.assert_array_equal(result, source[1:5, 2:6])
    assert client.read_many_calls == [
        (
            "matrix/0.0",
            "matrix/0.1",
            "matrix/1.0",
            "matrix/1.1",
            "matrix/2.0",
            "matrix/2.1",
        )
    ]


def test_fills_missing_chunks_and_skips_fetching_known_missing_keys() -> None:
    zarray = _zarray(shape=(6,), chunks=(2,), dtype="<i4", compressor=None, fill_value=-7)
    objects = {
        "values/0": _encode_chunk(np.array([0, 1], dtype="<i4"), zarray=zarray),
        "values/2": _encode_chunk(np.array([4, 5], dtype="<i4"), zarray=zarray),
    }
    client = _FakeManyClient(objects)

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="values",
            zarray=zarray,
            available_chunk_keys=frozenset(objects),
        )
    )

    np.testing.assert_array_equal(result, np.array([0, 1, -7, -7, 4, 5], dtype="<i4"))
    assert client.read_many_calls == [("values/0", "values/2")]


def test_read_object_fallback_supports_integer_axis_selection() -> None:
    zarray = _zarray(shape=(4, 5), chunks=(2, 3), dtype="<i4", compressor=None)
    source = np.arange(20, dtype="<i4").reshape(4, 5)
    objects = _chunk_objects_from_array("matrix", source, zarray)
    client = _FakeObjectClient(objects)

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="matrix",
            zarray=zarray,
            selection=(2, slice(1, 5)),
        )
    )

    np.testing.assert_array_equal(result, source[2, 1:5])
    assert client.read_object_calls == ["matrix/1.0", "matrix/1.1"]


def test_applies_numcodecs_filters_before_assembly() -> None:
    dtype = np.dtype("<i4")
    filter_codec = numcodecs.Delta(dtype=dtype)
    zarray = _zarray(
        shape=(6,),
        chunks=(3,),
        dtype=dtype.str,
        compressor=None,
        filters=(filter_codec.get_config(),),
    )
    objects = {
        "values/0": _encode_chunk(np.array([0, 1, 2], dtype=dtype), zarray=zarray),
        "values/1": _encode_chunk(np.array([3, 4, 5], dtype=dtype), zarray=zarray),
    }
    client = _FakeManyClient(objects)

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="values",
            zarray=zarray,
        )
    )

    np.testing.assert_array_equal(result, np.arange(6, dtype=dtype))


def test_missing_payload_from_client_uses_zarr_default_zero_fill() -> None:
    zarray = _zarray(shape=(4,), chunks=(2,), dtype="<i4", compressor=None)
    client = _FakeManyClient(
        {
            "values/0": _encode_chunk(np.array([10, 11], dtype="<i4"), zarray=zarray),
        }
    )

    result = asyncio.run(
        chunk_transfer._read_zarr_v2_array_selection(
            client,
            array_path="values",
            zarray=zarray,
        )
    )

    np.testing.assert_array_equal(result, np.array([10, 11, 0, 0], dtype="<i4"))


class _FakeManyClient:
    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self._objects = dict(objects)
        self.read_many_calls: list[tuple[str, ...]] = []

    async def read_many(self, keys: Sequence[str]) -> Mapping[str, bytes]:
        self.read_many_calls.append(tuple(keys))
        return {key: self._objects[key] for key in keys if key in self._objects}


class _FakeObjectClient:
    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self._objects = dict(objects)
        self.read_object_calls: list[str] = []

    async def read_object(self, key: str) -> bytes:
        self.read_object_calls.append(key)
        try:
            return self._objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc


def _zarray(
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    compressor: Mapping[str, object] | None,
    fill_value: object | None = None,
    filters: Sequence[Mapping[str, object]] | None = None,
    order: str = "C",
    dimension_separator: str = ".",
) -> dict[str, object]:
    return {
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": compressor,
        "fill_value": fill_value,
        "filters": None if filters is None else list(filters),
        "order": order,
        "dimension_separator": dimension_separator,
    }


def _chunk_objects_from_array(
    array_path: str,
    source: np.ndarray,
    zarray: Mapping[str, object],
) -> dict[str, bytes]:
    chunks = tuple(int(value) for value in zarray["chunks"])
    dimension_separator = str(zarray.get("dimension_separator", "."))
    objects: dict[str, bytes] = {}
    for row_coord in range(_chunk_count(source.shape[0], chunks[0])):
        for column_coord in range(_chunk_count(source.shape[1], chunks[1])):
            chunk = _chunk_from_array(source, (row_coord, column_coord), chunks, zarray)
            chunk_name = dimension_separator.join((str(row_coord), str(column_coord)))
            objects[f"{array_path}/{chunk_name}"] = _encode_chunk(chunk, zarray=zarray)
    return objects


def _chunk_from_array(
    source: np.ndarray,
    chunk_coords: tuple[int, int],
    chunks: tuple[int, int],
    zarray: Mapping[str, object],
) -> np.ndarray:
    fill_value = zarray.get("fill_value")
    dtype = np.dtype(str(zarray["dtype"]))
    chunk = np.full(chunks, 0 if fill_value is None else fill_value, dtype=dtype)
    row_start = chunk_coords[0] * chunks[0]
    column_start = chunk_coords[1] * chunks[1]
    row_stop = min(row_start + chunks[0], source.shape[0])
    column_stop = min(column_start + chunks[1], source.shape[1])
    row_count = row_stop - row_start
    column_count = column_stop - column_start
    chunk[:row_count, :column_count] = source[row_start:row_stop, column_start:column_stop]
    return chunk


def _chunk_count(axis_size: int, chunk_size: int) -> int:
    return (axis_size + chunk_size - 1) // chunk_size


def _encode_chunk(chunk: np.ndarray, *, zarray: Mapping[str, object]) -> bytes:
    encoded: object = np.asarray(chunk, order=str(zarray.get("order", "C")))
    filters = zarray.get("filters")
    if filters is not None:
        for filter_config in filters:
            codec = numcodecs_registry.get_codec(dict(filter_config))
            encoded = codec.encode(encoded)
    compressor_config = zarray.get("compressor")
    if compressor_config is not None:
        compressor = numcodecs_registry.get_codec(dict(compressor_config))
        return bytes(compressor.encode(encoded))
    if isinstance(encoded, np.ndarray):
        return encoded.tobytes(order="A")
    return bytes(encoded)
