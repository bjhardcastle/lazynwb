from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import typing
from collections.abc import Container, Mapping, Sequence

import numcodecs.registry as numcodecs_registry
import numpy as np

import lazynwb._zarr.chunk_planner as chunk_planner

logger = logging.getLogger(__name__)

_BytesLike = bytes | bytearray | memoryview
_SUPPORTED_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c"})
_SUPPORTED_ORDERS = frozenset({"C", "F"})


class _AsyncObjectClient(typing.Protocol):
    async def read_object(self, key: str) -> object: ...


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrV2ArraySpec:
    array_path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    order: str
    dimension_separator: str
    fill_value: object
    compressor: object | None
    filters: tuple[object, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ChunkTransferTimings:
    fetch_seconds: float
    decode_seconds: float
    assemble_seconds: float
    total_seconds: float


class _ZarrV2ChunkTransferEngine:
    """Private native Zarr v2 chunk transfer engine for exact array reads."""

    def __init__(self, object_client: _AsyncObjectClient) -> None:
        self._object_client = object_client

    async def _read_array_selection(
        self,
        *,
        array_path: str,
        zarray: Mapping[str, object],
        selection: object = None,
        available_chunk_keys: Container[str] | None = None,
        missing_chunk_keys: Container[str] | None = None,
    ) -> np.ndarray:
        spec, plan = _plan_zarr_v2_array_selection(
            array_path=array_path,
            zarray=zarray,
            selection=selection,
            available_chunk_keys=available_chunk_keys,
            missing_chunk_keys=missing_chunk_keys,
        )
        return await _read_zarr_v2_array_selection_from_plan(
            self._object_client,
            spec=spec,
            plan=plan,
        )


async def _read_zarr_v2_array_selection(
    object_client: _AsyncObjectClient,
    *,
    array_path: str,
    zarray: Mapping[str, object],
    selection: object = None,
    available_chunk_keys: Container[str] | None = None,
    missing_chunk_keys: Container[str] | None = None,
) -> np.ndarray:
    spec, plan = _plan_zarr_v2_array_selection(
        array_path=array_path,
        zarray=zarray,
        selection=selection,
        available_chunk_keys=available_chunk_keys,
        missing_chunk_keys=missing_chunk_keys,
    )
    return await _read_zarr_v2_array_selection_from_plan(
        object_client,
        spec=spec,
        plan=plan,
    )


async def _read_zarr_v2_array_selection_from_plan(
    object_client: _AsyncObjectClient,
    *,
    spec: _ZarrV2ArraySpec,
    plan: chunk_planner._ArrayChunkPlan,
) -> np.ndarray:
    return await _read_planned_chunks(object_client, spec, plan)


def _plan_zarr_v2_array_selection(
    *,
    array_path: str,
    zarray: Mapping[str, object],
    selection: object = None,
    available_chunk_keys: Container[str] | None = None,
    missing_chunk_keys: Container[str] | None = None,
) -> tuple[_ZarrV2ArraySpec, chunk_planner._ArrayChunkPlan]:
    spec = _zarr_v2_array_spec(array_path=array_path, zarray=zarray)
    plan = chunk_planner._plan_array_chunks(
        array_path=spec.array_path,
        shape=spec.shape,
        chunks=spec.chunks,
        selection=selection,
        dimension_separator=spec.dimension_separator,
        fill_value=spec.fill_value,
        available_chunk_keys=available_chunk_keys,
        missing_chunk_keys=missing_chunk_keys,
    )
    return spec, plan


async def _read_planned_chunks(
    object_client: _AsyncObjectClient,
    spec: _ZarrV2ArraySpec,
    plan: chunk_planner._ArrayChunkPlan,
) -> np.ndarray:
    started = time.perf_counter()
    output = np.empty(plan.output_shape, dtype=spec.dtype)
    keys_to_fetch = tuple(
        chunk_read.chunk_key
        for chunk_read in plan.chunk_reads
        if not chunk_read.requires_fill
    )
    logger.debug(
        "starting native Zarr v2 chunk transfer for %r: selection=%r "
        "output_shape=%s chunk_count=%d planned_missing_chunks=%d",
        plan.array_path,
        plan.selection,
        plan.output_shape,
        plan.chunk_count,
        plan.missing_chunk_count,
    )

    fetch_started = time.perf_counter()
    payloads = await _read_chunk_payloads(object_client, keys_to_fetch)
    fetch_seconds = time.perf_counter() - fetch_started
    fetched_bytes = sum(_payload_nbytes(payload) for payload in payloads.values())

    fetched_chunk_count = 0
    missing_chunk_count = 0
    decode_seconds = 0.0
    assemble_seconds = 0.0
    for chunk_read in plan.chunk_reads:
        payload = None if chunk_read.requires_fill else payloads.get(chunk_read.chunk_key)
        if payload is None:
            missing_chunk_count += 1
            assemble_started = time.perf_counter()
            _fill_output_selection(output, chunk_read.output_selection, spec.fill_value)
            assemble_seconds += time.perf_counter() - assemble_started
            logger.debug(
                "filled missing Zarr v2 chunk key=%r output=%r fill_value=%r",
                chunk_read.chunk_key,
                chunk_read.output_selection,
                spec.fill_value,
            )
            continue

        decode_started = time.perf_counter()
        chunk = _decode_chunk(payload, spec)
        decode_seconds += time.perf_counter() - decode_started
        assemble_started = time.perf_counter()
        output[chunk_read.output_selection] = chunk[chunk_read.input_selection]
        assemble_seconds += time.perf_counter() - assemble_started
        fetched_chunk_count += 1

    timings = _ChunkTransferTimings(
        fetch_seconds=fetch_seconds,
        decode_seconds=decode_seconds,
        assemble_seconds=assemble_seconds,
        total_seconds=time.perf_counter() - started,
    )
    _log_transfer_complete(
        plan=plan,
        fetched_chunk_count=fetched_chunk_count,
        missing_chunk_count=missing_chunk_count,
        fetched_bytes=fetched_bytes,
        timings=timings,
    )
    return output


async def _read_chunk_payloads(
    object_client: _AsyncObjectClient,
    keys: tuple[str, ...],
) -> dict[str, _BytesLike | None]:
    if not keys:
        return {}
    read_many = getattr(object_client, "read_many", None)
    if callable(read_many):
        response = await read_many(keys)
        payloads = _normalize_read_many_response(keys, response)
        logger.debug(
            "fetched Zarr v2 chunk batch via read_many: requested=%d returned=%d",
            len(keys),
            len(payloads),
        )
        return payloads

    read_object = getattr(object_client, "read_object", None)
    if not callable(read_object):
        msg = "Zarr v2 chunk object client must define read_many() or read_object()"
        raise TypeError(msg)
    results = await asyncio.gather(*(_read_one_object(read_object, key) for key in keys))
    logger.debug("fetched Zarr v2 chunk batch via read_object: requested=%d", len(keys))
    return dict(zip(keys, results))


async def _read_one_object(
    read_object: typing.Callable[[str], typing.Awaitable[object]],
    key: str,
) -> _BytesLike | None:
    try:
        payload = await read_object(key)
    except (FileNotFoundError, KeyError):
        logger.debug("Zarr v2 chunk object missing: key=%r", key)
        return None
    return _normalize_payload(payload)


def _normalize_read_many_response(
    keys: tuple[str, ...],
    response: object,
) -> dict[str, _BytesLike | None]:
    if isinstance(response, Mapping):
        return {
            str(key): _normalize_payload(payload)
            for key, payload in response.items()
            if str(key) in keys
        }
    if isinstance(response, Sequence) and not isinstance(
        response,
        (bytes, bytearray, memoryview, str),
    ):
        if len(response) != len(keys):
            msg = (
                "read_many() sequence response length must match requested keys: "
                f"{len(response)} != {len(keys)}"
            )
            raise ValueError(msg)
        return {
            key: _normalize_payload(payload) for key, payload in zip(keys, response)
        }
    msg = "read_many() must return a mapping or key-aligned sequence"
    raise TypeError(msg)


def _normalize_payload(payload: object) -> _BytesLike | None:
    if payload is None:
        return None
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return payload
    return bytes(payload)


def _payload_nbytes(payload: _BytesLike | None) -> int:
    if payload is None:
        return 0
    return len(payload)


def _decode_chunk(payload: _BytesLike, spec: _ZarrV2ArraySpec) -> np.ndarray:
    decoded: object = payload
    if spec.compressor is not None:
        decoded = _codec_decode(spec.compressor, decoded)
    for codec in reversed(spec.filters):
        decoded = _codec_decode(codec, decoded)

    chunk = _decoded_ndarray(decoded, spec.dtype)
    expected_items = _prod(spec.chunks)
    if chunk.size != expected_items:
        msg = (
            f"decoded Zarr v2 chunk has {chunk.size} items, expected {expected_items} "
            f"for chunk shape {spec.chunks}"
        )
        raise ValueError(msg)
    return chunk.reshape(-1, order="A").reshape(spec.chunks, order=spec.order)


def _codec_decode(codec: object, payload: object) -> object:
    decode = getattr(codec, "decode", None)
    if not callable(decode):
        msg = f"numcodecs codec {codec!r} has no decode() method"
        raise TypeError(msg)
    return decode(payload)


def _decoded_ndarray(decoded: object, dtype: np.dtype) -> np.ndarray:
    if isinstance(decoded, (bytes, bytearray, memoryview)):
        return np.frombuffer(decoded, dtype=dtype)
    chunk = np.asarray(decoded)
    if chunk.dtype == dtype:
        return chunk
    return chunk.view(dtype)


def _fill_output_selection(
    output: np.ndarray,
    output_selection: tuple[slice, ...],
    fill_value: object,
) -> None:
    output[output_selection] = fill_value


def _zarr_v2_array_spec(
    *,
    array_path: str,
    zarray: Mapping[str, object],
) -> _ZarrV2ArraySpec:
    _validate_zarr_format(zarray)
    dtype = np.dtype(str(zarray["dtype"]))
    _validate_dtype(dtype)
    shape = _int_tuple(zarray["shape"], field_name="shape")
    chunks = _int_tuple(zarray["chunks"], field_name="chunks")
    if len(shape) != len(chunks):
        msg = f"Zarr v2 shape rank {len(shape)} does not match chunks rank {len(chunks)}"
        raise ValueError(msg)
    order = _order(zarray.get("order", "C"))
    dimension_separator = _dimension_separator(zarray.get("dimension_separator", "."))
    fill_value = _decode_fill_value(zarray.get("fill_value"), dtype)
    return _ZarrV2ArraySpec(
        array_path=array_path,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        order=order,
        dimension_separator=dimension_separator,
        fill_value=fill_value,
        compressor=_codec_from_config(zarray.get("compressor"), field_name="compressor"),
        filters=_filters_from_config(zarray.get("filters")),
    )


def _validate_zarr_format(zarray: Mapping[str, object]) -> None:
    zarr_format = int(zarray.get("zarr_format", 2))
    if zarr_format != 2:
        msg = f"native chunk transfer supports Zarr v2 metadata, got {zarr_format}"
        raise ValueError(msg)


def _validate_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in _SUPPORTED_DTYPE_KINDS:
        msg = f"native Zarr v2 chunk transfer supports numeric dtypes, got {dtype!r}"
        raise TypeError(msg)


def _int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = f"Zarr v2 {field_name} must be a sequence"
        raise TypeError(msg)
    return tuple(int(item) for item in value)


def _order(value: object) -> str:
    order = str(value)
    if order not in _SUPPORTED_ORDERS:
        msg = f"Zarr v2 order must be 'C' or 'F', got {order!r}"
        raise ValueError(msg)
    return order


def _dimension_separator(value: object) -> str:
    if value is None:
        return "."
    return str(value)


def _decode_fill_value(value: object, dtype: np.dtype) -> object:
    if value is None:
        return np.array(0, dtype=dtype)[()]
    if dtype.kind == "f":
        return _decode_float_fill_value(value, dtype)
    if dtype.kind == "c":
        return _decode_complex_fill_value(value, dtype)
    return np.array(value, dtype=dtype)[()]


def _decode_float_fill_value(value: object, dtype: np.dtype) -> object:
    if value == "NaN":
        return np.array(np.nan, dtype=dtype)[()]
    if value == "Infinity":
        return np.array(np.inf, dtype=dtype)[()]
    if value == "-Infinity":
        return np.array(-np.inf, dtype=dtype)[()]
    return np.array(value, dtype=dtype)[()]


def _decode_complex_fill_value(value: object, dtype: np.dtype) -> object:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return np.array(value, dtype=dtype)[()]
    if len(value) != 2:
        msg = f"complex Zarr fill_value must have two items, got {value!r}"
        raise ValueError(msg)
    real_dtype = np.dtype(dtype.type().real.dtype)
    real = _decode_fill_value(value[0], real_dtype)
    imag = _decode_fill_value(value[1], real_dtype)
    return np.array(real + 1j * imag, dtype=dtype)[()]


def _codec_from_config(config: object, *, field_name: str) -> object | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        msg = f"Zarr v2 {field_name} codec config must be a mapping or null"
        raise TypeError(msg)
    return numcodecs_registry.get_codec(dict(config))


def _filters_from_config(config: object) -> tuple[object, ...]:
    if config is None:
        return ()
    if not isinstance(config, Sequence) or isinstance(config, (str, bytes, bytearray)):
        msg = "Zarr v2 filters codec config must be a sequence or null"
        raise TypeError(msg)
    filters: list[object] = []
    for index, filter_config in enumerate(config):
        codec = _codec_from_config(filter_config, field_name=f"filters[{index}]")
        if codec is not None:
            filters.append(codec)
    return tuple(filters)


def _prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _log_transfer_complete(
    *,
    plan: chunk_planner._ArrayChunkPlan,
    fetched_chunk_count: int,
    missing_chunk_count: int,
    fetched_bytes: int,
    timings: _ChunkTransferTimings,
) -> None:
    logger.debug(
        "completed native Zarr v2 chunk transfer for %r: chunk_count=%d "
        "fetched_chunks=%d missing_chunks=%d fetched_bytes=%d "
        "fetch=%.6fs decode=%.6fs assemble=%.6fs total=%.6fs",
        plan.array_path,
        plan.chunk_count,
        fetched_chunk_count,
        missing_chunk_count,
        fetched_bytes,
        timings.fetch_seconds,
        timings.decode_seconds,
        timings.assemble_seconds,
        timings.total_seconds,
    )
