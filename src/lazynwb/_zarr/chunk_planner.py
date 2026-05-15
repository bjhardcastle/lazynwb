from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Container, Iterator, Sequence

logger = logging.getLogger(__name__)

_SUPPORTED_RANKS = frozenset({1, 2})
_SUPPORTED_DIMENSION_SEPARATORS = frozenset({".", "/"})
_IndexSelection = int | slice


@dataclasses.dataclass(frozen=True, slots=True)
class _ChunkReadPlan:
    """Pure plan for one logical Zarr v2 chunk read."""

    chunk_coords: tuple[int, ...]
    chunk_key: str
    input_selection: tuple[_IndexSelection, ...]
    output_selection: tuple[slice, ...]
    array_selection: tuple[_IndexSelection, ...]
    is_missing: bool
    fill_value: object | None

    @property
    def requires_fill(self) -> bool:
        return self.is_missing


@dataclasses.dataclass(frozen=True, slots=True)
class _ArrayChunkPlan:
    """Pure exact-array chunk plan independent of transport and decoding."""

    array_path: str
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    selection: tuple[_IndexSelection, ...]
    dimension_separator: str
    output_shape: tuple[int, ...]
    chunk_reads: tuple[_ChunkReadPlan, ...]
    fill_value: object | None
    missing_chunk_count: int

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_reads)

    @property
    def has_missing_chunks(self) -> bool:
        return self.missing_chunk_count > 0


@dataclasses.dataclass(frozen=True, slots=True)
class _AxisSelection:
    start: int
    stop: int
    index: int | None
    output_axis: int | None

    @property
    def is_integer(self) -> bool:
        return self.index is not None

    @property
    def output_size(self) -> int:
        if self.is_integer:
            return 0
        return max(0, self.stop - self.start)

    def as_index_selection(self) -> _IndexSelection:
        if self.index is not None:
            return self.index
        return slice(self.start, self.stop)


def _plan_array_chunks(
    *,
    array_path: str,
    shape: Sequence[int],
    chunks: Sequence[int],
    selection: object = None,
    dimension_separator: str = ".",
    fill_value: object | None = None,
    available_chunk_keys: Container[str] | None = None,
    missing_chunk_keys: Container[str] | None = None,
) -> _ArrayChunkPlan:
    """Plan Zarr v2 chunk reads for a simple exact-array selection.

    Supports 1D/2D arrays, unit-step slices, and integer indices. Missing chunk
    metadata is only derived from caller-provided key containers; when neither
    container is provided, all planned chunks are treated as present.
    """

    normalized_shape = _normalize_shape(shape)
    chunk_shape = _normalize_chunk_shape(chunks, len(normalized_shape))
    normalized_dimension_separator = _normalize_dimension_separator(dimension_separator)
    normalized_array_path = _normalize_array_path(array_path)
    axis_selections = _normalize_selection(selection, normalized_shape)
    output_shape = tuple(
        axis.output_size for axis in axis_selections if axis.output_axis is not None
    )
    normalized_selection = tuple(axis.as_index_selection() for axis in axis_selections)
    _validate_missing_chunk_inputs(available_chunk_keys, missing_chunk_keys)

    logger.debug(
        "planning Zarr v2 chunks for %r shape=%s chunks=%s selection=%r "
        "dimension_separator=%r",
        normalized_array_path,
        normalized_shape,
        chunk_shape,
        normalized_selection,
        normalized_dimension_separator,
    )

    chunk_reads = tuple(
        _plan_chunk_read(
            array_path=normalized_array_path,
            shape=normalized_shape,
            chunk_shape=chunk_shape,
            axis_selections=axis_selections,
            chunk_coords=chunk_coords,
            dimension_separator=normalized_dimension_separator,
            fill_value=fill_value,
            available_chunk_keys=available_chunk_keys,
            missing_chunk_keys=missing_chunk_keys,
        )
        for chunk_coords in _iter_chunk_coords(axis_selections, chunk_shape)
    )
    missing_chunk_count = sum(1 for chunk_read in chunk_reads if chunk_read.is_missing)
    plan = _ArrayChunkPlan(
        array_path=normalized_array_path,
        shape=normalized_shape,
        chunk_shape=chunk_shape,
        selection=normalized_selection,
        dimension_separator=normalized_dimension_separator,
        output_shape=output_shape,
        chunk_reads=chunk_reads,
        fill_value=fill_value,
        missing_chunk_count=missing_chunk_count,
    )
    _log_plan(plan)
    return plan


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(axis_size) for axis_size in shape)
    if len(normalized) not in _SUPPORTED_RANKS:
        msg = f"Zarr exact-array chunk planning supports 1D/2D arrays, got rank {len(normalized)}"
        raise ValueError(msg)
    if any(axis_size < 0 for axis_size in normalized):
        msg = f"array shape values must be non-negative, got {normalized!r}"
        raise ValueError(msg)
    return normalized


def _normalize_chunk_shape(chunks: Sequence[int], rank: int) -> tuple[int, ...]:
    chunk_shape = tuple(int(chunk_size) for chunk_size in chunks)
    if len(chunk_shape) != rank:
        msg = f"chunk rank {len(chunk_shape)} does not match array rank {rank}"
        raise ValueError(msg)
    if any(chunk_size <= 0 for chunk_size in chunk_shape):
        msg = f"chunk sizes must be positive, got {chunk_shape!r}"
        raise ValueError(msg)
    return chunk_shape


def _normalize_dimension_separator(dimension_separator: str) -> str:
    if dimension_separator not in _SUPPORTED_DIMENSION_SEPARATORS:
        msg = (
            "Zarr v2 dimension_separator must be '.' or '/', "
            f"got {dimension_separator!r}"
        )
        raise ValueError(msg)
    return dimension_separator


def _normalize_array_path(array_path: str) -> str:
    if array_path in {"", "/"}:
        return ""
    return "/".join(part for part in str(array_path).split("/") if part)


def _normalize_selection(
    selection: object,
    shape: tuple[int, ...],
) -> tuple[_AxisSelection, ...]:
    raw_selection = _selection_tuple(selection, len(shape))
    output_axis = 0
    normalized_axes: list[_AxisSelection] = []
    for axis_size, axis_selection in zip(shape, raw_selection):
        normalized_axis = _normalize_axis_selection(
            axis_selection,
            axis_size=axis_size,
            output_axis=output_axis,
        )
        normalized_axes.append(normalized_axis)
        if normalized_axis.output_axis is not None:
            output_axis += 1
    return tuple(normalized_axes)


def _selection_tuple(selection: object, rank: int) -> tuple[object, ...]:
    if selection is None or selection is Ellipsis:
        return tuple(slice(None) for _ in range(rank))
    if not isinstance(selection, tuple):
        return _pad_selection((selection,), rank)
    if selection.count(Ellipsis) > 1:
        msg = "selection can contain at most one ellipsis"
        raise IndexError(msg)
    if Ellipsis in selection:
        ellipsis_index = selection.index(Ellipsis)
        explicit_count = len(selection) - 1
        if explicit_count > rank:
            msg = f"too many indices for {rank}D array"
            raise IndexError(msg)
        fill_count = rank - explicit_count
        expanded = (
            *selection[:ellipsis_index],
            *(slice(None) for _ in range(fill_count)),
            *selection[ellipsis_index + 1 :],
        )
        return _pad_selection(expanded, rank)
    return _pad_selection(selection, rank)


def _pad_selection(selection: tuple[object, ...], rank: int) -> tuple[object, ...]:
    if len(selection) > rank:
        msg = f"too many indices for {rank}D array"
        raise IndexError(msg)
    return (*selection, *(slice(None) for _ in range(rank - len(selection))))


def _normalize_axis_selection(
    axis_selection: object,
    *,
    axis_size: int,
    output_axis: int,
) -> _AxisSelection:
    if isinstance(axis_selection, slice):
        start, stop, step = axis_selection.indices(axis_size)
        if step != 1:
            msg = "only unit-step slices are supported for Zarr chunk planning"
            raise ValueError(msg)
        return _AxisSelection(
            start=start,
            stop=stop,
            index=None,
            output_axis=output_axis,
        )
    if isinstance(axis_selection, int):
        index = _normalize_integer_index(axis_selection, axis_size)
        return _AxisSelection(
            start=index,
            stop=index + 1,
            index=index,
            output_axis=None,
        )
    msg = f"unsupported Zarr chunk selection item {axis_selection!r}"
    raise TypeError(msg)


def _normalize_integer_index(index: int, axis_size: int) -> int:
    normalized_index = index + axis_size if index < 0 else index
    if normalized_index < 0 or normalized_index >= axis_size:
        msg = f"index {index} is out of bounds for axis with size {axis_size}"
        raise IndexError(msg)
    return normalized_index


def _iter_chunk_coords(
    axis_selections: tuple[_AxisSelection, ...],
    chunk_shape: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    per_axis_coords = tuple(
        _axis_chunk_coords(axis_selection, chunk_size)
        for axis_selection, chunk_size in zip(axis_selections, chunk_shape)
    )
    return itertools.product(*per_axis_coords)


def _axis_chunk_coords(
    axis_selection: _AxisSelection,
    chunk_size: int,
) -> range:
    if axis_selection.index is not None:
        chunk_coord = axis_selection.index // chunk_size
        return range(chunk_coord, chunk_coord + 1)
    if axis_selection.start >= axis_selection.stop:
        return range(0)
    first_chunk = axis_selection.start // chunk_size
    last_chunk = (axis_selection.stop - 1) // chunk_size
    return range(first_chunk, last_chunk + 1)


def _plan_chunk_read(
    *,
    array_path: str,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    axis_selections: tuple[_AxisSelection, ...],
    chunk_coords: tuple[int, ...],
    dimension_separator: str,
    fill_value: object | None,
    available_chunk_keys: Container[str] | None,
    missing_chunk_keys: Container[str] | None,
) -> _ChunkReadPlan:
    chunk_key = _chunk_key(array_path, chunk_coords, dimension_separator)
    input_selection, output_selection, array_selection = _chunk_selections(
        shape=shape,
        chunk_shape=chunk_shape,
        axis_selections=axis_selections,
        chunk_coords=chunk_coords,
    )
    return _ChunkReadPlan(
        chunk_coords=chunk_coords,
        chunk_key=chunk_key,
        input_selection=input_selection,
        output_selection=output_selection,
        array_selection=array_selection,
        is_missing=_is_missing_chunk(
            chunk_key,
            available_chunk_keys=available_chunk_keys,
            missing_chunk_keys=missing_chunk_keys,
        ),
        fill_value=fill_value,
    )


def _chunk_key(
    array_path: str,
    chunk_coords: tuple[int, ...],
    dimension_separator: str,
) -> str:
    chunk_name = dimension_separator.join(str(coord) for coord in chunk_coords)
    if not array_path:
        return chunk_name
    return f"{array_path}/{chunk_name}"


def _chunk_selections(
    *,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    axis_selections: tuple[_AxisSelection, ...],
    chunk_coords: tuple[int, ...],
) -> tuple[tuple[_IndexSelection, ...], tuple[slice, ...], tuple[_IndexSelection, ...]]:
    input_selection: list[_IndexSelection] = []
    output_selection: list[slice] = []
    array_selection: list[_IndexSelection] = []

    for axis_size, chunk_size, axis_selection, chunk_coord in zip(
        shape,
        chunk_shape,
        axis_selections,
        chunk_coords,
    ):
        chunk_start = chunk_coord * chunk_size
        chunk_stop = min(chunk_start + chunk_size, axis_size)
        _append_axis_selections(
            input_selection=input_selection,
            output_selection=output_selection,
            array_selection=array_selection,
            axis_selection=axis_selection,
            chunk_start=chunk_start,
            chunk_stop=chunk_stop,
        )

    return tuple(input_selection), tuple(output_selection), tuple(array_selection)


def _append_axis_selections(
    *,
    input_selection: list[_IndexSelection],
    output_selection: list[slice],
    array_selection: list[_IndexSelection],
    axis_selection: _AxisSelection,
    chunk_start: int,
    chunk_stop: int,
) -> None:
    if axis_selection.index is not None:
        input_selection.append(axis_selection.index - chunk_start)
        array_selection.append(axis_selection.index)
        return

    read_start = max(axis_selection.start, chunk_start)
    read_stop = min(axis_selection.stop, chunk_stop)
    input_selection.append(slice(read_start - chunk_start, read_stop - chunk_start))
    output_selection.append(
        slice(read_start - axis_selection.start, read_stop - axis_selection.start)
    )
    array_selection.append(slice(read_start, read_stop))


def _validate_missing_chunk_inputs(
    available_chunk_keys: Container[str] | None,
    missing_chunk_keys: Container[str] | None,
) -> None:
    if available_chunk_keys is not None and missing_chunk_keys is not None:
        msg = "provide either available_chunk_keys or missing_chunk_keys, not both"
        raise ValueError(msg)


def _is_missing_chunk(
    chunk_key: str,
    *,
    available_chunk_keys: Container[str] | None,
    missing_chunk_keys: Container[str] | None,
) -> bool:
    if available_chunk_keys is not None:
        return chunk_key not in available_chunk_keys
    if missing_chunk_keys is not None:
        return chunk_key in missing_chunk_keys
    return False


def _log_plan(plan: _ArrayChunkPlan) -> None:
    logger.debug(
        "planned Zarr v2 chunks for %r: output_shape=%s chunk_count=%d "
        "missing_chunk_count=%d fill_value=%r",
        plan.array_path,
        plan.output_shape,
        plan.chunk_count,
        plan.missing_chunk_count,
        plan.fill_value,
    )
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for chunk_read in plan.chunk_reads:
        logger.debug(
            "planned Zarr v2 chunk key=%r coords=%s input=%r output=%r "
            "array=%r missing=%s",
            chunk_read.chunk_key,
            chunk_read.chunk_coords,
            chunk_read.input_selection,
            chunk_read.output_selection,
            chunk_read.array_selection,
            chunk_read.is_missing,
        )
