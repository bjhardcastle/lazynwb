from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import logging
import math
import time
import typing

import numpy as np
import polars as pl

import lazynwb._cli._schema as cli_schema
import lazynwb.lazyframe as lazyframe

logger = logging.getLogger(__name__)

_DEFAULT_PREVIEW_LIMIT = 5
_MAX_PREVIEW_LIMIT = 50
_JSONValue: typing.TypeAlias = (
    str | int | float | bool | None | list["_JSONValue"] | dict[str, "_JSONValue"]
)
_NOT_SERIALIZED = object()


@dataclasses.dataclass(frozen=True, slots=True)
class _TablePreview:
    requested_table: str
    resolved_table_path: str
    limit: int
    columns: tuple[str, ...]
    rows: tuple[dict[str, _JSONValue], ...]


def _validate_preview_limit(
    limit: int | str | None,
    *,
    default_limit: int = _DEFAULT_PREVIEW_LIMIT,
    max_limit: int = _MAX_PREVIEW_LIMIT,
) -> int:
    try:
        resolved_limit = default_limit if limit is None else int(limit)
    except ValueError as exc:
        raise ValueError("preview limit must be an integer") from exc
    if resolved_limit <= 0:
        raise ValueError("preview limit must be positive")
    if resolved_limit > max_limit:
        raise OverflowError("preview limit exceeds the supported maximum")
    return resolved_limit


def _preview_table(
    nwb_sources: tuple[str, ...],
    *,
    table: str,
    limit: int,
) -> _TablePreview:
    table_path = cli_schema._resolve_table_path(table)
    logger.debug(
        "planning NWB table preview: source_count=%d requested_table=%s "
        "table_path=%s limit=%d max_limit=%d disable_progress=%s "
        "raise_on_missing=%s",
        len(nwb_sources),
        table,
        table_path,
        limit,
        _MAX_PREVIEW_LIMIT,
        True,
        False,
    )

    plan_started_at = time.perf_counter()
    try:
        preview_plan = lazyframe.scan_nwb(
            source=nwb_sources,
            table_path=table_path,
            disable_progress=True,
            raise_on_missing=False,
        ).head(limit)
    except Exception:
        elapsed_ms = (time.perf_counter() - plan_started_at) * 1000
        logger.debug(
            "NWB table preview planning failed: source_count=%d requested_table=%s "
            "table_path=%s limit=%d elapsed_ms=%.3f",
            len(nwb_sources),
            table,
            table_path,
            limit,
            elapsed_ms,
            exc_info=True,
        )
        raise

    elapsed_ms = (time.perf_counter() - plan_started_at) * 1000
    logger.debug(
        "planned NWB table preview: source_count=%d requested_table=%s "
        "table_path=%s limit=%d elapsed_ms=%.3f",
        len(nwb_sources),
        table,
        table_path,
        limit,
        elapsed_ms,
    )

    materialize_started_at = time.perf_counter()
    logger.debug(
        "starting NWB table preview materialization: requested_table=%s "
        "table_path=%s limit=%d disable_progress=%s",
        table,
        table_path,
        limit,
        True,
    )
    try:
        frame = preview_plan.collect()
    except Exception:
        elapsed_ms = (time.perf_counter() - materialize_started_at) * 1000
        logger.debug(
            "NWB table preview materialization failed: requested_table=%s "
            "table_path=%s limit=%d elapsed_ms=%.3f",
            table,
            table_path,
            limit,
            elapsed_ms,
            exc_info=True,
        )
        raise

    rows = _serialize_frame_rows(frame)
    columns = tuple(str(column) for column in frame.columns)
    elapsed_ms = (time.perf_counter() - materialize_started_at) * 1000
    logger.debug(
        "completed NWB table preview materialization: requested_table=%s "
        "table_path=%s limit=%d row_count=%d column_count=%d elapsed_ms=%.3f",
        table,
        table_path,
        limit,
        len(rows),
        len(columns),
        elapsed_ms,
    )
    return _TablePreview(
        requested_table=table,
        resolved_table_path=table_path,
        limit=limit,
        columns=columns,
        rows=rows,
    )


def _serialize_frame_rows(frame: pl.DataFrame) -> tuple[dict[str, _JSONValue], ...]:
    columns = tuple(str(column) for column in frame.columns)
    return tuple(
        {
            column: _to_json_value(row[column_index])
            for column_index, column in enumerate(columns)
        }
        for row in frame.iter_rows()
    )


def _to_json_value(value: object) -> _JSONValue:
    if isinstance(value, np.generic):
        return _to_json_value(value.item())
    scalar = _to_json_scalar(value)
    if scalar is not _NOT_SERIALIZED:
        return typing.cast(_JSONValue, scalar)
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())
    if isinstance(value, bytes):
        return _bytes_to_json_value(value)
    if isinstance(value, collections.abc.Mapping):
        return {
            str(key): _to_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_json_value(item) for item in sorted(value, key=str)]
    return str(value)


def _to_json_scalar(value: object) -> _JSONValue | object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return _NOT_SERIALIZED


def _bytes_to_json_value(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.hex()
