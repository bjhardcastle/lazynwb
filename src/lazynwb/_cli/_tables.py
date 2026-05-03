from __future__ import annotations

import dataclasses
import logging
import time
import typing

import lazynwb.conversion as conversion

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _SQLTable:
    name: str
    path: str


def _list_sql_tables(
    nwb_sources: tuple[str, ...],
    *,
    infer_schema_length: int | None,
) -> tuple[_SQLTable, ...]:
    logger.debug(
        "starting SQL context table discovery: source_count=%d full_path=%s "
        "exclude_timeseries=%s rename_general_metadata=%s disable_progress=%s eager=%s "
        "infer_schema_length=%s",
        len(nwb_sources),
        True,
        False,
        True,
        True,
        False,
        infer_schema_length,
    )
    started_at = time.perf_counter()
    sql_context = conversion.get_sql_context(
        nwb_sources=nwb_sources,
        full_path=True,
        exclude_timeseries=False,
        rename_general_metadata=True,
        disable_progress=True,
        infer_schema_length=infer_schema_length,
        eager=False,
    )
    table_names = _registered_table_names(sql_context)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.debug(
        "completed SQL context table discovery: table_count=%d elapsed_ms=%.3f",
        len(table_names),
        elapsed_ms,
    )
    return tuple(_SQLTable(name=table_name, path=table_name) for table_name in table_names)


def _registered_table_names(sql_context: object) -> tuple[str, ...]:
    tables = getattr(sql_context, "tables", None)
    if not callable(tables):
        return ()
    return tuple(sorted(str(table_name) for table_name in tables()))


def _sql_defaults_json_object(
    *,
    infer_schema_length: int | None,
) -> dict[str, typing.Any]:
    return {
        "disable_progress": True,
        "eager": False,
        "exclude_timeseries": False,
        "full_path": True,
        "infer_schema_length": infer_schema_length,
        "rename_general_metadata": True,
    }
