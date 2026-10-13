from __future__ import annotations

import dataclasses
import logging
import time

import lazynwb.tables as tables
import lazynwb.utils as utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _SchemaColumn:
    name: str
    dtype: str
    internal: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchema:
    requested_table: str
    resolved_table_path: str
    infer_schema_length: int | None
    columns: tuple[_SchemaColumn, ...]


def _resolve_table_path(table: str) -> str:
    return utils.normalize_internal_file_path(table)


def _inspect_table_schema(
    nwb_sources: tuple[str, ...],
    *,
    table: str,
    infer_schema_length: int | None,
) -> _TableSchema:
    table_path = _resolve_table_path(table)
    logger.debug(
        "starting NWB table schema inspection: source_count=%d requested_table=%s "
        "table_path=%s infer_schema_length=%s exclude_array_columns=%s "
        "exclude_internal_columns=%s raise_on_missing=%s",
        len(nwb_sources),
        table,
        table_path,
        infer_schema_length,
        False,
        False,
        False,
    )
    started_at = time.perf_counter()
    try:
        schema = tables.get_table_schema(
            file_paths=nwb_sources,
            table_path=table_path,
            first_n_files_to_infer_schema=infer_schema_length,
            exclude_array_columns=False,
            exclude_internal_columns=False,
            raise_on_missing=False,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "NWB table schema inspection failed: source_count=%d requested_table=%s "
            "table_path=%s infer_schema_length=%s elapsed_ms=%.3f",
            len(nwb_sources),
            table,
            table_path,
            infer_schema_length,
            elapsed_ms,
            exc_info=True,
        )
        raise

    columns = tuple(
        _SchemaColumn(
            name=column_name,
            dtype=str(dtype),
            internal=column_name in tables.INTERNAL_COLUMN_NAMES,
        )
        for column_name, dtype in schema.items()
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.debug(
        "completed NWB table schema inspection: source_count=%d requested_table=%s "
        "table_path=%s infer_schema_length=%s column_count=%d elapsed_ms=%.3f",
        len(nwb_sources),
        table,
        table_path,
        infer_schema_length,
        len(columns),
        elapsed_ms,
    )
    return _TableSchema(
        requested_table=table,
        resolved_table_path=table_path,
        infer_schema_length=infer_schema_length,
        columns=columns,
    )
