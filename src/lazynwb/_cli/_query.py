from __future__ import annotations

import dataclasses
import logging
import time
import typing

import polars as pl

import lazynwb._cli._preview as cli_preview
import lazynwb._cli._tables as cli_tables
import lazynwb.conversion as conversion

logger = logging.getLogger(__name__)

_DEFAULT_QUERY_LIMIT = 100
_JSONValue = cli_preview._JSONValue


@dataclasses.dataclass(frozen=True, slots=True)
class _SQLQueryResult:
    query: str
    limit: int
    allow_large: bool
    columns: tuple[str, ...]
    rows: tuple[dict[str, _JSONValue], ...]
    observed_count: int
    truncated: bool


class _QueryRowCapExceededError(Exception):
    def __init__(
        self,
        *,
        cap: int,
        limit: int,
        observed_count: int,
    ) -> None:
        super().__init__("SQL query result exceeded the configured row cap.")
        self.cap = cap
        self.limit = limit
        self.observed_count = observed_count


def _validate_query_limit(
    limit: int | str | None,
    *,
    allow_large: bool,
    default_limit: int = _DEFAULT_QUERY_LIMIT,
) -> int:
    try:
        resolved_limit = default_limit if limit is None else int(limit)
    except ValueError as exc:
        raise ValueError("query limit must be an integer") from exc
    if resolved_limit <= 0:
        raise ValueError("query limit must be positive")
    if not allow_large and resolved_limit > default_limit:
        raise OverflowError("query limit exceeds the default cap")
    return resolved_limit


def _execute_sql_query(
    nwb_sources: tuple[str, ...],
    *,
    query: str,
    limit: int,
    allow_large: bool,
    infer_schema_length: int | None,
) -> _SQLQueryResult:
    sql_defaults = cli_tables._sql_defaults_json_object(
        infer_schema_length=infer_schema_length,
    )
    logger.debug(
        "starting SQL context creation for query: source_count=%d query=%r "
        "limit=%d allow_large=%s full_path=%s exclude_timeseries=%s "
        "rename_general_metadata=%s disable_progress=%s eager=%s "
        "infer_schema_length=%s",
        len(nwb_sources),
        query,
        limit,
        allow_large,
        sql_defaults["full_path"],
        sql_defaults["exclude_timeseries"],
        sql_defaults["rename_general_metadata"],
        sql_defaults["disable_progress"],
        sql_defaults["eager"],
        infer_schema_length,
    )
    context_started_at = time.perf_counter()
    try:
        sql_context = conversion.get_sql_context(
            nwb_sources=nwb_sources,
            full_path=True,
            exclude_timeseries=False,
            rename_general_metadata=True,
            disable_progress=True,
            infer_schema_length=infer_schema_length,
            eager=False,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - context_started_at) * 1000
        logger.debug(
            "SQL context creation for query failed: source_count=%d query=%r "
            "elapsed_ms=%.3f",
            len(nwb_sources),
            query,
            elapsed_ms,
            exc_info=True,
        )
        raise

    elapsed_ms = (time.perf_counter() - context_started_at) * 1000
    logger.debug(
        "completed SQL context creation for query: source_count=%d table_count=%d "
        "elapsed_ms=%.3f",
        len(nwb_sources),
        len(cli_tables._registered_table_names(sql_context)),
        elapsed_ms,
    )

    logger.debug(
        "starting SQL query planning: query=%r limit=%d allow_large=%s eager=%s",
        query,
        limit,
        allow_large,
        False,
    )
    plan_started_at = time.perf_counter()
    try:
        query_plan = sql_context.execute(query, eager=False)
    except Exception:
        elapsed_ms = (time.perf_counter() - plan_started_at) * 1000
        logger.debug(
            "SQL query planning failed: query=%r limit=%d elapsed_ms=%.3f",
            query,
            limit,
            elapsed_ms,
            exc_info=True,
        )
        raise

    elapsed_ms = (time.perf_counter() - plan_started_at) * 1000
    logger.debug(
        "completed SQL query planning: query=%r plan_type=%s elapsed_ms=%.3f",
        query,
        type(query_plan).__name__,
        elapsed_ms,
    )

    collect_limit = limit + 1
    logger.debug(
        "starting SQL query materialization: query=%r limit=%d collect_limit=%d "
        "allow_large=%s",
        query,
        limit,
        collect_limit,
        allow_large,
    )
    materialize_started_at = time.perf_counter()
    try:
        frame = _collect_limited_frame(query_plan, collect_limit=collect_limit)
    except Exception:
        elapsed_ms = (time.perf_counter() - materialize_started_at) * 1000
        logger.debug(
            "SQL query materialization failed: query=%r limit=%d "
            "collect_limit=%d elapsed_ms=%.3f",
            query,
            limit,
            collect_limit,
            elapsed_ms,
            exc_info=True,
        )
        raise

    observed_count = frame.height
    if observed_count > limit and not allow_large:
        elapsed_ms = (time.perf_counter() - materialize_started_at) * 1000
        logger.debug(
            "SQL query row cap exceeded: query=%r cap=%d limit=%d "
            "observed_count=%d elapsed_ms=%.3f",
            query,
            limit,
            limit,
            observed_count,
            elapsed_ms,
        )
        raise _QueryRowCapExceededError(
            cap=limit,
            limit=limit,
            observed_count=observed_count,
        )

    truncated = observed_count > limit
    if truncated:
        frame = frame.head(limit)
    rows = cli_preview._serialize_frame_rows(frame)
    columns = tuple(str(column) for column in frame.columns)
    elapsed_ms = (time.perf_counter() - materialize_started_at) * 1000
    logger.debug(
        "completed SQL query materialization: query=%r limit=%d row_count=%d "
        "observed_count=%d column_count=%d truncated=%s elapsed_ms=%.3f",
        query,
        limit,
        len(rows),
        observed_count,
        len(columns),
        truncated,
        elapsed_ms,
    )
    return _SQLQueryResult(
        query=query,
        limit=limit,
        allow_large=allow_large,
        columns=columns,
        rows=rows,
        observed_count=observed_count,
        truncated=truncated,
    )


def _collect_limited_frame(query_plan: object, *, collect_limit: int) -> pl.DataFrame:
    limited_plan = query_plan.head(collect_limit)
    collect = getattr(limited_plan, "collect", None)
    if callable(collect):
        return typing.cast(pl.DataFrame, collect())
    if isinstance(limited_plan, pl.DataFrame):
        return limited_plan
    raise TypeError(
        f"SQL query produced unsupported plan type: {type(limited_plan).__name__}"
    )
