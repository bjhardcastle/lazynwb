from __future__ import annotations

import collections.abc
import dataclasses
import json
import logging
import shlex
import time
import typing

import lazynwb._cli._config as cli_config
import lazynwb._cli._sources as cli_sources
import lazynwb._cli._tables as cli_tables

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _AgentContext:
    config: dict[str, typing.Any]
    notes: tuple[str, ...]
    paths: tuple[dict[str, str], ...]
    python_snippets: tuple[str, ...]
    recommended_workflow: tuple[str, ...]
    resolved_source: cli_sources._ResolvedSource
    sql_defaults: dict[str, typing.Any]
    tables: tuple[cli_tables._SQLTable, ...]


def _build_agent_context(
    loaded_config: cli_config._LoadedConfig,
    resolved_source: cli_sources._ResolvedSource,
    *,
    paths: tuple[dict[str, str], ...],
    infer_schema_length: int | None,
) -> _AgentContext:
    logger.debug(
        "starting CLI context generation: source_kind=%s precedence=%s "
        "resolved_count=%d infer_schema_length=%s",
        resolved_source.kind,
        resolved_source.precedence,
        len(paths),
        infer_schema_length,
    )
    started_at = time.perf_counter()
    sql_defaults = cli_tables._sql_defaults_json_object(
        infer_schema_length=infer_schema_length,
    )
    tables = cli_tables._list_sql_tables(
        tuple(path["resolved"] for path in paths),
        infer_schema_length=infer_schema_length,
    )
    logger.debug(
        "discovered SQL tables for CLI context: table_count=%d",
        len(tables),
    )
    python_snippets = _python_snippets(
        paths,
        tables=tables,
        infer_schema_length=infer_schema_length,
    )
    recommended_workflow = _recommended_workflow(
        paths,
        tables=tables,
        infer_schema_length=infer_schema_length,
    )
    notes = _context_notes(tables)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.debug(
        "completed CLI context generation: table_count=%d snippet_count=%d "
        "workflow_steps=%d elapsed_ms=%.3f",
        len(tables),
        len(python_snippets),
        len(recommended_workflow),
        elapsed_ms,
    )
    return _AgentContext(
        config=cli_config._config_json_object(loaded_config),
        notes=notes,
        paths=paths,
        python_snippets=python_snippets,
        recommended_workflow=recommended_workflow,
        resolved_source=resolved_source,
        sql_defaults=sql_defaults,
        tables=tables,
    )


def _python_snippets(
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
    *,
    tables: collections.abc.Sequence[cli_tables._SQLTable],
    infer_schema_length: int | None,
) -> tuple[str, ...]:
    table_path = _snippet_table_path(tables)
    logger.debug(
        "building source-aware Python snippets: source_count=%d table_path=%s "
        "infer_schema_length=%s",
        len(paths),
        table_path,
        infer_schema_length,
    )
    started_at = time.perf_counter()
    sources_literal = _python_sources_literal(paths)
    snippets = (
        _scan_snippet(
            sources_literal,
            table_path=table_path,
            infer_schema_length=infer_schema_length,
        ),
        _sql_context_snippet(
            sources_literal,
            infer_schema_length=infer_schema_length,
        ),
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.debug(
        "built source-aware Python snippets: snippet_count=%d elapsed_ms=%.3f",
        len(snippets),
        elapsed_ms,
    )
    return snippets


def _scan_snippet(
    sources_literal: str,
    *,
    table_path: str,
    infer_schema_length: int | None,
) -> str:
    lines = [
        "import lazynwb",
        "",
        f"sources = {sources_literal}",
        "",
        "lazyframe = lazynwb.scan_nwb(",
        "    sources,",
        f"    {_python_string_literal(table_path)},",
        "    disable_progress=True,",
    ]
    if infer_schema_length is not None:
        lines.append(f"    infer_schema_length={infer_schema_length},")
    lines.append(")")
    return "\n".join(lines)


def _sql_context_snippet(
    sources_literal: str,
    *,
    infer_schema_length: int | None,
) -> str:
    lines = [
        "import lazynwb",
        "",
        f"sources = {sources_literal}",
        "",
        "sql_context = lazynwb.get_sql_context(",
        "    sources,",
        "    full_path=True,",
        "    exclude_timeseries=False,",
        "    rename_general_metadata=True,",
        "    disable_progress=True,",
        "    eager=False,",
    ]
    if infer_schema_length is not None:
        lines.append(f"    infer_schema_length={infer_schema_length},")
    lines.append(")")
    return "\n".join(lines)


def _python_sources_literal(
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
) -> str:
    return json.dumps([path["resolved"] for path in paths], indent=4)


def _python_string_literal(value: str) -> str:
    return json.dumps(value)


def _recommended_workflow(
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
    *,
    tables: collections.abc.Sequence[cli_tables._SQLTable],
    infer_schema_length: int | None,
) -> tuple[str, ...]:
    logger.debug(
        "building recommended CLI workflow: source_count=%d table_count=%d",
        len(paths),
        len(tables),
    )
    source_args = tuple(path["resolved"] for path in paths)
    table_path = _snippet_table_path(tables)
    infer_args = _infer_schema_args(infer_schema_length)
    schema_args = (*infer_args, table_path, *source_args)
    preview_args = (table_path, *source_args)
    table_args = infer_args + source_args
    query = f"SELECT * FROM {_sql_identifier(table_path)} LIMIT 5"
    query_args = (*infer_args, query, *source_args)
    return (
        _command("lazynwb", "paths", *source_args),
        _command("lazynwb", "tables", *table_args),
        _command("lazynwb", "schema", *schema_args),
        _command("lazynwb", "preview", "--limit", "5", *preview_args),
        _command("lazynwb", "query", *query_args),
    )


def _infer_schema_args(infer_schema_length: int | None) -> tuple[str, ...]:
    if infer_schema_length is None:
        return ()
    return ("--infer-schema-length", str(infer_schema_length))


def _command(*parts: str) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _sql_identifier(table_name: str) -> str:
    return f'"{table_name.replace(chr(34), chr(34) * 2)}"'


def _snippet_table_path(
    tables: collections.abc.Sequence[cli_tables._SQLTable],
) -> str:
    if not tables:
        return "<table>"
    return tables[0].path


def _context_notes(
    tables: collections.abc.Sequence[cli_tables._SQLTable],
) -> tuple[str, ...]:
    notes = [
        "JSON is written to stdout; debug logs, when enabled with --debug, are written to stderr.",
        "Use the listed SQL table names exactly as shown when running schema, preview, or query.",
        "This context is NWB-specific.",
    ]
    if not tables:
        notes.append(
            "No SQL tables were discovered; verify the resolved source list before querying."
        )
    return tuple(notes)
