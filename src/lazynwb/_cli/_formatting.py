from __future__ import annotations

import collections.abc
import json
import typing

import lazynwb._cli._config as cli_config
import lazynwb._cli._sources as cli_sources


class _SQLTableLike(typing.Protocol):
    name: str
    path: str


def _write_json(
    stream: typing.TextIO,
    payload: collections.abc.Mapping[str, typing.Any],
) -> None:
    stream.write(json.dumps(payload, indent=2, sort_keys=True))
    stream.write("\n")


def _source_paths_json_object(
    paths: collections.abc.Iterable[collections.abc.Mapping[str, str]],
    resolved_source: cli_sources._ResolvedSource,
) -> dict[str, typing.Any]:
    resolved_paths = tuple(paths)
    return {
        "command": "paths",
        "paths": list(resolved_paths),
        "source": cli_sources._source_json_object(
            resolved_source,
            paths=resolved_paths,
        ),
    }


def _sql_tables_json_object(
    tables: collections.abc.Sequence[_SQLTableLike],
    resolved_source: cli_sources._ResolvedSource,
    *,
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
    sql_defaults: collections.abc.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    return {
        "command": "tables",
        "resolved_count": len(paths),
        "source": cli_sources._source_json_object(
            resolved_source,
            paths=paths,
        ),
        "sql": dict(sql_defaults),
        "table_count": len(tables),
        "tables": [_sql_table_json_object(table) for table in tables],
    }


def _sql_table_json_object(table: _SQLTableLike) -> dict[str, str]:
    return {
        "name": table.name,
        "path": table.path,
    }


def _config_init_json_object(path: str) -> dict[str, typing.Any]:
    return {
        "command": "config init",
        "config": {
            "path": path,
            "version": cli_config._CONFIG_VERSION,
        },
    }


def _config_show_json_object(
    loaded_config: cli_config._LoadedConfig,
    resolved_source: cli_sources._ResolvedSource,
) -> dict[str, typing.Any]:
    return {
        "command": "config show",
        "commands": cli_config._commands_json_object(loaded_config.project.commands),
        "config": cli_config._config_json_object(loaded_config),
        "source": cli_sources._source_json_object(resolved_source),
    }


def _write_source_paths_table(
    stream: typing.TextIO,
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
) -> None:
    rows = tuple((path["input"], path["resolved"]) for path in paths)
    headers = ("input", "resolved")
    input_width = max((len(row[0]) for row in rows), default=0)
    resolved_width = max((len(row[1]) for row in rows), default=0)
    widths = (
        max(len(headers[0]), input_width),
        max(len(headers[1]), resolved_width),
    )

    stream.write(f"{headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}}\n")
    stream.write(f"{'-' * widths[0]}-+-{'-' * widths[1]}\n")
    for input_path, resolved_path in rows:
        stream.write(f"{input_path:<{widths[0]}} | {resolved_path:<{widths[1]}}\n")


def _write_sql_tables_table(
    stream: typing.TextIO,
    tables: collections.abc.Sequence[_SQLTableLike],
) -> None:
    rows = tuple((table.name, table.path) for table in tables)
    headers = ("name", "path")
    name_width = max((len(row[0]) for row in rows), default=0)
    path_width = max((len(row[1]) for row in rows), default=0)
    widths = (
        max(len(headers[0]), name_width),
        max(len(headers[1]), path_width),
    )

    stream.write(f"{headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}}\n")
    stream.write(f"{'-' * widths[0]}-+-{'-' * widths[1]}\n")
    for table_name, table_path in rows:
        stream.write(f"{table_name:<{widths[0]}} | {table_path:<{widths[1]}}\n")
