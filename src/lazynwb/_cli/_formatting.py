from __future__ import annotations

import collections.abc
import json
import typing

import lazynwb._cli._config as cli_config
import lazynwb._cli._preview as cli_preview
import lazynwb._cli._schema as cli_schema
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


def _schema_json_object(
    schema: cli_schema._TableSchema,
    resolved_source: cli_sources._ResolvedSource,
    *,
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
) -> dict[str, typing.Any]:
    return {
        "column_count": len(schema.columns),
        "columns": [_schema_column_json_object(column) for column in schema.columns],
        "command": "schema",
        "infer_schema_length": schema.infer_schema_length,
        "requested_table": schema.requested_table,
        "resolved_count": len(paths),
        "resolved_table_path": schema.resolved_table_path,
        "source": cli_sources._source_json_object(
            resolved_source,
            paths=paths,
        ),
    }


def _schema_column_json_object(
    column: cli_schema._SchemaColumn,
) -> dict[str, str | bool]:
    return {
        "dtype": column.dtype,
        "internal": column.internal,
        "name": column.name,
    }


def _preview_json_object(
    preview: cli_preview._TablePreview,
    resolved_source: cli_sources._ResolvedSource,
    *,
    paths: collections.abc.Sequence[collections.abc.Mapping[str, str]],
) -> dict[str, typing.Any]:
    return {
        "columns": list(preview.columns),
        "command": "preview",
        "limit": preview.limit,
        "max_limit": cli_preview._MAX_PREVIEW_LIMIT,
        "requested_table": preview.requested_table,
        "resolved_count": len(paths),
        "resolved_table_path": preview.resolved_table_path,
        "row_count": len(preview.rows),
        "rows": list(preview.rows),
        "source": cli_sources._source_json_object(
            resolved_source,
            paths=paths,
        ),
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


def _write_schema_table(
    stream: typing.TextIO,
    schema: cli_schema._TableSchema,
) -> None:
    rows = tuple(
        (column.name, column.dtype, str(column.internal).lower())
        for column in schema.columns
    )
    headers = ("name", "dtype", "internal")
    name_width = max((len(row[0]) for row in rows), default=0)
    dtype_width = max((len(row[1]) for row in rows), default=0)
    internal_width = max((len(row[2]) for row in rows), default=0)
    widths = (
        max(len(headers[0]), name_width),
        max(len(headers[1]), dtype_width),
        max(len(headers[2]), internal_width),
    )

    stream.write(
        f"{headers[0]:<{widths[0]}} | "
        f"{headers[1]:<{widths[1]}} | "
        f"{headers[2]:<{widths[2]}}\n"
    )
    stream.write(f"{'-' * widths[0]}-+-" f"{'-' * widths[1]}-+-" f"{'-' * widths[2]}\n")
    for column_name, dtype, internal in rows:
        stream.write(
            f"{column_name:<{widths[0]}} | "
            f"{dtype:<{widths[1]}} | "
            f"{internal:<{widths[2]}}\n"
        )


def _write_preview_table(
    stream: typing.TextIO,
    preview: cli_preview._TablePreview,
) -> None:
    columns = preview.columns
    if not columns:
        stream.write("(no columns)\n")
        return

    rows = tuple(
        tuple(_preview_cell(row.get(column)) for column in columns)
        for row in preview.rows
    )
    widths = tuple(
        max(
            len(column),
            max((len(row[column_index]) for row in rows), default=0),
        )
        for column_index, column in enumerate(columns)
    )

    stream.write(
        " | ".join(
            f"{column:<{widths[column_index]}}"
            for column_index, column in enumerate(columns)
        )
    )
    stream.write("\n")
    stream.write("-+-".join("-" * width for width in widths))
    stream.write("\n")
    for row in rows:
        stream.write(
            " | ".join(
                f"{cell:<{widths[column_index]}}"
                for column_index, cell in enumerate(row)
            )
        )
        stream.write("\n")


def _preview_cell(value: object, *, max_width: int = 80) -> str:
    if isinstance(value, (dict, list)):
        cell = json.dumps(value, sort_keys=True, separators=(",", ":"))
    elif value is None:
        cell = "null"
    else:
        cell = str(value)
    if len(cell) <= max_width:
        return cell
    return f"{cell[: max_width - 3]}..."
