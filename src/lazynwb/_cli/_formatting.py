from __future__ import annotations

import collections.abc
import json
import typing

import lazynwb._cli._config as cli_config
import lazynwb._cli._sources as cli_sources


def _write_json(
    stream: typing.TextIO,
    payload: collections.abc.Mapping[str, typing.Any],
) -> None:
    stream.write(json.dumps(payload, indent=2, sort_keys=True))
    stream.write("\n")


def _source_paths_json_object(
    paths: collections.abc.Iterable[collections.abc.Mapping[str, str]],
) -> dict[str, typing.Any]:
    return {
        "command": "paths",
        "paths": list(paths),
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
