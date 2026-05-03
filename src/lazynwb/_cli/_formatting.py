from __future__ import annotations

import collections.abc
import json
import typing


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
