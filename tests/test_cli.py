from __future__ import annotations

import importlib.metadata
import io
import json
import pathlib

import lazynwb._cli._main as cli_main


def test_console_entrypoint_smoke() -> None:
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    lazynwb_entry_points = [
        entry_point for entry_point in entry_points if entry_point.name == "lazynwb"
    ]

    assert len(lazynwb_entry_points) == 1
    assert lazynwb_entry_points[0].value == "lazynwb._cli._main:main"
    assert lazynwb_entry_points[0].load() is cli_main.main


def test_paths_command_writes_resolved_paths_json_to_stdout(
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first.nwb"
    second_path = tmp_path / "second.nwb"
    first_path.touch()
    second_path.touch()

    exit_code, stdout, stderr = _run_cli(["paths", str(first_path), str(second_path)])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": [
            {
                "input": str(first_path),
                "resolved": first_path.resolve().as_posix(),
            },
            {
                "input": str(second_path),
                "resolved": second_path.resolve().as_posix(),
            },
        ],
    }


def test_paths_command_supports_table_output(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "source.nwb"
    path.touch()

    exit_code, stdout, stderr = _run_cli(["paths", "--format", "table", str(path)])

    assert exit_code == 0
    assert stderr == ""
    assert "input" in stdout
    assert "resolved" in stdout
    assert str(path) in stdout
    assert path.resolve().as_posix() in stdout


def test_debug_logs_go_to_stderr_while_stdout_stays_json(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "debug.nwb"
    path.touch()

    exit_code, stdout, stderr = _run_cli(["--debug", "paths", str(path)])

    assert exit_code == 0
    assert json.loads(stdout)["paths"][0]["resolved"] == path.resolve().as_posix()
    assert "DEBUG:lazynwb._cli" in stderr
    assert "DEBUG:" not in stdout


def test_missing_explicit_path_returns_machine_readable_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    missing_path = tmp_path / "missing.nwb"

    exit_code, stdout, stderr = _run_cli(["paths", str(missing_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_path_not_found",
            "details": {
                "paths": [
                    {
                        "input": str(missing_path),
                        "resolved": missing_path.resolve(strict=False).as_posix(),
                    }
                ]
            },
            "message": "One or more explicit source paths do not exist.",
        }
    }


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = cli_main.main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()
