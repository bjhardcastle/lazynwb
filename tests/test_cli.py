from __future__ import annotations

import importlib.metadata
import io
import json
import pathlib

import pytest

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


def test_config_init_writes_versioned_project_local_toml(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code, stdout, stderr = _run_cli(["config", "init"])

    config_path = tmp_path / "lazynwb.toml"
    assert exit_code == 0
    assert stderr == ""
    assert config_path.exists()
    assert json.loads(stdout) == {
        "command": "config init",
        "config": {
            "path": config_path.resolve().as_posix(),
            "version": 1,
        },
    }

    exit_code, stdout, stderr = _run_cli(["config", "show"])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "config show",
        "commands": {"paths": {"format": "json"}},
        "config": {
            "exists": True,
            "path": config_path.resolve().as_posix(),
            "version": 1,
        },
        "source": {
            "dandi": None,
            "kind": "none",
            "local": None,
            "paths": [],
            "precedence": "none",
        },
    }


def test_config_show_resolves_config_paths_relative_to_config_file(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_path = data_dir / "source.nwb"
    source_path.touch()
    _write_config(
        tmp_path / "lazynwb.toml",
        """
        version = 1

        [source]
        paths = ["data/source.nwb"]

        [source.local]
        root = "ignored"
        glob = "*.nwb"

        [source.dandi]
        dandiset_id = "000001"
        version = "draft"
        anonymous_s3 = false

        [commands.paths]
        format = "table"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["config", "show", "--config", str(tmp_path / "lazynwb.toml")]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "config show",
        "commands": {"paths": {"format": "table"}},
        "config": {
            "exists": True,
            "path": (tmp_path / "lazynwb.toml").resolve().as_posix(),
            "version": 1,
        },
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": [
                {
                    "input": "data/source.nwb",
                    "resolved": source_path.resolve().as_posix(),
                }
            ],
            "precedence": "config_paths",
        },
    }


def test_paths_command_uses_config_paths_and_command_default_format(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.nwb"
    source_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["source.nwb"]

        [commands.paths]
        format = "table"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    assert exit_code == 0
    assert stderr == ""
    assert "source.nwb" in stdout
    assert source_path.resolve().as_posix() in stdout

    exit_code, stdout, stderr = _run_cli(
        ["paths", "--config", str(config_path), "--format", "json"]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": [
            {
                "input": "source.nwb",
                "resolved": source_path.resolve().as_posix(),
            }
        ],
    }


def test_command_line_paths_override_config_paths(tmp_path: pathlib.Path) -> None:
    config_source_path = tmp_path / "config.nwb"
    cli_source_path = tmp_path / "cli.nwb"
    config_source_path.touch()
    cli_source_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["config.nwb"]
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["paths", "--config", str(config_path), str(cli_source_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": [
            {
                "input": str(cli_source_path),
                "resolved": cli_source_path.resolve().as_posix(),
            }
        ],
    }


def test_local_root_glob_source_discovers_paths_when_no_config_paths(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first_path = data_dir / "a.nwb"
    second_path = data_dir / "b.nwb"
    ignored_path = data_dir / "ignored.txt"
    first_path.touch()
    second_path.touch()
    ignored_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "data"
        glob = "*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": [
            {
                "input": "data/a.nwb",
                "resolved": first_path.resolve().as_posix(),
            },
            {
                "input": "data/b.nwb",
                "resolved": second_path.resolve().as_posix(),
            },
        ],
    }


def test_dandi_config_show_supports_anonymous_s3_and_flag_overrides(
    tmp_path: pathlib.Path,
) -> None:
    config_source_path = tmp_path / "config.nwb"
    config_source_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["config.nwb"]

        [source.dandi]
        dandiset_id = "000001"
        version = "0.1.0"
        anonymous_s3 = true
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "config",
            "show",
            "--config",
            str(config_path),
            "--dandiset-id",
            "000002",
            "--dandi-version",
            "draft",
            "--no-anonymous-s3",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["source"] == {
        "dandi": {
            "anonymous_s3": False,
            "dandiset_id": "000002",
            "version": "draft",
        },
        "kind": "dandi",
        "local": None,
        "paths": [],
        "precedence": "command_line_dandi",
    }


def test_incomplete_local_source_returns_machine_readable_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "data"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["config", "show", "--config", str(config_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_config_incomplete",
            "details": {"missing": ["source.local.glob"]},
            "message": "Local source configuration requires both root and glob.",
        }
    }


def test_conflicting_command_line_source_overrides_return_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.nwb"
    source_path.touch()

    exit_code, stdout, stderr = _run_cli(
        [
            "config",
            "show",
            "--path",
            str(source_path),
            "--root",
            str(tmp_path),
            "--glob",
            "*.nwb",
        ]
    )

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_config_conflict",
            "details": {"sources": ["paths", "local"]},
            "message": "Command-line source overrides are mutually incompatible.",
        }
    }


def test_invalid_toml_config_returns_machine_readable_parse_error(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "lazynwb.toml"
    config_path.write_text("version = [", encoding="utf-8")

    exit_code, stdout, stderr = _run_cli(["config", "show", "--config", str(config_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "config_parse_error",
            "details": {"path": config_path.resolve().as_posix()},
            "message": "The lazynwb config file is not valid TOML.",
        }
    }


def test_config_resolution_debug_logs_go_to_stderr(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.nwb"
    source_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["source.nwb"]
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "config", "show", "--config", str(config_path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["source"]["precedence"] == "config_paths"
    assert "loading lazynwb config" in stderr
    assert "resolved active path source" in stderr
    assert "DEBUG:" not in stdout


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = cli_main.main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _write_config(path: pathlib.Path, content: str) -> None:
    path.write_text(f"{content.strip()}\n", encoding="utf-8")
