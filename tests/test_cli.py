from __future__ import annotations

import importlib.metadata
import io
import json
import pathlib
import typing

import pytest

import lazynwb._cli._main as cli_main
import lazynwb._cli._sources as cli_sources
import lazynwb.file_io


@pytest.fixture(autouse=True)
def reset_file_io_config() -> typing.Iterator[None]:
    original_anon = lazynwb.file_io.config.anon
    try:
        yield
    finally:
        lazynwb.file_io.config.anon = original_anon


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
    expected_paths = [
        {
            "input": str(first_path),
            "resolved": first_path.resolve().as_posix(),
        },
        {
            "input": str(second_path),
            "resolved": second_path.resolve().as_posix(),
        },
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": expected_paths,
            "precedence": "command_line_paths",
            "resolved_count": 2,
        },
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
            "resolved_count": 0,
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
            "resolved_count": 1,
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
    expected_paths = [
        {
            "input": "source.nwb",
            "resolved": source_path.resolve().as_posix(),
        }
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": expected_paths,
            "precedence": "config_paths",
            "resolved_count": 1,
        },
    }


def test_command_line_paths_override_config_paths(tmp_path: pathlib.Path) -> None:
    config_source_path = tmp_path / "config.nwb"
    cli_source_path = tmp_path / "cli.nwb"
    local_dir = tmp_path / "local"
    local_source_path = local_dir / "local.nwb"
    local_dir.mkdir()
    config_source_path.touch()
    cli_source_path.touch()
    local_source_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["config.nwb"]

        [source.local]
        root = "local"
        glob = "*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["paths", "--config", str(config_path), str(cli_source_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    expected_paths = [
        {
            "input": str(cli_source_path),
            "resolved": cli_source_path.resolve().as_posix(),
        }
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": expected_paths,
            "precedence": "command_line_paths",
            "resolved_count": 1,
        },
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
    expected_paths = [
        {
            "input": "data/a.nwb",
            "resolved": first_path.resolve().as_posix(),
        },
        {
            "input": "data/b.nwb",
            "resolved": second_path.resolve().as_posix(),
        },
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": None,
            "kind": "local",
            "local": {
                "glob": "*.nwb",
                "resolved_root": data_dir.resolve().as_posix(),
                "root": "data",
            },
            "paths": expected_paths,
            "precedence": "config_local",
            "resolved_count": 2,
        },
    }


def test_command_line_root_defaults_to_recursive_nwb_glob(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    nested_dir = data_dir / "nested"
    nested_dir.mkdir(parents=True)
    top_level_path = data_dir / "top.nwb"
    nested_path = nested_dir / "child.nwb"
    ignored_path = nested_dir / "ignored.txt"
    top_level_path.touch()
    nested_path.touch()
    ignored_path.touch()

    exit_code, stdout, stderr = _run_cli(["paths", "--root", str(data_dir)])

    assert exit_code == 0
    assert stderr == ""
    expected_paths = [
        {
            "input": f"{data_dir}/nested/child.nwb",
            "resolved": nested_path.resolve().as_posix(),
        },
        {
            "input": f"{data_dir}/top.nwb",
            "resolved": top_level_path.resolve().as_posix(),
        },
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": None,
            "kind": "local",
            "local": {
                "glob": "**/*.nwb",
                "resolved_root": data_dir.resolve().as_posix(),
                "root": str(data_dir),
            },
            "paths": expected_paths,
            "precedence": "command_line_local",
            "resolved_count": 2,
        },
    }


def test_config_show_reports_local_discovery_metadata_and_paths(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    nested_dir = data_dir / "nested"
    nested_dir.mkdir(parents=True)
    top_level_path = data_dir / "b.nwb"
    nested_path = nested_dir / "a.nwb"
    top_level_path.touch()
    nested_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "data"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["config", "show", "--config", str(config_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    expected_paths = [
        {
            "input": "data/b.nwb",
            "resolved": top_level_path.resolve().as_posix(),
        },
        {
            "input": "data/nested/a.nwb",
            "resolved": nested_path.resolve().as_posix(),
        },
    ]
    assert json.loads(stdout)["source"] == {
        "dandi": None,
        "kind": "local",
        "local": {
            "glob": "**/*.nwb",
            "resolved_root": data_dir.resolve().as_posix(),
            "root": "data",
        },
        "paths": expected_paths,
        "precedence": "config_local",
        "resolved_count": 2,
    }


def test_dandi_config_show_supports_anonymous_s3_and_flag_overrides(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.dandi]
        dandiset_id = "000001"
        version = "0.1.0"
        path_pattern = "sub-001/*.nwb"
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
            "--dandi-path-pattern",
            "sub-002/*.nwb",
            "--no-anonymous-s3",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert lazynwb.file_io.config.anon is False
    assert json.loads(stdout)["source"] == {
        "dandi": {
            "anonymous_s3": False,
            "dandiset_id": "000002",
            "path_pattern": "sub-002/*.nwb",
            "version": "draft",
        },
        "kind": "dandi",
        "local": None,
        "paths": [],
        "precedence": "command_line_dandi",
        "resolved_count": 0,
    }


def test_paths_command_resolves_dandi_config_with_version_and_path_pattern(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_calls: list[tuple[str, str | None, str]] = []
    url_calls: list[tuple[str, str, str | None]] = []

    def _get_dandiset_assets(
        dandiset_id: str,
        version: str | None,
        order: str = "path",
    ) -> list[dict[str, str]]:
        asset_calls.append((dandiset_id, version, order))
        return [
            {"asset_id": "asset-b", "path": "sub-002/session.nwb"},
            {"asset_id": "asset-txt", "path": "sub-001/readme.txt"},
            {"asset_id": "asset-c", "path": "other/session.nwb"},
            {"asset_id": "asset-a", "path": "sub-001/session.nwb"},
        ]

    def _get_asset_s3_url(
        dandiset_id: str,
        asset_id: str,
        version: str | None,
    ) -> str:
        url_calls.append((dandiset_id, asset_id, version))
        return f"s3://dandiarchive/{asset_id}.nwb"

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _get_dandiset_assets,
    )
    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_asset_s3_url",
        _get_asset_s3_url,
    )
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.dandi]
        dandiset_id = "000001"
        version = "0.1.0"
        path_pattern = "sub-*/*.nwb"
        anonymous_s3 = false
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    expected_paths = [
        {
            "asset_id": "asset-a",
            "input": "sub-001/session.nwb",
            "resolved": "s3://dandiarchive/asset-a.nwb",
        },
        {
            "asset_id": "asset-b",
            "input": "sub-002/session.nwb",
            "resolved": "s3://dandiarchive/asset-b.nwb",
        },
    ]
    assert exit_code == 0
    assert stderr == ""
    assert lazynwb.file_io.config.anon is False
    assert asset_calls == [("000001", "0.1.0", "path")]
    assert sorted(url_calls) == [
        ("000001", "asset-a", "0.1.0"),
        ("000001", "asset-b", "0.1.0"),
    ]
    assert json.loads(stdout) == {
        "command": "paths",
        "paths": expected_paths,
        "source": {
            "dandi": {
                "anonymous_s3": False,
                "dandiset_id": "000001",
                "path_pattern": "sub-*/*.nwb",
                "version": "0.1.0",
            },
            "kind": "dandi",
            "local": None,
            "paths": expected_paths,
            "precedence": "config_dandi",
            "resolved_count": 2,
        },
    }


def test_paths_command_reports_empty_dandi_asset_matches(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _get_dandiset_assets(
        dandiset_id: str,
        version: str | None,
        order: str = "path",
    ) -> list[dict[str, str]]:
        assert (dandiset_id, version, order) == ("000001", "draft", "path")
        return [
            {"asset_id": "asset-a", "path": "sub-001/session.nwb"},
            {"asset_id": "asset-txt", "path": "sub-001/readme.txt"},
        ]

    def _unexpected_get_asset_s3_url(*args: object, **kwargs: object) -> str:
        raise AssertionError("S3 URL lookup should not run for empty DANDI matches")

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _get_dandiset_assets,
    )
    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_asset_s3_url",
        _unexpected_get_asset_s3_url,
    )
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.dandi]
        dandiset_id = "000001"
        version = "draft"
        path_pattern = "sub-999/*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_dandi_no_assets",
            "details": {
                "asset_count": 2,
                "dandiset_id": "000001",
                "nwb_asset_count": 1,
                "path_pattern": "sub-999/*.nwb",
                "version": "draft",
            },
            "message": "No NWB assets matched the active DANDI source.",
        }
    }


def test_explicit_paths_take_priority_over_dandi_flags_without_network(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.nwb"
    source_path.touch()

    def _unexpected_get_dandiset_assets(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("DANDI helpers should not run for explicit paths")

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _unexpected_get_dandiset_assets,
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "paths",
            "--dandiset-id",
            "000001",
            "--dandi-path-pattern",
            "*.nwb",
            str(source_path),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["source"]["precedence"] == "command_line_paths"
    assert json.loads(stdout)["paths"] == [
        {
            "input": str(source_path),
            "resolved": source_path.resolve().as_posix(),
        }
    ]


def test_config_paths_take_priority_over_flagged_dandi_without_network(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.nwb"
    source_path.touch()

    def _unexpected_get_dandiset_assets(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("DANDI helpers should not run for config paths")

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _unexpected_get_dandiset_assets,
    )
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["source.nwb"]

        [source.dandi]
        dandiset_id = "000001"
        path_pattern = "*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "paths",
            "--config",
            str(config_path),
            "--dandiset-id",
            "000002",
            "--dandi-version",
            "draft",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["source"]["precedence"] == "config_paths"
    assert json.loads(stdout)["paths"] == [
        {
            "input": "source.nwb",
            "resolved": source_path.resolve().as_posix(),
        }
    ]


def test_local_source_takes_priority_over_dandi_fallback_without_network(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_path = data_dir / "source.nwb"
    source_path.touch()

    def _unexpected_get_dandiset_assets(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("DANDI helpers should not run for local discovery")

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _unexpected_get_dandiset_assets,
    )
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "data"
        glob = "*.nwb"

        [source.dandi]
        dandiset_id = "000001"
        path_pattern = "*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["source"]["precedence"] == "config_local"
    assert json.loads(stdout)["paths"] == [
        {
            "input": "data/source.nwb",
            "resolved": source_path.resolve().as_posix(),
        }
    ]


def test_incomplete_local_source_returns_machine_readable_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        glob = "*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["config", "show", "--config", str(config_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_config_incomplete",
            "details": {"missing": ["source.local.root"]},
            "message": "Local source configuration requires root when glob is provided.",
        }
    }


def test_missing_local_root_returns_machine_readable_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "lazynwb.toml"
    missing_root = tmp_path / "missing"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "missing"
        """,
    )

    exit_code, stdout, stderr = _run_cli(["paths", "--config", str(config_path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_root_not_found",
            "details": {
                "resolved_root": missing_root.resolve(strict=False).as_posix(),
                "root": "missing",
            },
            "message": "The configured local source root does not exist.",
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


def test_local_discovery_debug_logs_timing_and_match_count(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first_path = data_dir / "a.nwb"
    second_path = data_dir / "b.nwb"
    first_path.touch()
    second_path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.local]
        root = "data"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "paths", "--config", str(config_path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["source"]["resolved_count"] == 2
    assert "starting local source discovery" in stderr
    assert "completed local source discovery" in stderr
    assert "glob=**/*.nwb" in stderr
    assert "match_count=2" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def test_dandi_resolution_debug_logs_timing_and_filter_counts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _get_dandiset_assets(
        dandiset_id: str,
        version: str | None,
        order: str = "path",
    ) -> list[dict[str, str]]:
        assert (dandiset_id, version, order) == ("000001", "0.1.0", "path")
        return [
            {"asset_id": "asset-a", "path": "sub-001/session.nwb"},
            {"asset_id": "asset-txt", "path": "sub-001/readme.txt"},
        ]

    def _get_asset_s3_url(
        dandiset_id: str,
        asset_id: str,
        version: str | None,
    ) -> str:
        assert (dandiset_id, asset_id, version) == ("000001", "asset-a", "0.1.0")
        return "s3://dandiarchive/asset-a.nwb"

    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_dandiset_assets",
        _get_dandiset_assets,
    )
    monkeypatch.setattr(
        cli_sources.dandi,
        "_get_asset_s3_url",
        _get_asset_s3_url,
    )
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source.dandi]
        dandiset_id = "000001"
        version = "0.1.0"
        path_pattern = "sub-001/*.nwb"
        """,
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "paths", "--config", str(config_path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["source"]["resolved_count"] == 1
    assert "starting DANDI source resolution" in stderr
    assert "fetched DANDI asset metadata" in stderr
    assert "asset_count=2" in stderr
    assert "filtered DANDI assets to NWB paths" in stderr
    assert "before=2 after=1" in stderr
    assert "filtered DANDI assets by path pattern" in stderr
    assert "before=1 after=1" in stderr
    assert "resolved DANDI S3 URLs" in stderr
    assert "url_count=1" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = cli_main.main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _write_config(path: pathlib.Path, content: str) -> None:
    path.write_text(f"{content.strip()}\n", encoding="utf-8")
