from __future__ import annotations

import importlib.metadata
import io
import json
import pathlib
import typing

import polars
import pytest

import lazynwb._cli._main as cli_main
import lazynwb._cli._preview as cli_preview
import lazynwb._cli._query as cli_query
import lazynwb._cli._schema as cli_schema
import lazynwb._cli._sources as cli_sources
import lazynwb._cli._tables as cli_tables
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


def test_root_usage_error_includes_compact_recovery_fields() -> None:
    exit_code, stdout, stderr = _run_cli([])

    assert exit_code == 2
    assert stderr == ""
    assert stdout.endswith("\n")
    payload = _load_single_json_object(stdout)
    assert payload["error"]["code"] == "usage_error"
    assert (
        payload["error"]["message"] == "the following arguments are required: command"
    )
    assert payload["error"]["details"] == {
        "help_command": "lazynwb --help",
        "recovery_hint": "Run lazynwb --help to choose a command.",
        "usage": (
            "usage: lazynwb [-h] [--debug]\n"
            "               {paths,tables,context,schema,preview,query,config} ..."
        ),
        "valid_commands": [
            "paths",
            "tables",
            "context",
            "schema",
            "preview",
            "query",
            "config",
        ],
    }


def test_unknown_root_command_reports_valid_commands_in_cli_order() -> None:
    exit_code, stdout, stderr = _run_cli(["bogus"])

    assert exit_code == 2
    assert stderr == ""
    payload = _load_single_json_object(stdout)
    assert payload["error"]["code"] == "usage_error"
    assert payload["error"]["message"] == (
        "argument command: invalid choice: 'bogus' "
        "(choose from 'paths', 'tables', 'context', 'schema', 'preview', "
        "'query', 'config')"
    )
    assert payload["error"]["details"]["help_command"] == "lazynwb --help"
    assert payload["error"]["details"]["recovery_hint"] == (
        "Run lazynwb --help to choose a command."
    )
    assert payload["error"]["details"]["valid_commands"] == [
        "paths",
        "tables",
        "context",
        "schema",
        "preview",
        "query",
        "config",
    ]
    assert payload["error"]["details"]["usage"].startswith("usage: lazynwb")


def test_subcommand_usage_error_points_at_specific_help_command() -> None:
    exit_code, stdout, stderr = _run_cli(["paths", "--format", "bogus"])

    assert exit_code == 2
    assert stderr == ""
    payload = _load_single_json_object(stdout)
    assert payload["error"]["code"] == "usage_error"
    assert payload["error"]["message"] == (
        "argument --format: invalid choice: 'bogus' (choose from 'json', 'table')"
    )
    assert payload["error"]["details"]["help_command"] == "lazynwb paths --help"
    assert payload["error"]["details"]["recovery_hint"] == (
        "Run lazynwb paths --help to inspect valid options."
    )
    assert payload["error"]["details"]["usage"].startswith("usage: lazynwb paths")
    assert "Inspect NWB sources" not in stdout
    assert "valid_commands" not in payload["error"]["details"]


def test_debug_usage_error_keeps_stdout_parseable_and_logs_to_stderr() -> None:
    exit_code, stdout, stderr = _run_cli(["--debug", "paths", "--format", "bogus"])

    assert exit_code == 2
    payload = _load_single_json_object(stdout)
    assert payload["error"]["details"]["help_command"] == "lazynwb paths --help"
    assert "DEBUG:lazynwb._cli" in stderr
    assert "CLI usage error recovery guidance" in stderr
    assert "DEBUG:" not in stdout


def test_tables_command_lists_sql_ready_tables_json(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(["tables", str(local_hdf5_path)])

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    table_names = [table["name"] for table in payload["tables"]]
    assert table_names == sorted(table_names)
    assert payload["command"] == "tables"
    assert payload["resolved_count"] == 1
    assert payload["source"] == {
        "dandi": None,
        "kind": "paths",
        "local": None,
        "paths": [
            {
                "input": str(local_hdf5_path),
                "resolved": local_hdf5_path.resolve().as_posix(),
            }
        ],
        "precedence": "command_line_paths",
        "resolved_count": 1,
    }
    assert payload["sql"] == {
        "disable_progress": True,
        "eager": False,
        "exclude_timeseries": False,
        "full_path": True,
        "infer_schema_length": None,
        "rename_general_metadata": True,
    }
    assert payload["table_count"] == len(table_names)
    assert "intervals/trials" in table_names
    assert "processing/behavior/running_speed_with_rate" in table_names
    assert "processing/behavior/running_speed_with_timestamps" in table_names
    assert "session" in table_names
    assert "general" not in table_names


def test_tables_command_supports_table_output(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()

    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units", "intervals/trials")),
    )

    exit_code, stdout, stderr = _run_cli(["tables", "--format", "table", str(path)])

    assert exit_code == 0
    assert stderr == ""
    assert "name" in stdout
    assert "path" in stdout
    assert "intervals/trials" in stdout
    assert "units" in stdout


def test_tables_command_passes_agent_friendly_sql_defaults(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    calls: list[dict[str, object]] = []

    def _get_sql_context(
        nwb_sources: tuple[str, ...],
        **kwargs: object,
    ) -> _FakeSQLContext:
        calls.append({"nwb_sources": nwb_sources, **kwargs})
        return _FakeSQLContext(("units",))

    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        _get_sql_context,
    )

    exit_code, stdout, stderr = _run_cli(
        ["tables", "--infer-schema-length", "1", str(path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["tables"] == [{"name": "units", "path": "units"}]
    assert calls == [
        {
            "disable_progress": True,
            "eager": False,
            "exclude_timeseries": False,
            "full_path": True,
            "infer_schema_length": 1,
            "nwb_sources": (path.resolve().as_posix(),),
            "rename_general_metadata": True,
        }
    ]


def test_tables_command_uses_config_paths_when_no_path_args(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    config_path = tmp_path / "lazynwb.toml"
    _write_config(
        config_path,
        """
        version = 1

        [source]
        paths = ["source.nwb"]
        """,
    )
    calls: list[tuple[str, ...]] = []

    def _get_sql_context(
        nwb_sources: tuple[str, ...],
        **kwargs: object,
    ) -> _FakeSQLContext:
        calls.append(nwb_sources)
        return _FakeSQLContext(("units",))

    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        _get_sql_context,
    )

    exit_code, stdout, stderr = _run_cli(["tables", "--config", str(config_path)])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["source"]["precedence"] == "config_paths"
    assert calls == [(path.resolve().as_posix(),)]


def test_tables_command_returns_missing_source_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code, stdout, stderr = _run_cli(["tables"])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_not_configured",
            "details": {
                "source": {
                    "dandi": None,
                    "kind": "none",
                    "local": None,
                    "paths": [],
                    "precedence": "none",
                    "resolved_count": 0,
                }
            },
            "message": "No lazynwb source is configured.",
        }
    }


def test_tables_command_returns_missing_table_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(()),
    )

    exit_code, stdout, stderr = _run_cli(["tables", str(path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "tables_not_found",
            "details": {
                "resolved_count": 1,
                "source": {
                    "dandi": None,
                    "kind": "paths",
                    "local": None,
                    "paths": [
                        {
                            "input": str(path),
                            "resolved": path.resolve().as_posix(),
                        }
                    ],
                    "precedence": "command_line_paths",
                    "resolved_count": 1,
                },
                "sql": {
                    "disable_progress": True,
                    "eager": False,
                    "exclude_timeseries": False,
                    "full_path": True,
                    "infer_schema_length": None,
                    "rename_general_metadata": True,
                },
            },
            "message": "No SQL-ready NWB tables were found for the resolved sources.",
        }
    }


def test_tables_command_debug_logs_source_resolution_and_sql_discovery(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units",)),
    )

    exit_code, stdout, stderr = _run_cli(["--debug", "tables", str(path)])

    assert exit_code == 0
    assert json.loads(stdout)["tables"] == [{"name": "units", "path": "units"}]
    assert "resolving active source" in stderr
    assert "resolved table source paths" in stderr
    assert "starting SQL context table discovery" in stderr
    assert "disable_progress=True" in stderr
    assert "completed SQL context table discovery" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def test_context_command_reports_explicit_sources_tables_and_workflow_json(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.nwb"
    second_path = tmp_path / "second.nwb"
    first_path.touch()
    second_path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units", "intervals/trials")),
    )

    exit_code, stdout, stderr = _run_cli(["context", str(first_path), str(second_path)])

    assert exit_code == 0
    assert stderr == ""
    assert stdout.endswith("\n")
    payload = json.loads(stdout)
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
    assert payload["command"] == "context"
    assert payload["resolved_count"] == 2
    assert payload["source"] == {
        "dandi": None,
        "kind": "paths",
        "local": None,
        "paths": expected_paths,
        "precedence": "command_line_paths",
        "resolved_count": 2,
    }
    assert payload["tables"] == [
        {"name": "intervals/trials", "path": "intervals/trials"},
        {"name": "units", "path": "units"},
    ]
    assert payload["table_count"] == 2
    assert payload["sql"] == {
        "disable_progress": True,
        "eager": False,
        "exclude_timeseries": False,
        "full_path": True,
        "infer_schema_length": None,
        "rename_general_metadata": True,
    }
    assert payload["recommended_workflow"][0].startswith("lazynwb paths ")
    assert "lazynwb tables " in payload["recommended_workflow"][1]
    assert "lazynwb schema intervals/trials " in payload["recommended_workflow"][2]
    assert (
        'SELECT * FROM "intervals/trials" LIMIT 5' in payload["recommended_workflow"][4]
    )


def test_context_command_python_snippets_are_source_aware(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units",)),
    )

    exit_code, stdout, stderr = _run_cli(
        ["context", "--infer-schema-length", "1", str(path)]
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert len(payload["python_snippets"]) == 2
    scan_snippet, sql_snippet = payload["python_snippets"]
    resolved_path = path.resolve().as_posix()
    assert f'"{resolved_path}"' in scan_snippet
    assert f'"{resolved_path}"' in sql_snippet
    assert 'lazyframe = lazynwb.scan_nwb(\n    sources,\n    "units",' in scan_snippet
    assert "disable_progress=True" in scan_snippet
    assert "infer_schema_length=1" in scan_snippet
    assert "sql_context = lazynwb.get_sql_context(" in sql_snippet
    assert "full_path=True" in sql_snippet
    assert "exclude_timeseries=False" in sql_snippet
    assert "rename_general_metadata=True" in sql_snippet
    assert "eager=False" in sql_snippet
    assert (
        "lazynwb tables --infer-schema-length 1 " in payload["recommended_workflow"][1]
    )
    assert (
        "lazynwb schema --infer-schema-length 1 units "
        in payload["recommended_workflow"][2]
    )
    assert "lazynwb preview --limit 5 units " in payload["recommended_workflow"][3]
    assert (
        "lazynwb query --infer-schema-length 1 " in payload["recommended_workflow"][4]
    )


def test_context_command_emits_empty_table_context_with_placeholder_snippet(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(()),
    )

    exit_code, stdout, stderr = _run_cli(["context", str(path)])

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["tables"] == []
    assert payload["table_count"] == 0
    assert '"<table>"' in payload["python_snippets"][0]
    assert any("No SQL tables were discovered" in note for note in payload["notes"])


def test_context_command_supports_text_output(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units",)),
    )

    exit_code, stdout, stderr = _run_cli(["context", "--format", "text", str(path)])

    assert exit_code == 0
    assert stderr == ""
    assert stdout.startswith("lazynwb context\n")
    assert "Source\n" in stdout
    assert "SQL tables\n" in stdout
    assert "  - units\n" in stdout
    assert "Next CLI commands\n" in stdout
    assert "Python snippets\n" in stdout
    assert "lazynwb.scan_nwb" in stdout
    assert "lazynwb.get_sql_context" in stdout


def test_context_command_debug_logs_stay_on_stderr(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_tables.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeSQLContext(("units",)),
    )

    exit_code, stdout, stderr = _run_cli(["--debug", "context", str(path)])

    assert exit_code == 0
    assert json.loads(stdout)["tables"] == [{"name": "units", "path": "units"}]
    assert "resolving active source" in stderr
    assert "resolved context source paths" in stderr
    assert "starting CLI context generation" in stderr
    assert "starting SQL context table discovery" in stderr
    assert "building source-aware Python snippets" in stderr
    assert "built source-aware Python snippets" in stderr
    assert "completed CLI context generation" in stderr
    assert "serializing CLI context" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def test_schema_command_inspects_local_fixture_schema_json(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(["schema", "units", str(local_hdf5_path)])

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "column_count": 8,
        "columns": [
            {"dtype": "Int64", "internal": False, "name": "id"},
            {
                "dtype": "List(Array(Float64, shape=(2,)))",
                "internal": False,
                "name": "obs_intervals",
            },
            {"dtype": "List(Float64)", "internal": False, "name": "spike_times"},
            {"dtype": "String", "internal": False, "name": "structure"},
            {
                "dtype": "Array(Float64, shape=(25, 384))",
                "internal": False,
                "name": "waveform_mean",
            },
            {"dtype": "String", "internal": True, "name": "_nwb_path"},
            {"dtype": "String", "internal": True, "name": "_table_path"},
            {"dtype": "UInt32", "internal": True, "name": "_table_index"},
        ],
        "command": "schema",
        "infer_schema_length": None,
        "requested_table": "units",
        "resolved_count": 1,
        "resolved_table_path": "units",
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": [
                {
                    "input": str(local_hdf5_path),
                    "resolved": local_hdf5_path.resolve().as_posix(),
                }
            ],
            "precedence": "command_line_paths",
            "resolved_count": 1,
        },
    }


def test_schema_command_accepts_session_alias_for_general_table(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(["schema", "session", str(local_hdf5_path)])

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["requested_table"] == "session"
    assert payload["resolved_table_path"] == "general"
    column_names = [column["name"] for column in payload["columns"]]
    assert "session_description" in column_names
    assert "_nwb_path" in column_names


def test_schema_command_passes_schema_inference_limit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    calls: list[dict[str, object]] = []

    def _get_table_schema(**kwargs: object) -> polars.Schema:
        calls.append(kwargs)
        return polars.Schema({"unit_id": polars.Int64})

    monkeypatch.setattr(
        cli_schema.tables,
        "get_table_schema",
        _get_table_schema,
    )

    exit_code, stdout, stderr = _run_cli(
        ["schema", "--infer-schema-length", "1", "units", str(path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["infer_schema_length"] == 1
    assert calls == [
        {
            "exclude_array_columns": False,
            "exclude_internal_columns": False,
            "file_paths": (path.resolve().as_posix(),),
            "first_n_files_to_infer_schema": 1,
            "raise_on_missing": False,
            "table_path": "units",
        }
    ]


def test_schema_command_supports_table_output(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()

    monkeypatch.setattr(
        cli_schema.tables,
        "get_table_schema",
        lambda **kwargs: polars.Schema(
            {"unit_id": polars.Int64, "_nwb_path": polars.String}
        ),
    )

    exit_code, stdout, stderr = _run_cli(
        ["schema", "--format", "table", "units", str(path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "name" in stdout
    assert "dtype" in stdout
    assert "internal" in stdout
    assert "unit_id" in stdout
    assert "Int64" in stdout
    assert "_nwb_path" in stdout
    assert "true" in stdout


def test_schema_command_returns_missing_table_error(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        ["schema", "--infer-schema-length", "1", "not_a_table", str(local_hdf5_path)]
    )

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "schema_not_found",
            "details": {
                "infer_schema_length": 1,
                "requested_table": "not_a_table",
                "resolved_count": 1,
                "resolved_table_path": "not_a_table",
                "source": {
                    "dandi": None,
                    "kind": "paths",
                    "local": None,
                    "paths": [
                        {
                            "input": str(local_hdf5_path),
                            "resolved": local_hdf5_path.resolve().as_posix(),
                        }
                    ],
                    "precedence": "command_line_paths",
                    "resolved_count": 1,
                },
            },
            "message": "No NWB table schema was found for the resolved sources.",
        }
    }


def test_schema_command_debug_logs_source_resolution_and_schema_inspection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_schema.tables,
        "get_table_schema",
        lambda **kwargs: polars.Schema({"unit_id": polars.Int64}),
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "schema", "--infer-schema-length", "1", "units", str(path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["columns"] == [
        {"dtype": "Int64", "internal": False, "name": "unit_id"}
    ]
    assert "resolving active source" in stderr
    assert "resolved schema source paths" in stderr
    assert "starting NWB table schema inspection" in stderr
    assert "table_path=units" in stderr
    assert "infer_schema_length=1" in stderr
    assert "completed NWB table schema inspection" in stderr
    assert "column_count=1" in stderr
    assert "serializing NWB table schema" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def test_preview_command_reads_local_fixture_rows_json(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        ["preview", "intervals/trials", str(local_hdf5_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    resolved_path = local_hdf5_path.resolve().as_posix()
    assert json.loads(stdout) == {
        "columns": [
            "condition",
            "id",
            "start_time",
            "stop_time",
            "_nwb_path",
            "_table_path",
            "_table_index",
        ],
        "command": "preview",
        "limit": 5,
        "max_limit": 50,
        "requested_table": "intervals/trials",
        "resolved_count": 1,
        "resolved_table_path": "intervals/trials",
        "row_count": 5,
        "rows": [
            {
                "_nwb_path": resolved_path,
                "_table_index": 0,
                "_table_path": "intervals/trials",
                "condition": "A",
                "id": 0,
                "start_time": 0.05,
                "stop_time": 1.85,
            },
            {
                "_nwb_path": resolved_path,
                "_table_index": 1,
                "_table_path": "intervals/trials",
                "condition": "B",
                "id": 1,
                "start_time": 2.05,
                "stop_time": 3.8499999999999996,
            },
            {
                "_nwb_path": resolved_path,
                "_table_index": 2,
                "_table_path": "intervals/trials",
                "condition": "C",
                "id": 2,
                "start_time": 4.05,
                "stop_time": 5.85,
            },
            {
                "_nwb_path": resolved_path,
                "_table_index": 3,
                "_table_path": "intervals/trials",
                "condition": "D",
                "id": 3,
                "start_time": 6.05,
                "stop_time": 7.85,
            },
            {
                "_nwb_path": resolved_path,
                "_table_index": 4,
                "_table_path": "intervals/trials",
                "condition": "E",
                "id": 4,
                "start_time": 8.05,
                "stop_time": 9.850000000000001,
            },
        ],
        "source": {
            "dandi": None,
            "kind": "paths",
            "local": None,
            "paths": [
                {
                    "input": str(local_hdf5_path),
                    "resolved": resolved_path,
                }
            ],
            "precedence": "command_line_paths",
            "resolved_count": 1,
        },
    }


def test_preview_command_passes_default_limit_to_lazy_head(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    calls: list[dict[str, object]] = []

    def _scan_nwb(**kwargs: object) -> _FakeLazyFrame:
        calls.append(kwargs)
        return _FakeLazyFrame(polars.DataFrame({"id": [0, 1, 2, 3, 4, 5]}), calls)

    monkeypatch.setattr(cli_preview.lazyframe, "scan_nwb", _scan_nwb)

    exit_code, stdout, stderr = _run_cli(["preview", "units", str(path)])

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["row_count"] == 5
    assert calls == [
        {
            "disable_progress": True,
            "raise_on_missing": False,
            "source": (path.resolve().as_posix(),),
            "table_path": "units",
        },
        {"head": 5},
        {"collect": True},
    ]


def test_preview_command_accepts_explicit_limit_and_table_alias(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        ["preview", "--limit", "2", "trials", str(local_hdf5_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["limit"] == 2
    assert payload["requested_table"] == "trials"
    assert payload["resolved_table_path"] == "intervals/trials"
    assert payload["row_count"] == 2
    assert [row["condition"] for row in payload["rows"]] == ["A", "B"]


def test_preview_command_returns_oversized_limit_error() -> None:
    exit_code, stdout, stderr = _run_cli(["preview", "--limit", "51", "units"])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "preview_limit_exceeded",
            "details": {
                "max_limit": 50,
                "requested_limit": 51,
            },
            "message": "Preview row limit exceeds the supported maximum; use --limit <= 50.",
        }
    }


@pytest.mark.parametrize(
    ("limit", "requested_limit"),
    [
        ("0", 0),
        ("not-an-int", "not-an-int"),
    ],
)
def test_preview_command_returns_invalid_limit_error(
    limit: str,
    requested_limit: int | str,
) -> None:
    exit_code, stdout, stderr = _run_cli(["preview", "--limit", limit, "units"])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "preview_limit_invalid",
            "details": {
                "default_limit": 5,
                "max_limit": 50,
                "requested_limit": requested_limit,
            },
            "message": "Preview row limit must be a positive integer.",
        }
    }


def test_preview_command_returns_missing_table_error(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        ["preview", "not_a_table", str(local_hdf5_path)]
    )

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "preview_not_found",
            "details": {
                "limit": 5,
                "requested_table": "not_a_table",
                "resolved_count": 1,
                "resolved_table_path": "not_a_table",
                "source": {
                    "dandi": None,
                    "kind": "paths",
                    "local": None,
                    "paths": [
                        {
                            "input": str(local_hdf5_path),
                            "resolved": local_hdf5_path.resolve().as_posix(),
                        }
                    ],
                    "precedence": "command_line_paths",
                    "resolved_count": 1,
                },
            },
            "message": "No NWB table was found for the resolved sources.",
        }
    }


def test_preview_command_supports_table_output(
    local_hdf5_path: pathlib.Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        [
            "preview",
            "--format",
            "table",
            "--limit",
            "2",
            "intervals/trials",
            str(local_hdf5_path),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "condition" in stdout
    assert "start_time" in stdout
    assert "intervals/trials" in stdout
    assert "A" in stdout
    assert "B" in stdout


def test_preview_command_debug_logs_read_planning_and_materialization(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()

    monkeypatch.setattr(
        cli_preview.lazyframe,
        "scan_nwb",
        lambda **kwargs: _FakeLazyFrame(polars.DataFrame({"unit_id": [1, 2]})),
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "preview", "--limit", "2", "units", str(path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["rows"] == [{"unit_id": 1}, {"unit_id": 2}]
    assert "resolving active source" in stderr
    assert "resolved preview source paths" in stderr
    assert "planning NWB table preview" in stderr
    assert "limit=2" in stderr
    assert "disable_progress=True" in stderr
    assert "starting NWB table preview materialization" in stderr
    assert "completed NWB table preview materialization" in stderr
    assert "row_count=2" in stderr
    assert "serializing NWB table preview" in stderr
    assert "elapsed_ms=" in stderr
    assert "DEBUG:" not in stdout


def test_query_command_executes_local_fixture_sql_json(
    local_hdf5_path: pathlib.Path,
) -> None:
    query = (
        'SELECT id, condition FROM "intervals/trials" '
        "WHERE id >= 1 AND id <= 3 ORDER BY id"
    )

    exit_code, stdout, stderr = _run_cli(["query", query, str(local_hdf5_path)])

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["command"] == "query"
    assert payload["query"] == query
    assert payload["limit"] == 100
    assert payload["allow_large"] is False
    assert payload["row_count"] == 3
    assert payload["observed_count"] == 3
    assert payload["truncated"] is False
    assert payload["columns"] == ["id", "condition"]
    assert payload["rows"] == [
        {"condition": "B", "id": 1},
        {"condition": "C", "id": 2},
        {"condition": "D", "id": 3},
    ]
    assert payload["source"] == {
        "dandi": None,
        "kind": "paths",
        "local": None,
        "paths": [
            {
                "input": str(local_hdf5_path),
                "resolved": local_hdf5_path.resolve().as_posix(),
            }
        ],
        "precedence": "command_line_paths",
        "resolved_count": 1,
    }
    assert payload["sql"] == {
        "disable_progress": True,
        "eager": False,
        "exclude_timeseries": False,
        "full_path": True,
        "infer_schema_length": None,
        "rename_general_metadata": True,
    }


def test_query_command_passes_agent_friendly_sql_defaults(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    calls: list[dict[str, object]] = []

    def _get_sql_context(
        nwb_sources: tuple[str, ...],
        **kwargs: object,
    ) -> _FakeQuerySQLContext:
        calls.append({"nwb_sources": nwb_sources, **kwargs})
        return _FakeQuerySQLContext(polars.DataFrame({"id": [1]}), calls)

    monkeypatch.setattr(
        cli_query.conversion,
        "get_sql_context",
        _get_sql_context,
    )

    exit_code, stdout, stderr = _run_cli(
        ["query", "--infer-schema-length", "1", "SELECT id FROM units", str(path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["rows"] == [{"id": 1}]
    assert calls == [
        {
            "disable_progress": True,
            "eager": False,
            "exclude_timeseries": False,
            "full_path": True,
            "infer_schema_length": 1,
            "nwb_sources": (path.resolve().as_posix(),),
            "rename_general_metadata": True,
        },
        {"execute": "SELECT id FROM units", "eager": False},
        {"head": 101},
        {"collect": True},
    ]


def test_query_command_default_cap_failure_returns_machine_readable_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    frame = polars.DataFrame({"id": list(range(101))})
    monkeypatch.setattr(
        cli_query.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeQuerySQLContext(frame),
    )

    exit_code, stdout, stderr = _run_cli(["query", "SELECT id FROM units", str(path)])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "query_row_cap_exceeded",
            "details": {
                "allow_large": False,
                "cap": 100,
                "limit": 100,
                "observed_count": 101,
                "query": "SELECT id FROM units",
                "resolved_count": 1,
                "source": {
                    "dandi": None,
                    "kind": "paths",
                    "local": None,
                    "paths": [
                        {
                            "input": str(path),
                            "resolved": path.resolve().as_posix(),
                        }
                    ],
                    "precedence": "command_line_paths",
                    "resolved_count": 1,
                },
                "sql": {
                    "disable_progress": True,
                    "eager": False,
                    "exclude_timeseries": False,
                    "full_path": True,
                    "infer_schema_length": None,
                    "rename_general_metadata": True,
                },
            },
            "message": (
                "SQL query returned more rows than the configured row cap; "
                "use --allow-large with an explicit --limit to return more rows."
            ),
        }
    }


def test_query_command_allow_large_returns_rows_above_default_cap(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    frame = polars.DataFrame({"id": list(range(101))})
    monkeypatch.setattr(
        cli_query.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeQuerySQLContext(frame),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "query",
            "--allow-large",
            "--limit",
            "101",
            "SELECT id FROM units",
            str(path),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["allow_large"] is True
    assert payload["limit"] == 101
    assert payload["row_count"] == 101
    assert payload["observed_count"] == 101
    assert payload["rows"][0] == {"id": 0}
    assert payload["rows"][-1] == {"id": 100}


def test_query_command_rejects_large_limit_without_bypass() -> None:
    exit_code, stdout, stderr = _run_cli(
        ["query", "--limit", "101", "SELECT id FROM units"]
    )

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "query_limit_exceeded",
            "details": {
                "allow_large": False,
                "default_limit": 100,
                "requested_limit": 101,
            },
            "message": (
                "Query row limit exceeds the default cap; use --allow-large "
                "to request more rows."
            ),
        }
    }


@pytest.mark.parametrize(
    ("limit", "requested_limit"),
    [
        ("0", 0),
        ("not-an-int", "not-an-int"),
    ],
)
def test_query_command_returns_invalid_limit_error(
    limit: str,
    requested_limit: int | str,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        ["query", "--limit", limit, "SELECT id FROM units"]
    )

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "query_limit_invalid",
            "details": {
                "allow_large": False,
                "default_limit": 100,
                "requested_limit": requested_limit,
            },
            "message": "Query row limit must be a positive integer.",
        }
    }


def test_query_command_supports_jsonl_output(
    local_hdf5_path: pathlib.Path,
) -> None:
    query = 'SELECT id, condition FROM "intervals/trials" ORDER BY id LIMIT 2'

    exit_code, stdout, stderr = _run_cli(
        ["query", "--format", "jsonl", query, str(local_hdf5_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert [json.loads(line) for line in stdout.splitlines()] == [
        {"condition": "A", "id": 0},
        {"condition": "B", "id": 1},
    ]


def test_query_command_supports_table_output(
    local_hdf5_path: pathlib.Path,
) -> None:
    query = 'SELECT id, condition FROM "intervals/trials" ORDER BY id LIMIT 2'

    exit_code, stdout, stderr = _run_cli(
        ["query", "--format", "table", query, str(local_hdf5_path)]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "id" in stdout
    assert "condition" in stdout
    assert "A" in stdout
    assert "B" in stdout


def test_query_command_returns_query_failed_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_query.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeQuerySQLContext(
            polars.DataFrame({"id": [1]}),
            error=ValueError("bad SQL"),
        ),
    )

    exit_code, stdout, stderr = _run_cli(["query", "SELECT nope", str(path)])

    assert exit_code == 3
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["error"]["code"] == "query_failed"
    assert payload["error"]["message"] == "SQL query failed."
    assert payload["error"]["details"]["error_type"] == "ValueError"
    assert payload["error"]["details"]["error_message"] == "bad SQL"
    assert payload["error"]["details"]["query"] == "SELECT nope"
    assert payload["error"]["details"]["resolved_count"] == 1


def test_query_command_returns_missing_source_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code, stdout, stderr = _run_cli(["query", "SELECT 1"])

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == {
        "error": {
            "code": "source_not_configured",
            "details": {
                "source": {
                    "dandi": None,
                    "kind": "none",
                    "local": None,
                    "paths": [],
                    "precedence": "none",
                    "resolved_count": 0,
                }
            },
            "message": "No lazynwb source is configured.",
        }
    }


def test_query_command_debug_logs_context_execution_and_materialization(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nwb"
    path.touch()
    monkeypatch.setattr(
        cli_query.conversion,
        "get_sql_context",
        lambda *args, **kwargs: _FakeQuerySQLContext(
            polars.DataFrame({"unit_id": [1, 2]})
        ),
    )

    exit_code, stdout, stderr = _run_cli(
        ["--debug", "query", "--limit", "2", "SELECT unit_id FROM units", str(path)]
    )

    assert exit_code == 0
    assert json.loads(stdout)["rows"] == [{"unit_id": 1}, {"unit_id": 2}]
    assert "resolving active source" in stderr
    assert "resolved query source paths" in stderr
    assert "starting SQL context creation for query" in stderr
    assert "full_path=True" in stderr
    assert "exclude_timeseries=False" in stderr
    assert "rename_general_metadata=True" in stderr
    assert "disable_progress=True" in stderr
    assert "eager=False" in stderr
    assert "completed SQL context creation for query" in stderr
    assert "starting SQL query planning" in stderr
    assert "completed SQL query planning" in stderr
    assert "starting SQL query materialization" in stderr
    assert "completed SQL query materialization" in stderr
    assert "row_count=2" in stderr
    assert "serializing SQL query result" in stderr
    assert "elapsed_ms=" in stderr
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

    def _unexpected_get_dandiset_assets(
        *args: object, **kwargs: object
    ) -> list[object]:
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

    def _unexpected_get_dandiset_assets(
        *args: object, **kwargs: object
    ) -> list[object]:
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

    def _unexpected_get_dandiset_assets(
        *args: object, **kwargs: object
    ) -> list[object]:
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

    exit_code, stdout, stderr = _run_cli(
        ["config", "show", "--config", str(config_path)]
    )

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

    exit_code, stdout, stderr = _run_cli(
        ["config", "show", "--config", str(config_path)]
    )

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


def _load_single_json_object(stdout: str) -> dict[str, typing.Any]:
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(stdout)
    assert stdout[end:] == "\n"
    return typing.cast(dict[str, typing.Any], payload)


class _FakeSQLContext:
    def __init__(self, table_names: tuple[str, ...]) -> None:
        self._table_names = table_names

    def tables(self) -> list[str]:
        return list(self._table_names)


class _FakeQuerySQLContext:
    def __init__(
        self,
        frame: polars.DataFrame,
        calls: list[dict[str, object]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._frame = frame
        self._calls = calls
        self._error = error

    def execute(self, query: str, *, eager: bool | None = None) -> _FakeLazyFrame:
        if self._calls is not None:
            self._calls.append({"execute": query, "eager": eager})
        if self._error is not None:
            raise self._error
        return _FakeLazyFrame(self._frame, self._calls)

    def tables(self) -> list[str]:
        return ["units"]


class _FakeLazyFrame:
    def __init__(
        self,
        frame: polars.DataFrame,
        calls: list[dict[str, object]] | None = None,
    ) -> None:
        self._frame = frame
        self._calls = calls

    def head(self, limit: int) -> _FakeLazyFrame:
        if self._calls is not None:
            self._calls.append({"head": limit})
        return _FakeLazyFrame(self._frame.head(limit), self._calls)

    def collect(self) -> polars.DataFrame:
        if self._calls is not None:
            self._calls.append({"collect": True})
        return self._frame


def _write_config(path: pathlib.Path, content: str) -> None:
    path.write_text(f"{content.strip()}\n", encoding="utf-8")
