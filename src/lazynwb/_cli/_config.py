from __future__ import annotations

import dataclasses
import logging
import pathlib
import types
import typing

import lazynwb._cli._errors as cli_errors

logger = logging.getLogger(__name__)

_CONFIG_VERSION = 1
_DEFAULT_CONFIG_FILENAME = "lazynwb.toml"

_DEFAULT_CONFIG_TEXT = """version = 1

[source]
paths = []

[source.local]
# root = "data"
# glob = "**/*.nwb"

[source.dandi]
# dandiset_id = "000363"
# version = "draft"
# path_pattern = "sub-*/*.nwb"
anonymous_s3 = true

[commands.paths]
format = "json"
"""


@dataclasses.dataclass(frozen=True, slots=True)
class _LocalSourceConfig:
    root: str | None = None
    glob: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _DandiSourceConfig:
    dandiset_id: str | None = None
    version: str | None = None
    path_pattern: str | None = None
    anonymous_s3: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceConfig:
    paths: tuple[str, ...] = ()
    local: _LocalSourceConfig = dataclasses.field(default_factory=_LocalSourceConfig)
    dandi: _DandiSourceConfig = dataclasses.field(default_factory=_DandiSourceConfig)


@dataclasses.dataclass(frozen=True, slots=True)
class _PathsCommandConfig:
    output_format: str = "json"


@dataclasses.dataclass(frozen=True, slots=True)
class _CommandConfig:
    paths: _PathsCommandConfig = dataclasses.field(default_factory=_PathsCommandConfig)


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjectConfig:
    version: int = _CONFIG_VERSION
    source: _SourceConfig = dataclasses.field(default_factory=_SourceConfig)
    commands: _CommandConfig = dataclasses.field(default_factory=_CommandConfig)


@dataclasses.dataclass(frozen=True, slots=True)
class _LoadedConfig:
    path: pathlib.Path
    exists: bool
    project: _ProjectConfig

    @property
    def base_dir(self) -> pathlib.Path:
        return self.path.parent


def _load_project_config(
    config_path: str | None,
    *,
    cwd: pathlib.Path | None = None,
) -> _LoadedConfig:
    working_directory = pathlib.Path.cwd() if cwd is None else cwd
    resolved_path, explicit_path = _resolve_config_path(config_path, cwd=working_directory)
    logger.debug(
        "resolved lazynwb config path: path=%s explicit=%s",
        resolved_path.as_posix(),
        explicit_path,
    )

    if not resolved_path.exists():
        if explicit_path:
            logger.debug("explicit lazynwb config path does not exist: %s", resolved_path)
            raise cli_errors._CLIError(
                code=cli_errors._ErrorCode.CONFIG_NOT_FOUND,
                details={"path": resolved_path.as_posix()},
                exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
                message="The requested lazynwb config file does not exist.",
            )
        logger.debug("no lazynwb config found; using built-in defaults")
        return _LoadedConfig(
            path=resolved_path,
            exists=False,
            project=_ProjectConfig(),
        )

    logger.debug("loading lazynwb config from %s", resolved_path)
    data = _read_toml_config(resolved_path)
    project = _parse_project_config(data, path=resolved_path)
    logger.debug("loaded lazynwb config version %d", project.version)
    return _LoadedConfig(path=resolved_path, exists=True, project=project)


def _read_toml_config(path: pathlib.Path) -> dict[str, object]:
    toml_reader = _toml_reader(path)
    try:
        with path.open("rb") as stream:
            data = typing.cast(dict[str, object], toml_reader.load(stream))
    except toml_reader.TOMLDecodeError as exc:
        logger.debug("failed to parse lazynwb config: %s", exc)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.CONFIG_PARSE_ERROR,
            details={"path": path.as_posix()},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="The lazynwb config file is not valid TOML.",
        ) from exc
    return data


def _toml_reader(path: pathlib.Path) -> types.ModuleType:
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli
        except ModuleNotFoundError as exc:
            logger.debug("tomli is unavailable for Python 3.10 TOML config parsing")
            raise cli_errors._CLIError(
                code=cli_errors._ErrorCode.CONFIG_PARSE_ERROR,
                details={"path": path.as_posix(), "parser": "tomli"},
                exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
                message="TOML config parsing on Python 3.10 requires tomli.",
            ) from exc
        return tomli


def _initialize_project_config(
    config_path: str | None,
    *,
    force: bool,
    cwd: pathlib.Path | None = None,
) -> pathlib.Path:
    working_directory = pathlib.Path.cwd() if cwd is None else cwd
    if config_path is None:
        path = working_directory / _DEFAULT_CONFIG_FILENAME
    else:
        path = _as_path(config_path, cwd=working_directory)
    resolved_path = path.resolve(strict=False)
    logger.debug("initializing lazynwb config at %s", resolved_path)

    if resolved_path.exists() and not force:
        logger.debug("refusing to overwrite existing lazynwb config: %s", resolved_path)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.CONFIG_ALREADY_EXISTS,
            details={"path": resolved_path.as_posix()},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="The lazynwb config file already exists.",
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(_DEFAULT_CONFIG_TEXT, encoding="utf-8")
    logger.debug("wrote lazynwb config template to %s", resolved_path)
    return resolved_path


def _config_json_object(loaded_config: _LoadedConfig) -> dict[str, typing.Any]:
    project = loaded_config.project
    return {
        "exists": loaded_config.exists,
        "path": loaded_config.path.as_posix(),
        "version": project.version,
    }


def _commands_json_object(commands: _CommandConfig) -> dict[str, typing.Any]:
    return {
        "paths": {
            "format": commands.paths.output_format,
        }
    }


def _resolve_config_path(
    config_path: str | None,
    *,
    cwd: pathlib.Path,
) -> tuple[pathlib.Path, bool]:
    if config_path is not None:
        return _as_path(config_path, cwd=cwd).resolve(strict=False), True

    for directory in (cwd, *cwd.parents):
        candidate = directory / _DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return candidate.resolve(strict=False), False

    return (cwd / _DEFAULT_CONFIG_FILENAME).resolve(strict=False), False


def _as_path(path: str, *, cwd: pathlib.Path) -> pathlib.Path:
    expanded_path = pathlib.Path(path).expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return cwd / expanded_path


def _parse_project_config(
    data: dict[str, object],
    *,
    path: pathlib.Path,
) -> _ProjectConfig:
    _reject_unknown_keys(
        data,
        allowed_keys=("version", "source", "commands"),
        path=path,
        table_name="",
    )

    version = data.get("version", _CONFIG_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        _raise_invalid_config(
            path=path,
            field="version",
            message="Config version must be an integer.",
        )
    if version != _CONFIG_VERSION:
        _raise_invalid_config(
            path=path,
            field="version",
            message=f"Unsupported lazynwb config version: {version}.",
        )

    source = _parse_source_config(data.get("source", {}), path=path)
    commands = _parse_command_config(data.get("commands", {}), path=path)
    return _ProjectConfig(version=version, source=source, commands=commands)


def _parse_source_config(value: object, *, path: pathlib.Path) -> _SourceConfig:
    table = _expect_table(value, path=path, field="source")
    _reject_unknown_keys(
        table,
        allowed_keys=("paths", "local", "dandi"),
        path=path,
        table_name="source",
    )

    paths = _expect_str_tuple(table.get("paths", []), path=path, field="source.paths")
    local = _parse_local_source_config(table.get("local", {}), path=path)
    dandi = _parse_dandi_source_config(table.get("dandi", {}), path=path)
    return _SourceConfig(paths=paths, local=local, dandi=dandi)


def _parse_local_source_config(
    value: object,
    *,
    path: pathlib.Path,
) -> _LocalSourceConfig:
    table = _expect_table(value, path=path, field="source.local")
    _reject_unknown_keys(
        table,
        allowed_keys=("root", "glob"),
        path=path,
        table_name="source.local",
    )
    return _LocalSourceConfig(
        root=_expect_optional_str(table.get("root"), path=path, field="source.local.root"),
        glob=_expect_optional_str(table.get("glob"), path=path, field="source.local.glob"),
    )


def _parse_dandi_source_config(
    value: object,
    *,
    path: pathlib.Path,
) -> _DandiSourceConfig:
    table = _expect_table(value, path=path, field="source.dandi")
    _reject_unknown_keys(
        table,
        allowed_keys=("dandiset_id", "version", "path_pattern", "anonymous_s3"),
        path=path,
        table_name="source.dandi",
    )
    anonymous_s3 = table.get("anonymous_s3", True)
    if not isinstance(anonymous_s3, bool):
        _raise_invalid_config(
            path=path,
            field="source.dandi.anonymous_s3",
            message="DANDI anonymous_s3 must be a boolean.",
        )
    return _DandiSourceConfig(
        dandiset_id=_expect_optional_str(
            table.get("dandiset_id"),
            path=path,
            field="source.dandi.dandiset_id",
        ),
        version=_expect_optional_str(table.get("version"), path=path, field="source.dandi.version"),
        path_pattern=_expect_optional_str(
            table.get("path_pattern"),
            path=path,
            field="source.dandi.path_pattern",
        ),
        anonymous_s3=anonymous_s3,
    )


def _parse_command_config(value: object, *, path: pathlib.Path) -> _CommandConfig:
    table = _expect_table(value, path=path, field="commands")
    _reject_unknown_keys(table, allowed_keys=("paths",), path=path, table_name="commands")
    paths = _parse_paths_command_config(table.get("paths", {}), path=path)
    return _CommandConfig(paths=paths)


def _parse_paths_command_config(
    value: object,
    *,
    path: pathlib.Path,
) -> _PathsCommandConfig:
    table = _expect_table(value, path=path, field="commands.paths")
    _reject_unknown_keys(
        table,
        allowed_keys=("format",),
        path=path,
        table_name="commands.paths",
    )
    output_format = table.get("format", "json")
    if output_format not in ("json", "table"):
        _raise_invalid_config(
            path=path,
            field="commands.paths.format",
            message="commands.paths.format must be 'json' or 'table'.",
        )
    return _PathsCommandConfig(output_format=output_format)


def _expect_table(
    value: object,
    *,
    path: pathlib.Path,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        _raise_invalid_config(path=path, field=field, message=f"{field} must be a TOML table.")
    return typing.cast(dict[str, object], value)


def _expect_str_tuple(
    value: object,
    *,
    path: pathlib.Path,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        _raise_invalid_config(path=path, field=field, message=f"{field} must be a list of strings.")
    paths: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _raise_invalid_config(
                path=path,
                field=f"{field}[{index}]",
                message=f"{field} must be a list of strings.",
            )
        paths.append(item)
    return tuple(paths)


def _expect_optional_str(
    value: object,
    *,
    path: pathlib.Path,
    field: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _raise_invalid_config(path=path, field=field, message=f"{field} must be a string.")
    if value == "":
        _raise_invalid_config(path=path, field=field, message=f"{field} must not be empty.")
    return value


def _reject_unknown_keys(
    values: dict[str, object],
    *,
    allowed_keys: tuple[str, ...],
    path: pathlib.Path,
    table_name: str,
) -> None:
    allowed = set(allowed_keys)
    unknown_keys = sorted(key for key in values if key not in allowed)
    if not unknown_keys:
        return

    prefix = f"{table_name}." if table_name else ""
    _raise_invalid_config(
        path=path,
        field=f"{prefix}{unknown_keys[0]}",
        message=f"Unknown lazynwb config key: {prefix}{unknown_keys[0]}.",
    )


def _raise_invalid_config(*, path: pathlib.Path, field: str, message: str) -> typing.NoReturn:
    logger.debug("invalid lazynwb config field %s in %s: %s", field, path, message)
    raise cli_errors._CLIError(
        code=cli_errors._ErrorCode.CONFIG_INVALID,
        details={"field": field, "path": path.as_posix()},
        exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
        message=message,
    )
