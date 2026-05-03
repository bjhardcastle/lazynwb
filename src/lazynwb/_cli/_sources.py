from __future__ import annotations

import dataclasses
import logging
import pathlib
import typing
import urllib.parse
import urllib.request

import lazynwb._cli._config as cli_config
import lazynwb._cli._errors as cli_errors

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceOverrides:
    paths: tuple[str, ...] = ()
    local_root: str | None = None
    local_glob: str | None = None
    dandi_dandiset_id: str | None = None
    dandi_version: str | None = None
    dandi_anonymous_s3: bool | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ResolvedSource:
    kind: str
    precedence: str
    paths: tuple[dict[str, str], ...] = ()
    local: dict[str, str] | None = None
    dandi: dict[str, typing.Any] | None = None


def _resolve_source(
    loaded_config: cli_config._LoadedConfig,
    overrides: _SourceOverrides,
    *,
    validate_paths: bool,
    discover_local: bool,
) -> _ResolvedSource:
    logger.debug(
        "resolving active source: config_exists=%s override_paths=%d",
        loaded_config.exists,
        len(overrides.paths),
    )
    override_mode = _source_override_mode(overrides)
    if override_mode == "paths":
        return _resolve_path_source(
            paths=overrides.paths,
            precedence="command_line_paths",
            base_dir=None,
            validate_paths=validate_paths,
        )
    if override_mode == "local":
        root = (
            overrides.local_root
            if overrides.local_root is not None
            else loaded_config.project.source.local.root
        )
        return _resolve_local_source(
            root=root,
            glob=overrides.local_glob
            if overrides.local_glob is not None
            else loaded_config.project.source.local.glob,
            precedence="command_line_local",
            base_dir=None if overrides.local_root is not None else loaded_config.base_dir,
            discover_local=discover_local,
        )
    if override_mode == "dandi":
        return _resolve_dandi_source(
            dandiset_id=overrides.dandi_dandiset_id
            if overrides.dandi_dandiset_id is not None
            else loaded_config.project.source.dandi.dandiset_id,
            version=overrides.dandi_version
            if overrides.dandi_version is not None
            else loaded_config.project.source.dandi.version,
            anonymous_s3=overrides.dandi_anonymous_s3
            if overrides.dandi_anonymous_s3 is not None
            else loaded_config.project.source.dandi.anonymous_s3,
            precedence="command_line_dandi",
        )

    source_config = loaded_config.project.source
    if source_config.paths:
        return _resolve_path_source(
            paths=source_config.paths,
            precedence="config_paths",
            base_dir=loaded_config.base_dir,
            validate_paths=validate_paths,
        )

    if source_config.local.root is not None or source_config.local.glob is not None:
        return _resolve_local_source(
            root=source_config.local.root,
            glob=source_config.local.glob,
            precedence="config_local",
            base_dir=loaded_config.base_dir,
            discover_local=discover_local,
        )

    if source_config.dandi.dandiset_id is not None or source_config.dandi.version is not None:
        return _resolve_dandi_source(
            dandiset_id=source_config.dandi.dandiset_id,
            version=source_config.dandi.version,
            anonymous_s3=source_config.dandi.anonymous_s3
            if overrides.dandi_anonymous_s3 is None
            else overrides.dandi_anonymous_s3,
            precedence="config_dandi",
        )

    logger.debug("no active lazynwb source is configured")
    return _ResolvedSource(kind="none", precedence="none")


def _paths_for_source(
    resolved_source: _ResolvedSource,
) -> tuple[dict[str, str], ...]:
    if resolved_source.kind in ("paths", "local"):
        return resolved_source.paths
    if resolved_source.kind == "dandi":
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_PATHS_UNAVAILABLE,
            details={"source": _source_json_object(resolved_source)},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="The active DANDI source cannot be expanded to paths without remote lookup.",
        )
    raise cli_errors._CLIError(
        code=cli_errors._ErrorCode.SOURCE_NOT_CONFIGURED,
        details={"source": _source_json_object(resolved_source)},
        exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
        message="No lazynwb source is configured.",
    )


def _source_json_object(resolved_source: _ResolvedSource) -> dict[str, typing.Any]:
    return {
        "dandi": resolved_source.dandi,
        "kind": resolved_source.kind,
        "local": resolved_source.local,
        "paths": list(resolved_source.paths),
        "precedence": resolved_source.precedence,
    }


def _list_explicit_paths(
    paths: tuple[str, ...],
    *,
    base_dir: pathlib.Path | None = None,
    validate_paths: bool = True,
) -> tuple[dict[str, str], ...]:
    logger.debug("resolving %d explicit source path(s)", len(paths))
    resolved_paths: list[dict[str, str]] = []
    missing_paths: list[dict[str, str]] = []

    for path in paths:
        logger.debug("resolving explicit source path: %s", path)
        resolved_path = _resolve_path(path, base_dir=base_dir)
        source_path = {
            "input": path,
            "resolved": resolved_path,
        }
        resolved_paths.append(source_path)
        if validate_paths and _is_missing_local_path(path, base_dir=base_dir):
            logger.debug("explicit source path is missing: %s", resolved_path)
            missing_paths.append(source_path)
        else:
            logger.debug("explicit source path resolved: %s", resolved_path)

    if missing_paths:
        logger.debug(
            "explicit source path validation failed for %d path(s)", len(missing_paths)
        )
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_PATH_NOT_FOUND,
            details={"paths": missing_paths},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="One or more explicit source paths do not exist.",
        )

    return tuple(resolved_paths)


def _source_override_mode(overrides: _SourceOverrides) -> str | None:
    has_paths = bool(overrides.paths)
    has_local = overrides.local_root is not None or overrides.local_glob is not None
    has_dandi = overrides.dandi_dandiset_id is not None or overrides.dandi_version is not None
    active_modes = [
        name
        for name, is_active in (
            ("paths", has_paths),
            ("local", has_local),
            ("dandi", has_dandi),
        )
        if is_active
    ]
    if len(active_modes) > 1:
        logger.debug("source override conflict: %s", active_modes)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_CONFIG_CONFLICT,
            details={"sources": active_modes},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="Command-line source overrides are mutually incompatible.",
        )
    if not active_modes:
        return None
    return active_modes[0]


def _resolve_path_source(
    *,
    paths: tuple[str, ...],
    precedence: str,
    base_dir: pathlib.Path | None,
    validate_paths: bool,
) -> _ResolvedSource:
    resolved_paths = _list_explicit_paths(
        paths,
        base_dir=base_dir,
        validate_paths=validate_paths,
    )
    logger.debug(
        "resolved active path source: precedence=%s paths=%d",
        precedence,
        len(resolved_paths),
    )
    return _ResolvedSource(kind="paths", precedence=precedence, paths=resolved_paths)


def _resolve_local_source(
    *,
    root: str | None,
    glob: str | None,
    precedence: str,
    base_dir: pathlib.Path | None,
    discover_local: bool,
) -> _ResolvedSource:
    if root is None or glob is None:
        missing_fields = []
        if root is None:
            missing_fields.append("source.local.root")
        if glob is None:
            missing_fields.append("source.local.glob")
        logger.debug("incomplete local source config: missing=%s", missing_fields)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_CONFIG_INCOMPLETE,
            details={"missing": missing_fields},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="Local source configuration requires both root and glob.",
        )

    root_path = _local_path(root, base_dir=base_dir).resolve(strict=False)
    local = {
        "glob": glob,
        "resolved_root": root_path.as_posix(),
        "root": root,
    }
    paths = (
        _list_local_glob_paths(root=root, root_path=root_path, glob=glob)
        if discover_local
        else ()
    )
    logger.debug(
        "resolved active local source: precedence=%s root=%s glob=%s discovered=%d",
        precedence,
        root_path,
        glob,
        len(paths),
    )
    return _ResolvedSource(kind="local", precedence=precedence, paths=paths, local=local)


def _resolve_dandi_source(
    *,
    dandiset_id: str | None,
    version: str | None,
    anonymous_s3: bool,
    precedence: str,
) -> _ResolvedSource:
    if dandiset_id is None:
        logger.debug("incomplete DANDI source config: missing source.dandi.dandiset_id")
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_CONFIG_INCOMPLETE,
            details={"missing": ["source.dandi.dandiset_id"]},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="DANDI source configuration requires dandiset_id.",
        )
    dandi = {
        "anonymous_s3": anonymous_s3,
        "dandiset_id": dandiset_id,
        "version": version,
    }
    logger.debug(
        "resolved active DANDI source: precedence=%s dandiset_id=%s version=%s anonymous_s3=%s",
        precedence,
        dandiset_id,
        version,
        anonymous_s3,
    )
    return _ResolvedSource(kind="dandi", precedence=precedence, dandi=dandi)


def _list_local_glob_paths(
    *,
    root: str,
    root_path: pathlib.Path,
    glob: str,
) -> tuple[dict[str, str], ...]:
    if not root_path.exists():
        logger.debug("local source root does not exist: %s", root_path)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_ROOT_NOT_FOUND,
            details={"root": root, "resolved_root": root_path.as_posix()},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="The configured local source root does not exist.",
        )
    paths: list[dict[str, str]] = []
    for matched_path in sorted(root_path.glob(glob), key=lambda value: value.as_posix()):
        paths.append(
            {
                "input": _local_glob_input(root=root, root_path=root_path, path=matched_path),
                "resolved": matched_path.resolve(strict=False).as_posix(),
            }
        )
    logger.debug("local source glob matched %d path(s)", len(paths))
    return tuple(paths)


def _local_glob_input(*, root: str, root_path: pathlib.Path, path: pathlib.Path) -> str:
    relative_path = path.relative_to(root_path).as_posix()
    if root in ("", "."):
        return relative_path
    return f"{root.rstrip('/')}/{relative_path}"


def _resolve_path(path: str, *, base_dir: pathlib.Path | None = None) -> str:
    if not _is_local_path(path):
        return path
    return _local_path(path, base_dir=base_dir).resolve(strict=False).as_posix()


def _is_missing_local_path(path: str, *, base_dir: pathlib.Path | None = None) -> bool:
    return _is_local_path(path) and not _local_path(path, base_dir=base_dir).exists()


def _is_local_path(path: str) -> bool:
    if _is_windows_drive_path(path):
        return True
    scheme = urllib.parse.urlsplit(path).scheme
    return scheme in ("", "file")


def _is_windows_drive_path(path: str) -> bool:
    return len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/")


def _local_path(path: str, *, base_dir: pathlib.Path | None = None) -> pathlib.Path:
    parsed = urllib.parse.urlsplit(path)
    if parsed.scheme == "file":
        return pathlib.Path(urllib.request.url2pathname(parsed.path)).expanduser()
    local_path = pathlib.Path(path).expanduser()
    if (
        base_dir is not None
        and not local_path.is_absolute()
        and not _is_windows_drive_path(path)
    ):
        return base_dir / local_path
    return local_path
