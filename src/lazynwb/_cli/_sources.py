from __future__ import annotations

import logging
import pathlib
import urllib.parse
import urllib.request

import lazynwb._cli._errors as cli_errors

logger = logging.getLogger(__name__)


def _list_explicit_paths(paths: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    logger.debug("resolving %d explicit source path(s)", len(paths))
    resolved_paths: list[dict[str, str]] = []
    missing_paths: list[dict[str, str]] = []

    for path in paths:
        logger.debug("resolving explicit source path: %s", path)
        resolved_path = _resolve_path(path)
        source_path = {
            "input": path,
            "resolved": resolved_path,
        }
        resolved_paths.append(source_path)
        if _is_missing_local_path(path):
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


def _resolve_path(path: str) -> str:
    if not _is_local_path(path):
        return path
    return _local_path(path).resolve(strict=False).as_posix()


def _is_missing_local_path(path: str) -> bool:
    return _is_local_path(path) and not _local_path(path).exists()


def _is_local_path(path: str) -> bool:
    if _is_windows_drive_path(path):
        return True
    scheme = urllib.parse.urlsplit(path).scheme
    return scheme in ("", "file")


def _is_windows_drive_path(path: str) -> bool:
    return len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/")


def _local_path(path: str) -> pathlib.Path:
    parsed = urllib.parse.urlsplit(path)
    if parsed.scheme == "file":
        return pathlib.Path(urllib.request.url2pathname(parsed.path)).expanduser()
    return pathlib.Path(path).expanduser()
