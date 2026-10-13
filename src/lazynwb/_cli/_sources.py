from __future__ import annotations

import concurrent.futures
import dataclasses
import fnmatch
import logging
import pathlib
import time
import typing
import urllib.parse
import urllib.request

import lazynwb._cli._config as cli_config
import lazynwb._cli._errors as cli_errors
import lazynwb.dandi as dandi
import lazynwb.file_io as file_io
import lazynwb.utils as utils

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_GLOB = "**/*.nwb"


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceOverrides:
    paths: tuple[str, ...] = ()
    local_root: str | None = None
    local_glob: str | None = None
    dandi_dandiset_id: str | None = None
    dandi_version: str | None = None
    dandi_path_pattern: str | None = None
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
    override_mode = _direct_source_override_mode(overrides)
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

    if _has_dandi_source(overrides=overrides, config=source_config.dandi):
        precedence = (
            "command_line_dandi"
            if _has_dandi_override(overrides)
            else "config_dandi"
        )
        return _resolve_dandi_source(
            dandiset_id=overrides.dandi_dandiset_id
            if overrides.dandi_dandiset_id is not None
            else source_config.dandi.dandiset_id,
            version=overrides.dandi_version
            if overrides.dandi_version is not None
            else source_config.dandi.version,
            path_pattern=overrides.dandi_path_pattern
            if overrides.dandi_path_pattern is not None
            else source_config.dandi.path_pattern,
            anonymous_s3=source_config.dandi.anonymous_s3
            if overrides.dandi_anonymous_s3 is None
            else overrides.dandi_anonymous_s3,
            precedence=precedence,
        )

    logger.debug("no active lazynwb source is configured")
    return _ResolvedSource(kind="none", precedence="none")


def _paths_for_source(
    resolved_source: _ResolvedSource,
) -> tuple[dict[str, str], ...]:
    if resolved_source.kind in ("paths", "local"):
        return resolved_source.paths
    if resolved_source.kind == "dandi":
        return _list_dandi_source_paths(resolved_source)
    raise cli_errors._CLIError(
        code=cli_errors._ErrorCode.SOURCE_NOT_CONFIGURED,
        details={"source": _source_json_object(resolved_source)},
        exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
        message="No lazynwb source is configured.",
    )


def _source_json_object(
    resolved_source: _ResolvedSource,
    *,
    paths: tuple[typing.Mapping[str, str], ...] | None = None,
) -> dict[str, typing.Any]:
    resolved_paths = resolved_source.paths if paths is None else paths
    return {
        "dandi": resolved_source.dandi,
        "kind": resolved_source.kind,
        "local": resolved_source.local,
        "paths": list(resolved_paths),
        "precedence": resolved_source.precedence,
        "resolved_count": len(resolved_paths),
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


def _direct_source_override_mode(overrides: _SourceOverrides) -> str | None:
    has_paths = bool(overrides.paths)
    has_local = overrides.local_root is not None or overrides.local_glob is not None
    active_modes = [
        name
        for name, is_active in (
            ("paths", has_paths),
            ("local", has_local),
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
    if root is None:
        missing_fields = ["source.local.root"]
        logger.debug("incomplete local source config: missing=%s", missing_fields)
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_CONFIG_INCOMPLETE,
            details={"missing": missing_fields},
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="Local source configuration requires root when glob is provided.",
        )

    resolved_glob = _DEFAULT_LOCAL_GLOB if glob is None else glob
    root_path = _local_path(root, base_dir=base_dir).resolve(strict=False)
    local = {
        "glob": resolved_glob,
        "resolved_root": root_path.as_posix(),
        "root": root,
    }
    paths = (
        _list_local_glob_paths(root=root, root_path=root_path, glob=resolved_glob)
        if discover_local
        else ()
    )
    logger.debug(
        "resolved active local source: precedence=%s root=%s glob=%s discovered=%d",
        precedence,
        root_path,
        resolved_glob,
        len(paths),
    )
    return _ResolvedSource(kind="local", precedence=precedence, paths=paths, local=local)


def _resolve_dandi_source(
    *,
    dandiset_id: str | None,
    version: str | None,
    path_pattern: str | None,
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
        "path_pattern": path_pattern,
        "version": version,
    }
    _apply_dandi_file_io_config(anonymous_s3=anonymous_s3)
    logger.debug(
        "resolved active DANDI source: precedence=%s dandiset_id=%s version=%s "
        "path_pattern=%s anonymous_s3=%s",
        precedence,
        dandiset_id,
        version,
        path_pattern,
        anonymous_s3,
    )
    return _ResolvedSource(kind="dandi", precedence=precedence, dandi=dandi)


def _has_dandi_source(
    *,
    overrides: _SourceOverrides,
    config: cli_config._DandiSourceConfig,
) -> bool:
    return any(
        value is not None
        for value in (
            overrides.dandi_dandiset_id,
            overrides.dandi_version,
            overrides.dandi_path_pattern,
            config.dandiset_id,
            config.version,
            config.path_pattern,
        )
    )


def _has_dandi_override(overrides: _SourceOverrides) -> bool:
    return any(
        value is not None
        for value in (
            overrides.dandi_dandiset_id,
            overrides.dandi_version,
            overrides.dandi_path_pattern,
            overrides.dandi_anonymous_s3,
        )
    )


def _apply_dandi_file_io_config(*, anonymous_s3: bool) -> None:
    previous_anon = file_io.config.anon
    file_io.config.anon = anonymous_s3
    logger.debug(
        "applied DANDI anonymous S3 setting to file I/O config: anonymous_s3=%s "
        "previous_anon=%s",
        anonymous_s3,
        previous_anon,
    )


def _list_dandi_source_paths(
    resolved_source: _ResolvedSource,
) -> tuple[dict[str, str], ...]:
    if resolved_source.dandi is None:
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.INTERNAL_ERROR,
            details={"source": _source_json_object(resolved_source)},
            exit_code=cli_errors._ExitCode.INTERNAL_ERROR,
            message="DANDI source metadata is unavailable.",
        )

    dandiset_id = typing.cast(str, resolved_source.dandi["dandiset_id"])
    version = typing.cast(str | None, resolved_source.dandi["version"])
    path_pattern = typing.cast(str | None, resolved_source.dandi["path_pattern"])
    anonymous_s3 = typing.cast(bool, resolved_source.dandi["anonymous_s3"])
    start_time = time.perf_counter()
    logger.debug(
        "starting DANDI source resolution: dandiset_id=%s version=%s "
        "path_pattern=%s anonymous_s3=%s",
        dandiset_id,
        version,
        path_pattern,
        anonymous_s3,
    )

    assets = _get_dandi_assets(dandiset_id=dandiset_id, version=version)
    logger.debug(
        "fetched DANDI asset metadata: dandiset_id=%s version=%s asset_count=%d",
        dandiset_id,
        version,
        len(assets),
    )
    nwb_assets = _filter_dandi_nwb_assets(assets)
    logger.debug(
        "filtered DANDI assets to NWB paths: before=%d after=%d",
        len(assets),
        len(nwb_assets),
    )
    matched_assets = _filter_dandi_assets_by_pattern(
        nwb_assets,
        path_pattern=path_pattern,
    )
    if path_pattern is not None:
        logger.debug(
            "filtered DANDI assets by path pattern: pattern=%s before=%d after=%d",
            path_pattern,
            len(nwb_assets),
            len(matched_assets),
        )

    if not matched_assets:
        logger.debug(
            "DANDI source resolution found no matching NWB assets: dandiset_id=%s "
            "version=%s path_pattern=%s asset_count=%d nwb_asset_count=%d",
            dandiset_id,
            version,
            path_pattern,
            len(assets),
            len(nwb_assets),
        )
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.SOURCE_DANDI_NO_ASSETS,
            details={
                "asset_count": len(assets),
                "dandiset_id": dandiset_id,
                "nwb_asset_count": len(nwb_assets),
                "path_pattern": path_pattern,
                "version": version,
            },
            exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
            message="No NWB assets matched the active DANDI source.",
        )

    urls = _get_dandi_asset_s3_urls(
        dandiset_id=dandiset_id,
        version=version,
        assets=matched_assets,
    )
    paths = tuple(
        {
            "asset_id": _dandi_asset_id(asset),
            "input": _dandi_asset_path(asset),
            "resolved": url,
        }
        for asset, url in zip(matched_assets, urls, strict=True)
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "resolved DANDI S3 URLs: dandiset_id=%s version=%s url_count=%d "
        "elapsed_ms=%.3f",
        dandiset_id,
        version,
        len(paths),
        elapsed_ms,
    )
    return paths


def _get_dandi_assets(
    *,
    dandiset_id: str,
    version: str | None,
) -> tuple[dict[str, typing.Any], ...]:
    return tuple(dandi._get_dandiset_assets(dandiset_id, version, order="path"))


def _filter_dandi_nwb_assets(
    assets: tuple[dict[str, typing.Any], ...],
) -> tuple[dict[str, typing.Any], ...]:
    return tuple(
        sorted(
            (asset for asset in assets if _is_dandi_nwb_asset(asset)),
            key=_dandi_asset_sort_key,
        )
    )


def _filter_dandi_assets_by_pattern(
    assets: tuple[dict[str, typing.Any], ...],
    *,
    path_pattern: str | None,
) -> tuple[dict[str, typing.Any], ...]:
    if path_pattern is None:
        return assets
    return tuple(
        asset
        for asset in assets
        if fnmatch.fnmatchcase(_dandi_asset_path(asset), path_pattern)
    )


def _is_dandi_nwb_asset(asset: dict[str, typing.Any]) -> bool:
    path = asset.get("path")
    return isinstance(path, str) and path.lower().endswith(".nwb")


def _dandi_asset_sort_key(asset: dict[str, typing.Any]) -> tuple[str, str]:
    return (_dandi_asset_path(asset), _dandi_asset_id(asset))


def _dandi_asset_path(asset: dict[str, typing.Any]) -> str:
    path = asset.get("path")
    if not isinstance(path, str) or not path:
        return _dandi_asset_id(asset)
    return path


def _dandi_asset_id(asset: dict[str, typing.Any]) -> str:
    asset_id = asset.get("asset_id")
    if isinstance(asset_id, str) and asset_id:
        return asset_id
    return ""


def _get_dandi_asset_s3_urls(
    *,
    dandiset_id: str,
    version: str | None,
    assets: tuple[dict[str, typing.Any], ...],
) -> tuple[str, ...]:
    urls: list[str | None] = [None] * len(assets)
    executor = utils.get_threadpool_executor()
    future_to_asset = {
        executor.submit(
            dandi._get_asset_s3_url,
            dandiset_id,
            _dandi_asset_id(asset),
            version,
        ): (index, asset)
        for index, asset in enumerate(assets)
    }
    for future in concurrent.futures.as_completed(future_to_asset):
        index, asset = future_to_asset[future]
        try:
            url = future.result()
        except Exception as exc:
            logger.debug(
                "failed to resolve DANDI S3 URL: dandiset_id=%s version=%s "
                "asset_id=%s path=%s",
                dandiset_id,
                version,
                _dandi_asset_id(asset),
                _dandi_asset_path(asset),
                exc_info=True,
            )
            raise cli_errors._CLIError(
                code=cli_errors._ErrorCode.SOURCE_DANDI_URL_UNAVAILABLE,
                details={
                    "asset_id": _dandi_asset_id(asset),
                    "dandiset_id": dandiset_id,
                    "path": _dandi_asset_path(asset),
                    "version": version,
                },
                exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
                message="A DANDI asset did not provide a usable S3 URL.",
            ) from exc
        if not isinstance(url, str) or not url:
            logger.debug(
                "DANDI S3 URL helper returned an empty URL: dandiset_id=%s "
                "version=%s asset_id=%s path=%s",
                dandiset_id,
                version,
                _dandi_asset_id(asset),
                _dandi_asset_path(asset),
            )
            raise cli_errors._CLIError(
                code=cli_errors._ErrorCode.SOURCE_DANDI_URL_UNAVAILABLE,
                details={
                    "asset_id": _dandi_asset_id(asset),
                    "dandiset_id": dandiset_id,
                    "path": _dandi_asset_path(asset),
                    "version": version,
                },
                exit_code=cli_errors._ExitCode.VALIDATION_ERROR,
                message="A DANDI asset did not provide a usable S3 URL.",
            )
        urls[index] = url
    return tuple(typing.cast(str, url) for url in urls)


def _list_local_glob_paths(
    *,
    root: str,
    root_path: pathlib.Path,
    glob: str,
) -> tuple[dict[str, str], ...]:
    start_time = time.perf_counter()
    logger.debug(
        "starting local source discovery: root=%s resolved_root=%s glob=%s",
        root,
        root_path.as_posix(),
        glob,
    )
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
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "completed local source discovery: root=%s glob=%s match_count=%d elapsed_ms=%.3f",
        root_path.as_posix(),
        glob,
        len(paths),
        elapsed_ms,
    )
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
