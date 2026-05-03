from __future__ import annotations

import argparse
import collections.abc
import logging
import sys
import typing

import lazynwb._cli._config as cli_config
import lazynwb._cli._errors as cli_errors
import lazynwb._cli._formatting as cli_formatting
import lazynwb._cli._sources as cli_sources

logger = logging.getLogger(__name__)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> typing.NoReturn:
        raise cli_errors._CLIError(
            code=cli_errors._ErrorCode.USAGE_ERROR,
            details={"usage": self.format_usage().strip()},
            exit_code=cli_errors._ExitCode.USAGE_ERROR,
            message=message,
        )


def main(
    argv: collections.abc.Sequence[str] | None = None,
    stdout: typing.TextIO | None = None,
    stderr: typing.TextIO | None = None,
) -> int:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = _build_parser()

    try:
        args = parser.parse_args(argv)
        _configure_logging(debug=args.debug, stderr=stderr)
        logger.debug("parsed CLI arguments: %s", vars(args))
        exit_code = args._handler(args, stdout)
        logger.debug("CLI command completed with exit code %d", exit_code)
        return int(exit_code)
    except cli_errors._CLIError as exc:
        _configure_logging(debug=_debug_requested(argv), stderr=stderr)
        logger.debug(
            "CLI command failed with stable error code %s and exit code %d",
            exc.code.value,
            exc.exit_code,
        )
        cli_formatting._write_json(stdout, exc._to_json_object())
        return int(exc.exit_code)
    except Exception:
        _configure_logging(debug=_debug_requested(argv), stderr=stderr)
        logger.debug("CLI command failed unexpectedly", exc_info=True)
        cli_formatting._write_json(
            stdout,
            cli_errors._CLIError(
                code=cli_errors._ErrorCode.INTERNAL_ERROR,
                details={},
                exit_code=cli_errors._ExitCode.INTERNAL_ERROR,
                message="An unexpected lazynwb CLI error occurred.",
            )._to_json_object(),
        )
        return int(cli_errors._ExitCode.INTERNAL_ERROR)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="lazynwb",
        description="Inspect NWB sources with deterministic machine-readable output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write debug logs to stderr",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser(
        "paths",
        help="list active NWB source paths",
    )
    _add_config_argument(paths_parser)
    paths_parser.add_argument(
        "--format",
        choices=("json", "table"),
        default=None,
        dest="output_format",
        help="output format; defaults to the config value or json",
    )
    _add_source_override_arguments(paths_parser, include_paths=False)
    paths_parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="*",
        help="explicit NWB file or store path",
    )
    paths_parser.set_defaults(_handler=_handle_paths)

    config_parser = subparsers.add_parser(
        "config",
        help="initialize or show project-local CLI configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )

    config_init_parser = config_subparsers.add_parser(
        "init",
        help="initialize a project-local lazynwb TOML config",
    )
    _add_config_argument(config_init_parser)
    config_init_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing lazynwb config",
    )
    config_init_parser.set_defaults(_handler=_handle_config_init)

    config_show_parser = config_subparsers.add_parser(
        "show",
        help="show the active resolved lazynwb config as JSON",
    )
    _add_config_argument(config_show_parser)
    _add_source_override_arguments(config_show_parser, include_paths=True)
    config_show_parser.set_defaults(_handler=_handle_config_show)
    return parser


def _handle_paths(
    args: argparse.Namespace, stdout: typing.TextIO
) -> cli_errors._ExitCode:
    loaded_config = cli_config._load_project_config(args.config)
    resolved_source = cli_sources._resolve_source(
        loaded_config,
        _source_overrides_from_args(args, paths=tuple(args.paths)),
        validate_paths=True,
        discover_local=True,
    )
    paths = cli_sources._paths_for_source(resolved_source)
    output_format = args.output_format or loaded_config.project.commands.paths.output_format
    if output_format == "table":
        cli_formatting._write_source_paths_table(stdout, paths)
    else:
        cli_formatting._write_json(
            stdout,
            cli_formatting._source_paths_json_object(paths, resolved_source),
        )
    return cli_errors._ExitCode.OK


def _handle_config_init(
    args: argparse.Namespace, stdout: typing.TextIO
) -> cli_errors._ExitCode:
    path = cli_config._initialize_project_config(args.config, force=args.force)
    cli_formatting._write_json(
        stdout,
        cli_formatting._config_init_json_object(path.as_posix()),
    )
    return cli_errors._ExitCode.OK


def _handle_config_show(
    args: argparse.Namespace, stdout: typing.TextIO
) -> cli_errors._ExitCode:
    loaded_config = cli_config._load_project_config(args.config)
    resolved_source = cli_sources._resolve_source(
        loaded_config,
        _source_overrides_from_args(args, paths=tuple(args.source_paths or ())),
        validate_paths=True,
        discover_local=True,
    )
    cli_formatting._write_json(
        stdout,
        cli_formatting._config_show_json_object(loaded_config, resolved_source),
    )
    return cli_errors._ExitCode.OK


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="path to a lazynwb TOML config; defaults to lazynwb.toml in the project tree",
    )


def _add_source_override_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_paths: bool,
) -> None:
    if include_paths:
        parser.add_argument(
            "--path",
            action="append",
            default=None,
            dest="source_paths",
            help="explicit NWB file or store path; may be provided more than once",
        )
    parser.add_argument(
        "--root",
        default=None,
        dest="local_root",
        help="local source root directory",
    )
    parser.add_argument(
        "--glob",
        default=None,
        dest="local_glob",
        help="local source glob relative to --root",
    )
    parser.add_argument(
        "--dandiset-id",
        default=None,
        dest="dandi_dandiset_id",
        help="DANDI dandiset identifier",
    )
    parser.add_argument(
        "--dandi-version",
        default=None,
        dest="dandi_version",
        help="DANDI dandiset version; omit to defer latest-version resolution",
    )
    parser.add_argument(
        "--anonymous-s3",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="dandi_anonymous_s3",
        help="use anonymous S3 access for DANDI assets",
    )


def _source_overrides_from_args(
    args: argparse.Namespace,
    *,
    paths: tuple[str, ...],
) -> cli_sources._SourceOverrides:
    return cli_sources._SourceOverrides(
        paths=paths,
        local_root=getattr(args, "local_root", None),
        local_glob=getattr(args, "local_glob", None),
        dandi_dandiset_id=getattr(args, "dandi_dandiset_id", None),
        dandi_version=getattr(args, "dandi_version", None),
        dandi_anonymous_s3=getattr(args, "dandi_anonymous_s3", None),
    )


def _configure_logging(*, debug: bool, stderr: typing.TextIO) -> None:
    cli_logger = logging.getLogger("lazynwb._cli")
    cli_logger.handlers.clear()
    handler = logging.StreamHandler(stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    cli_logger.addHandler(handler)
    cli_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    cli_logger.propagate = False


def _debug_requested(argv: collections.abc.Sequence[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    return "--debug" in args


if __name__ == "__main__":
    raise SystemExit(main())
