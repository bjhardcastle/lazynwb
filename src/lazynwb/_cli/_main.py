from __future__ import annotations

import argparse
import collections.abc
import logging
import sys
import typing

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
        help="list explicitly provided NWB source paths",
    )
    paths_parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        dest="output_format",
        help="output format; defaults to json",
    )
    paths_parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="+",
        help="explicit NWB file or store path",
    )
    paths_parser.set_defaults(_handler=_handle_paths)
    return parser


def _handle_paths(
    args: argparse.Namespace, stdout: typing.TextIO
) -> cli_errors._ExitCode:
    paths = cli_sources._list_explicit_paths(tuple(args.paths))
    if args.output_format == "table":
        cli_formatting._write_source_paths_table(stdout, paths)
    else:
        cli_formatting._write_json(
            stdout,
            cli_formatting._source_paths_json_object(paths),
        )
    return cli_errors._ExitCode.OK


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
