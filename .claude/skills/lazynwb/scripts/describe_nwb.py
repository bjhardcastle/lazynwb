"""Print a one-shot summary of one or more NWB files: metadata, internal
paths, and per-table schemas.

Usage:
    python describe_nwb.py <path-or-url> [<path-or-url> ...]
    python describe_nwb.py --paths file1.nwb file2.nwb --max-paths 50

Paths may be local files, S3/GCS/Azure URLs, or https URLs.
"""

from __future__ import annotations

import logging
import sys

import pydantic
import pydantic_settings

import lazynwb

logger = logging.getLogger(__name__)


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case="all",
        cli_implicit_flags=True,
    )

    paths: list[str] = pydantic.Field(
        default_factory=list,
        description="NWB file paths or URLs to describe (positional or via --paths).",
    )
    max_paths: int = pydantic.Field(
        default=50,
        description="Truncate the internal-paths listing to this many entries.",
    )
    schemas_for: list[str] = pydantic.Field(
        default_factory=lambda: ["/intervals/trials", "/units"],
        description="Table paths to show unified schemas for, if present.",
    )
    verbose: bool = pydantic.Field(
        default=False,
        description="Enable DEBUG-level logging.",
    )


def _print_metadata(paths: list[str]) -> None:
    logger.info("fetching session/subject metadata for %d file(s)", len(paths))
    meta = lazynwb.get_metadata_df(paths, as_polars=True, disable_progress=True)
    print("# Session / subject metadata")
    print(meta)
    print()


def _print_internal_paths(path: str, limit: int) -> None:
    logger.info("listing internal paths for %s", path)
    paths_map = lazynwb.get_internal_paths(path)
    print(f"# Internal paths in {path}")
    keys = sorted(paths_map)
    for k in keys[:limit]:
        print(f"  {k}")
    if len(keys) > limit:
        print(f"  ... ({len(keys) - limit} more; raise --max-paths to see all)")
    print()


def _print_schema(paths: list[str], table_path: str) -> None:
    try:
        schema = lazynwb.get_table_schema(paths, table_path)
    except Exception as exc:
        logger.info("skipping schema for %s: %s", table_path, exc)
        return
    print(f"# Schema: {table_path}")
    for name, dtype in schema.items():
        print(f"  {name:32s}  {dtype}")
    print()


def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s",
    )
    if not settings.paths:
        print("error: pass one or more NWB paths", file=sys.stderr)
        raise SystemExit(2)

    _print_metadata(settings.paths)
    _print_internal_paths(settings.paths[0], settings.max_paths)
    for table_path in settings.schemas_for:
        _print_schema(settings.paths, table_path)


if __name__ == "__main__":
    main()
