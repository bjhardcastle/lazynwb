"""Template: join multiple NWB tables (e.g. trials x units) across one or
more files, with predicate/projection pushdown preserved end-to-end.

**The rule.** When joining tables from the same set of NWB files, the join
key MUST include `lazynwb.NWB_PATH_COLUMN_NAME` (the `_nwb_path` column).
Otherwise trials from session A can silently pair with units from session B,
and nothing complains — you just get wrong numbers.

Usage:
    python multi_table_join.py --paths file1.nwb file2.nwb \\
        --trials-table /intervals/trials \\
        --units-table  /units \\
        --min-presence-ratio 0.9
"""

from __future__ import annotations

import logging

import polars as pl
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
        description="NWB paths or URLs to query.",
    )
    trials_table: str = pydantic.Field(
        default="/intervals/trials",
        description="Internal path to the trials-like table.",
    )
    units_table: str = pydantic.Field(
        default="/units",
        description="Internal path to the units table.",
    )
    min_presence_ratio: float = pydantic.Field(
        default=0.9,
        description="Filter: units.presence_ratio > this.",
    )
    how: str = pydantic.Field(
        default="inner",
        description="Polars join type: inner | left | outer | cross | semi | anti.",
    )
    verbose: bool = pydantic.Field(default=False)


def join_trials_and_units(
    paths: list[str],
    trials_table: str,
    units_table: str,
    min_presence_ratio: float,
    how: str,
) -> pl.DataFrame:
    """Scan both tables lazily, join on `_nwb_path`, filter units by quality,
    collect. Include any array columns (e.g. `spike_times`) in the `.select()`
    call before `.collect()` — the pushdown fetches them only for surviving rows.
    """
    trials = lazynwb.scan_nwb(paths, trials_table)
    units = lazynwb.scan_nwb(paths, units_table)

    joined = trials.join(
        units.filter(pl.col("presence_ratio") > min_presence_ratio),
        on=lazynwb.NWB_PATH_COLUMN_NAME,
        how=how,
        suffix="_unit",
    )

    logger.info("collecting joined LazyFrame")
    return joined.collect()


def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s",
    )
    if not settings.paths:
        raise SystemExit("error: pass one or more NWB paths via --paths")

    df = join_trials_and_units(
        paths=settings.paths,
        trials_table=settings.trials_table,
        units_table=settings.units_table,
        min_presence_ratio=settings.min_presence_ratio,
        how=settings.how,
    )
    print(df)
    print(f"\nshape={df.shape}")


if __name__ == "__main__":
    main()
