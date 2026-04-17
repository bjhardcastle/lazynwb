"""Compute a peri-stimulus time histogram (PSTH) from an NWB `/units` table
and an NWB `/intervals/trials` table (or compatible), vectorised with Polars.

STATUS: PLACEHOLDER.

The intended shape:

  1. Load the units table lazily, filter to units of interest (e.g. brain
     area, quality metric), collect.
  2. Select `spike_times` in `.select()` — the pushdown fetches the array
     only for surviving rows, so no `merge_array_column` is needed here.
  3. Load the trials table, pick an event column (e.g. `stim_start_time`).
  4. For each (unit, trial) pair, select spikes in a window around the
     event and bin them. The efficient way to do this across many
     (unit, trial) pairs is a vectorised range-join using
     `polars_vec_ops.join_between` — much faster than Python-level loops.
  5. Group by (unit, bin) and count.

The maintainer intends to fill in the `polars_vec_ops.join_between` logic
here. Until then, this script only sketches the inputs and the function
signature so callers can wire it up.

Rough signature the filled-in version will expose:

    compute_psth(
        nwb_paths,
        *,
        unit_filter: pl.Expr | None = None,
        event_column: str = "stim_start_time",
        pre_s: float = -0.5,
        post_s: float = 1.5,
        bin_s: float = 0.01,
    ) -> pl.DataFrame      # columns: unit_id, bin_center_s, count, n_trials
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
        description="NWB paths or URLs.",
    )
    units_table: str = pydantic.Field(default="/units")
    trials_table: str = pydantic.Field(default="/intervals/trials")
    event_column: str = pydantic.Field(default="stim_start_time")
    pre_s: float = pydantic.Field(default=-0.5)
    post_s: float = pydantic.Field(default=1.5)
    bin_s: float = pydantic.Field(default=0.01)
    min_presence_ratio: float = pydantic.Field(default=0.9)


def compute_psth(
    nwb_paths: list[str],
    *,
    unit_filter: pl.Expr | None = None,
    trials_table: str = "/intervals/trials",
    units_table: str = "/units",
    event_column: str = "stim_start_time",
    pre_s: float = -0.5,
    post_s: float = 1.5,
    bin_s: float = 0.01,
) -> pl.DataFrame:
    """Return a tidy PSTH: one row per (unit, bin).

    TODO(user): implement with polars_vec_ops.join_between. See module docstring.
    """
    # Step 1-2: filter units, select spike_times (pushdown fetches arrays only
    # for surviving rows — no merge_array_column needed with scan_nwb)
    lf = lazynwb.scan_nwb(nwb_paths, units_table)
    if unit_filter is not None:
        lf = lf.filter(unit_filter)
    units_df = lf.select("unit_id", "spike_times", lazynwb.NWB_PATH_COLUMN_NAME).collect()  # noqa: F841

    # Step 3: load trials
    trials_df = lazynwb.scan_nwb(nwb_paths, trials_table).collect()  # noqa: F841

    # Step 4-5: vectorised range join + bin + count
    # -------------------------------------------------------------------
    # TODO(user): use polars_vec_ops.join_between to assign each spike to
    # the (unit, trial, bin) triple it falls into, then group and count.
    # -------------------------------------------------------------------
    raise NotImplementedError(
        "PSTH computation is not yet implemented; see module docstring."
    )


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    if not settings.paths:
        raise SystemExit("error: pass one or more NWB paths via --paths")

    psth = compute_psth(
        settings.paths,
        unit_filter=pl.col("presence_ratio") > settings.min_presence_ratio,
        trials_table=settings.trials_table,
        units_table=settings.units_table,
        event_column=settings.event_column,
        pre_s=settings.pre_s,
        post_s=settings.post_s,
        bin_s=settings.bin_s,
    )
    print(psth)


if __name__ == "__main__":
    main()
