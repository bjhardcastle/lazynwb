"""Benchmark filtered remote /units reads through lazynwb.scan_nwb.

Each repeat uses a new empty catalog cache and clears lazynwb process caches before
building and collecting the LazyFrame. This makes the measured path cold with
respect to lazynwb's schema/catalog caches.

Examples
--------
    uv run python benchmarks/filtered_units_cold_cache_benchmark.py
    uv run python benchmarks/filtered_units_cold_cache_benchmark.py --repeats 3
    uv run python benchmarks/filtered_units_cold_cache_benchmark.py --row-mode all
    uv run python benchmarks/filtered_units_cold_cache_benchmark.py \
        --row-mode structure --structure CA3
    uv run python benchmarks/filtered_units_cold_cache_benchmark.py \
        --row-mode filtered \
        --filter-column firing_rate --operator gt --filter-value 5 \
        --select unit_id --select structure --select firing_rate
"""

from __future__ import annotations

import argparse
import collections.abc
import dataclasses
import gc
import json
import logging
import os
import pathlib
import statistics
import tempfile
import time

import polars as pl

import lazynwb.attrs as attrs
import lazynwb.file_io as file_io
import lazynwb.lazyframe as lazyframe
import lazynwb.tables as tables

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SOURCE = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/" "620263_2022-07-26.nwb"
)
_TABLE_PATH = "/units"
_DEFAULT_SELECT_COLUMNS = (
    "unit_id",
    "structure",
    "firing_rate",
    "snr",
    "default_qc",
    "amplitude",
    "spike_times",
)
_INTERNAL_SELECT_COLUMNS = ("_nwb_path", "_table_index")
_OPERATORS = ("eq", "ne", "gt", "ge", "lt", "le", "is-null", "is-not-null")
_ROW_MODES = ("structure", "all", "filtered")
_DEFAULT_STRUCTURE = "CA3"


@dataclasses.dataclass(frozen=True, slots=True)
class _RunMetric:
    repeat: int
    source: str
    table_path: str
    cache_path: str
    row_mode: str
    filter_column: str
    operator: str
    filter_value: object
    select_columns: tuple[str, ...]
    scan_seconds: float
    collect_seconds: float
    total_seconds: float
    rows: int
    columns: int
    estimated_size_bytes: int

    def to_json_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def main(argv: collections.abc.Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    file_io.config.anon = args.anon
    os.environ.setdefault("AWS_REGION", args.aws_region)

    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="lazynwb-filtered-units-cold-") as tmp:
            metrics = _run_benchmark(args, pathlib.Path(tmp))
    else:
        metrics = _run_benchmark(args, args.cache_dir)

    _print_metrics(metrics)
    _print_summary(metrics)
    _write_json(args.json_output, metrics)


def _parse_args(argv: collections.abc.Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a filtered remote /units table read with scan_nwb using "
            "a cold lazynwb catalog cache for every repeat."
        )
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("LAZYNWB_FILTERED_UNITS_BENCH_SOURCE", _DEFAULT_SOURCE),
        help="Remote NWB URL/path. Defaults to a public dynamic-routing S3 object.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=_env_int("LAZYNWB_FILTERED_UNITS_BENCH_REPEATS", 1),
        help="Number of cold-cache repeats.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_FILTERED_UNITS_BENCH_CACHE_DIR"),
        help="Parent directory for per-repeat cold catalog caches.",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_FILTERED_UNITS_BENCH_JSON"),
        help="Optional JSON metrics output path.",
    )
    parser.add_argument(
        "--row-mode",
        choices=_ROW_MODES,
        default=os.environ.get("LAZYNWB_FILTERED_UNITS_BENCH_ROW_MODE", "structure"),
        help=(
            "Rows to collect: 'structure' filters to --structure, 'all' reads the "
            "full table, and 'filtered' uses --filter-column/--operator/--filter-value."
        ),
    )
    parser.add_argument(
        "--structure",
        default=os.environ.get(
            "LAZYNWB_FILTERED_UNITS_BENCH_STRUCTURE",
            _DEFAULT_STRUCTURE,
        ),
        help=(
            "Structure value used when --row-mode=structure. The default source has "
            f"{_DEFAULT_STRUCTURE!r} on 3 rows."
        ),
    )
    parser.add_argument(
        "--filter-column",
        default="default_qc",
        help="Column used for the pushed-down predicate when --row-mode=filtered.",
    )
    parser.add_argument(
        "--operator",
        choices=_OPERATORS,
        default="eq",
        help="Predicate operator.",
    )
    parser.add_argument(
        "--filter-value",
        default="true",
        help="Predicate literal. Ignored for is-null and is-not-null.",
    )
    parser.add_argument(
        "--select",
        action="append",
        dest="select_columns",
        help=(
            "Column to project after filtering. Repeat to override the default "
            f"projection: {', '.join(_DEFAULT_SELECT_COLUMNS)}."
        ),
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Also select _nwb_path and _table_index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("LAZYNWB_FILTERED_UNITS_BENCH_LIMIT", 0),
        help=(
            "Optional row limit after row-mode filtering. 0 means collect all rows "
            "selected by --row-mode."
        ),
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Pass low_memory=True to scan_nwb.",
    )
    parser.add_argument(
        "--anon",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("LAZYNWB_FILTERED_UNITS_BENCH_ANON", True),
        help="Use anonymous object-store access. Defaults to true.",
    )
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="AWS region to set when AWS_REGION is unset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging, including lazynwb cache/materialization logs.",
    )
    return parser.parse_args(argv)


def _run_benchmark(
    args: argparse.Namespace,
    cache_root: pathlib.Path,
) -> tuple[_RunMetric, ...]:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    cache_root.mkdir(parents=True, exist_ok=True)
    select_columns = _select_columns(args)
    metrics: list[_RunMetric] = []

    print(f"source={args.source}")
    print(f"table_path={_TABLE_PATH}")
    print(f"row_mode={args.row_mode}")
    print(f"predicate={_predicate_label(args)}")
    print(f"select_columns={', '.join(select_columns)}")
    print(f"limit={args.limit or 'all'}")
    print(f"anonymous={file_io.config.anon}")
    print("cache_mode=cold per repeat (lazynwb process caches + catalog sqlite)")
    print()

    for repeat in range(1, args.repeats + 1):
        cache_path = cache_root / f"repeat-{repeat:03d}" / "catalog.sqlite"
        metrics.append(_run_once(args, repeat, cache_path, select_columns))

    return tuple(metrics)


def _run_once(
    args: argparse.Namespace,
    repeat: int,
    cache_path: pathlib.Path,
    select_columns: tuple[str, ...],
) -> _RunMetric:
    _prepare_cold_cache(cache_path)
    predicate = _build_row_predicate(args)

    _LOGGER.debug(
        "starting cold scan repeat=%d source=%s cache_path=%s",
        repeat,
        args.source,
        cache_path,
    )
    try:
        started = time.perf_counter()
        scan_started = time.perf_counter()
        units = lazyframe.scan_nwb(
            args.source,
            table_path=_TABLE_PATH,
            exclude_array_columns=False,
            low_memory=args.low_memory,
            disable_progress=True,
        )
        query: pl.LazyFrame | None = None
        if predicate is not None and args.limit:
            _LOGGER.debug(
                "using explicit limited two-pass collection because Polars does "
                "not pass post-predicate head() into Python IO sources"
            )
        else:
            query = units
            if predicate is not None:
                query = query.filter(predicate)
            query = query.select(select_columns)
            if args.limit:
                query = query.head(args.limit)
        scan_seconds = time.perf_counter() - scan_started

        collect_started = time.perf_counter()
        if predicate is not None and args.limit:
            df = _collect_limited_filtered_units(
                units=units,
                predicate=predicate,
                select_columns=select_columns,
                limit=args.limit,
                args=args,
            )
        else:
            if query is None:
                raise RuntimeError("unlimited benchmark query was not built")
            df = query.collect()
        collect_seconds = time.perf_counter() - collect_started
        total_seconds = time.perf_counter() - started

        metric = _RunMetric(
            repeat=repeat,
            source=args.source,
            table_path=_TABLE_PATH,
            cache_path=cache_path.as_posix(),
            row_mode=args.row_mode,
            filter_column=_metric_filter_column(args),
            operator=_metric_operator(args),
            filter_value=_metric_filter_value(args),
            select_columns=select_columns,
            scan_seconds=scan_seconds,
            collect_seconds=collect_seconds,
            total_seconds=total_seconds,
            rows=df.height,
            columns=df.width,
            estimated_size_bytes=int(df.estimated_size()),
        )
        _LOGGER.debug("finished repeat=%d metric=%s", repeat, metric)
        return metric
    finally:
        file_io.clear_cache()
        attrs.clear_attrs_cache()


def _collect_limited_filtered_units(
    units: pl.LazyFrame,
    predicate: pl.Expr,
    select_columns: tuple[str, ...],
    limit: int,
    args: argparse.Namespace,
) -> pl.DataFrame:
    """Apply row limit before fetching expensive projected columns.

    Polars 1.38 does not pass a post-predicate ``head`` into Python IO sources as
    ``n_rows``. This mirrors lazynwb's two-pass predicate strategy explicitly:
    filter with scalar predicate columns first, then fetch projected columns only
    for the limited row set.
    """
    predicate_columns = tuple(predicate.meta.root_names())
    initial_columns = tuple(
        dict.fromkeys((*predicate_columns, *tables.INTERNAL_COLUMN_NAMES))
    )
    filtered_rows = units.select(initial_columns).collect().filter(predicate).head(limit)
    if filtered_rows.is_empty():
        schema = units.collect_schema()
        return pl.DataFrame(schema={column: schema[column] for column in select_columns})

    additional_columns = tuple(
        column
        for column in select_columns
        if column not in set(initial_columns)
    )
    if not additional_columns:
        return filtered_rows.select(select_columns)

    nwb_path_to_row_indices = tables._get_path_to_row_indices(filtered_rows)
    projected_rows = tables.get_df(
        nwb_data_sources=nwb_path_to_row_indices.keys(),
        search_term=_TABLE_PATH,
        exact_path=True,
        include_column_names=additional_columns,
        nwb_path_to_row_indices=nwb_path_to_row_indices,
        disable_progress=True,
        use_process_pool=False,
        as_polars=True,
        ignore_errors=False,
        low_memory=args.low_memory,
    )
    return (
        filtered_rows.join(
            projected_rows,
            on=[
                tables.NWB_PATH_COLUMN_NAME,
                tables.TABLE_PATH_COLUMN_NAME,
                tables.TABLE_INDEX_COLUMN_NAME,
            ],
            how="inner",
        )
        .select(select_columns)
        .head(limit)
    )


def _prepare_cold_cache(cache_path: pathlib.Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()
    file_io.clear_cache()
    attrs.clear_attrs_cache()
    gc.collect()
    _LOGGER.debug("prepared cold cache at %s", cache_path)


def _select_columns(args: argparse.Namespace) -> tuple[str, ...]:
    columns = tuple(args.select_columns or _DEFAULT_SELECT_COLUMNS)
    if args.include_internal:
        columns = (*columns, *_INTERNAL_SELECT_COLUMNS)
    return tuple(dict.fromkeys(columns))


def _build_predicate(args: argparse.Namespace) -> pl.Expr:
    column = pl.col(args.filter_column)
    if args.operator == "is-null":
        return column.is_null()
    if args.operator == "is-not-null":
        return column.is_not_null()

    literal = pl.lit(_coerce_literal(args.filter_value))
    if args.operator == "eq":
        return column == literal
    if args.operator == "ne":
        return column != literal
    if args.operator == "gt":
        return column > literal
    if args.operator == "ge":
        return column >= literal
    if args.operator == "lt":
        return column < literal
    if args.operator == "le":
        return column <= literal
    raise ValueError(f"unsupported operator: {args.operator!r}")


def _build_row_predicate(args: argparse.Namespace) -> pl.Expr | None:
    if args.row_mode == "all":
        return None
    if args.row_mode == "structure":
        return pl.col("structure") == pl.lit(args.structure)
    if args.row_mode == "filtered":
        return _build_predicate(args)
    raise ValueError(f"unsupported row mode: {args.row_mode!r}")


def _predicate_label(args: argparse.Namespace) -> str:
    if args.row_mode == "all":
        return "<none>"
    if args.row_mode == "structure":
        return f"structure eq {args.structure}"
    suffix = (
        ""
        if args.operator in {"is-null", "is-not-null"}
        else f" {args.filter_value}"
    )
    return f"{args.filter_column} {args.operator}{suffix}"


def _metric_filter_value(args: argparse.Namespace) -> object:
    if args.row_mode == "all":
        return None
    if args.row_mode == "structure":
        return args.structure
    if args.operator in {"is-null", "is-not-null"}:
        return None
    return _coerce_literal(args.filter_value)


def _metric_filter_column(args: argparse.Namespace) -> str:
    if args.row_mode == "all":
        return ""
    if args.row_mode == "structure":
        return "structure"
    return args.filter_column


def _metric_operator(args: argparse.Namespace) -> str:
    if args.row_mode == "all":
        return ""
    if args.row_mode == "structure":
        return "eq"
    return args.operator


def _coerce_literal(value: str) -> object:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _print_metrics(metrics: collections.abc.Sequence[_RunMetric]) -> None:
    print("repeat scan_s collect_s total_s rows cols size_mib cache_path")
    print("-" * 96)
    for metric in metrics:
        print(
            f"{metric.repeat:6d} "
            f"{metric.scan_seconds:6.3f} "
            f"{metric.collect_seconds:9.3f} "
            f"{metric.total_seconds:7.3f} "
            f"{metric.rows:4d} "
            f"{metric.columns:4d} "
            f"{metric.estimated_size_bytes / 1024 / 1024:8.3f} "
            f"{metric.cache_path}"
        )
    print()


def _print_summary(metrics: collections.abc.Sequence[_RunMetric]) -> None:
    totals = [metric.total_seconds for metric in metrics]
    collects = [metric.collect_seconds for metric in metrics]
    scans = [metric.scan_seconds for metric in metrics]
    print(
        "summary "
        f"repeats={len(metrics)} "
        f"total_median={statistics.median(totals):.3f}s "
        f"total_best={min(totals):.3f}s "
        f"scan_median={statistics.median(scans):.3f}s "
        f"collect_median={statistics.median(collects):.3f}s"
    )


def _write_json(
    path: pathlib.Path | None,
    metrics: collections.abc.Sequence[_RunMetric],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metrics": [metric.to_json_dict() for metric in metrics]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"json_output={path}")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_path(name: str) -> pathlib.Path | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return pathlib.Path(value)


if __name__ == "__main__":
    main()
