from __future__ import annotations

import dataclasses
import platform
import sys
from collections.abc import Callable, Iterator

import polars as pl
import polars.io.plugins


@dataclasses.dataclass
class _ScanCall:
    with_columns: list[str] | None
    predicate_received: bool
    n_rows: int | None
    batch_size: int | None
    batches_yielded: list[int] = dataclasses.field(default_factory=list)


def _make_lazy_frame() -> tuple[pl.LazyFrame, list[_ScanCall]]:
    calls: list[_ScanCall] = []

    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        call = _ScanCall(
            with_columns=with_columns,
            predicate_received=predicate is not None,
            n_rows=n_rows,
            batch_size=batch_size,
        )
        calls.append(call)

        rows_emitted = 0
        for batch_index in range(5):
            start = batch_index * 10
            frame = pl.DataFrame(
                {
                    "x": range(start, start + 10),
                    "payload": [f"row-{row}" for row in range(start, start + 10)],
                }
            )
            if predicate is not None:
                frame = frame.filter(predicate)
            if n_rows is not None:
                remaining = n_rows - rows_emitted
                if remaining <= 0:
                    break
                frame = frame.head(remaining)
            if with_columns is not None:
                frame = frame.select(with_columns)
            if frame.is_empty():
                continue

            call.batches_yielded.append(batch_index)
            rows_emitted += frame.height
            yield frame

            if n_rows is not None and rows_emitted >= n_rows:
                break

    return (
        polars.io.plugins.register_io_source(
            io_source=_io_source,
            schema={"x": pl.Int64, "payload": pl.String},
        ),
        calls,
    )


def _run_case(
    name: str,
    query_factory: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> None:
    lazy_frame, calls = _make_lazy_frame()
    query = query_factory(lazy_frame)

    print(f"\n=== {name} ===")
    print(query.explain())
    result = query.collect()
    print(result)

    if len(calls) != 1:
        raise AssertionError(f"expected one IO source call, got {len(calls)}")
    call = calls[0]
    print(
        "io_source_call="
        f"with_columns={call.with_columns!r} "
        f"predicate_received={call.predicate_received} "
        f"n_rows={call.n_rows!r} "
        f"batch_size={call.batch_size!r} "
        f"batches_yielded={call.batches_yielded!r}"
    )


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"polars={pl.__version__}")

    _run_case(
        "head_only",
        lambda lazy_frame: lazy_frame.head(3),
    )
    _run_case(
        "filter_head",
        lambda lazy_frame: lazy_frame.filter(pl.col("x") >= 0).head(3),
    )
    _run_case(
        "filter_select_head",
        lambda lazy_frame: lazy_frame.filter(pl.col("x") >= 0).select("x").head(3),
    )


if __name__ == "__main__":
    main()
