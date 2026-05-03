from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import time
from collections.abc import Callable

import pytest

import lazynwb
import lazynwb._hdf5.reader as hdf5_reader

_REMOTE_SCHEMA_SOURCE = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
    "620263_2022-07-26.nwb"
)
_REMOTE_TABLES = ("/intervals/trials", "/units")

pytestmark = [
    pytest.mark.remote_schema_budget,
    pytest.mark.skipif(
        os.environ.get("LAZYNWB_REMOTE_SCHEMA_TESTS") != "1",
        reason="set LAZYNWB_REMOTE_SCHEMA_TESTS=1 to run remote schema budget tests",
    ),
]


@dataclasses.dataclass(frozen=True, slots=True)
class _SchemaBudgetMetric:
    source_url: str
    table_path: str
    phase: str
    request_count: int
    fetched_bytes: int
    elapsed_seconds: float
    cache_status: str
    column_count: int
    validator_kind: str

    def as_properties(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def failure_detail(self) -> str:
        return (
            f"source={self.source_url!r} table={self.table_path!r} "
            f"phase={self.phase!r} cache={self.cache_status!r} "
            f"validator={self.validator_kind!r} requests={self.request_count} "
            f"bytes={self.fetched_bytes} elapsed={self.elapsed_seconds:.3f}s "
            f"columns={self.column_count}"
        )


@pytest.mark.parametrize("table_path", _REMOTE_TABLES)
def test_remote_hdf5_schema_request_budget_and_warm_cache(
    table_path: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    lazynwb.config.anon = True

    cold = _read_remote_schema_metric(_REMOTE_SCHEMA_SOURCE, table_path, phase="cold")
    warm = _read_remote_schema_metric(_REMOTE_SCHEMA_SOURCE, table_path, phase="warm")

    for metric in (cold, warm):
        _record_metric(record_property, metric)

    max_cold_requests = _env_int("LAZYNWB_REMOTE_SCHEMA_MAX_GETS", 3500)
    max_cold_bytes = _env_int("LAZYNWB_REMOTE_SCHEMA_MAX_BYTES", 128 * 1024 * 1024)

    assert cold.column_count > 0, cold.failure_detail()
    assert cold.request_count > 0, cold.failure_detail()
    assert cold.request_count <= max_cold_requests, cold.failure_detail()
    assert cold.fetched_bytes <= max_cold_bytes, cold.failure_detail()
    assert warm.column_count == cold.column_count, (
        f"warm-cache schema changed: cold={cold.failure_detail()} "
        f"warm={warm.failure_detail()}"
    )
    if cold.validator_kind in {"version_id", "etag", "last_modified_content_length"}:
        assert warm.request_count == 0, warm.failure_detail()


def _read_remote_schema_metric(
    source_url: str,
    table_path: str,
    phase: str,
) -> _SchemaBudgetMetric:
    reader = hdf5_reader._default_hdf5_backend_reader(source_url)
    t0 = time.perf_counter()
    snapshot = asyncio.run(
        reader.read_table_schema_snapshot(lazynwb.normalize_internal_file_path(table_path))
    )
    elapsed_seconds = time.perf_counter() - t0
    request_count = int(getattr(reader._range_reader, "request_count", 0))
    fetched_bytes = int(getattr(reader._range_reader, "bytes_fetched", 0))
    cache_status = "hit" if request_count == 0 else "miss"
    return _SchemaBudgetMetric(
        source_url=source_url,
        table_path=table_path,
        phase=phase,
        request_count=request_count,
        fetched_bytes=fetched_bytes,
        elapsed_seconds=elapsed_seconds,
        cache_status=cache_status,
        column_count=len(snapshot.columns),
        validator_kind=snapshot.source_identity.validator_kind,
    )


def _record_metric(
    record_property: Callable[[str, object], None],
    metric: _SchemaBudgetMetric,
) -> None:
    prefix = f"{metric.table_path.strip('/').replace('/', '_')}_{metric.phase}"
    for name, value in metric.as_properties().items():
        record_property(f"{prefix}_{name}", value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)
