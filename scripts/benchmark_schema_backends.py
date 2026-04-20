from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import pathlib
import signal
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO

import h5py
import obstore.fsspec
import polars as pl
import polars.datatypes.convert
import s3fs
from h5coro import H5Coro
from h5coro.h5metadata import H5Metadata
from h5coro.webdriver import HTTPDriver

DEFAULT_REGION = "us-west-2"
DEFAULT_LIMIT = 2
DEFAULT_PATHS_FILE = pathlib.Path(__file__).resolve().parents[1] / "paths.txt"
DEFAULT_S3FS_BLOCK_SIZE = 256 * 1024
DEFAULT_AUTO_FILE_WORKERS = 8
DEFAULT_AUTO_COLUMN_WORKERS = 16

DEFAULT_BACKENDS = (
    "fsspec_h5py",
    "h5coro",
    "obstore_h5py",
)

CORRECT_UNITS_SCHEMA = pl.Schema(
    [
        ("activity_drift", pl.Float64),
        ("amplitude", pl.Float64),
        ("amplitude_cutoff", pl.Float64),
        ("amplitude_cv_median", pl.Float64),
        ("amplitude_cv_range", pl.Float64),
        ("amplitude_median", pl.Float64),
        ("ccf_ap", pl.Float64),
        ("ccf_dv", pl.Float64),
        ("ccf_ml", pl.Float64),
        ("channels", pl.List(pl.Int64)),
        ("cluster_id", pl.Int64),
        ("d_prime", pl.Float64),
        ("decoder_label", pl.String),
        ("decoder_probability", pl.Float64),
        ("default_qc", pl.Boolean),
        ("device_name", pl.String),
        ("drift_mad", pl.Float64),
        ("drift_ptp", pl.Float64),
        ("drift_std", pl.Float64),
        ("electrode_group", pl.String),
        ("electrode_group_name", pl.String),
        ("electrodes", pl.List(pl.Int64)),
        ("exp_decay", pl.Float64),
        ("firing_range", pl.Float64),
        ("firing_rate", pl.Float64),
        ("half_width", pl.Float64),
        ("id", pl.Int64),
        ("is_not_drift", pl.Boolean),
        ("isi_violations_count", pl.Float64),
        ("isi_violations_ratio", pl.Float64),
        ("isolation_distance", pl.Float64),
        ("l_ratio", pl.Float64),
        ("location", pl.String),
        ("nn_hit_rate", pl.Float64),
        ("nn_miss_rate", pl.Float64),
        ("num_negative_peaks", pl.Int64),
        ("num_positive_peaks", pl.Int64),
        ("num_spikes", pl.Float64),
        ("obs_intervals", pl.List(pl.Array(pl.Float64, shape=(2,)))),
        ("peak_channel", pl.Int64),
        ("peak_electrode", pl.Int64),
        ("peak_to_valley", pl.Float64),
        ("peak_trough_ratio", pl.Float64),
        ("peak_waveform_index", pl.Int64),
        ("presence_ratio", pl.Float64),
        ("recovery_slope", pl.Float64),
        ("repolarization_slope", pl.Float64),
        ("rp_contamination", pl.Float64),
        ("rp_violations", pl.Float64),
        ("silhouette", pl.Float64),
        ("sliding_rp_violation", pl.Float64),
        ("snr", pl.Float64),
        ("spike_amplitudes", pl.List(pl.Float32)),
        ("spike_times", pl.List(pl.Float64)),
        ("spread", pl.Float64),
        ("structure", pl.String),
        ("sync_spike_2", pl.Float64),
        ("sync_spike_4", pl.Float64),
        ("sync_spike_8", pl.Float64),
        ("unit_id", pl.String),
        ("velocity_above", pl.Float64),
        ("velocity_below", pl.Float64),
        ("waveform_mean", pl.Array(pl.Float64, shape=(210, 384))),
        ("waveform_sd", pl.Array(pl.Float64, shape=(210, 384))),
    ]
)


@dataclass(frozen=True)
class SchemaTask:
    name: str
    table_path: str
    expected_schema: pl.Schema
    excluded_columns: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ColumnDescriptor:
    name: str
    shape: tuple[int, ...]
    base_dtype: pl.DataType


@dataclass
class FileSchemaResult:
    schema: dict[str, pl.DataType]
    unsupported_columns: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


@dataclass
class BenchmarkRun:
    backend: str
    elapsed_s: float
    per_file_s: list[float]
    schema: pl.Schema | None
    valid: bool
    error: str | None = None
    unsupported_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    diff_lines: tuple[str, ...] = ()


TASKS = {
    "units": SchemaTask(
        name="units",
        table_path="/units",
        expected_schema=pl.Schema(
            [
                (name, dtype)
                for name, dtype in CORRECT_UNITS_SCHEMA.items()
                if name != "electrode_group"
            ]
        ),
        excluded_columns=frozenset({"electrode_group"}),
    )
}


def decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def s3_to_https(path: str) -> str:
    if not path.startswith("s3://"):
        return path
    bucket, _, key = path[5:].partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def load_paths(paths_file: pathlib.Path, limit: int | None) -> tuple[str, ...]:
    paths = tuple(
        line.strip()
        for line in paths_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if limit is not None:
        return paths[:limit]
    return paths


def unique_preserve_order(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def schema_to_dict(schema: pl.Schema) -> dict[str, pl.DataType]:
    return {name: schema[name] for name in schema}


def reorder_schema_like_expected(
    schema_map: dict[str, pl.DataType],
    expected_schema: pl.Schema,
) -> pl.Schema:
    ordered = pl.Schema()
    remaining = dict(schema_map)
    for name in expected_schema:
        if name in remaining:
            ordered[name] = remaining.pop(name)
    for name, dtype in remaining.items():
        ordered[name] = dtype
    return ordered


def compare_schemas(actual: pl.Schema, expected: pl.Schema) -> tuple[bool, tuple[str, ...]]:
    actual_map = schema_to_dict(actual)
    expected_map = schema_to_dict(expected)
    if actual_map == expected_map:
        return True, ()

    lines: list[str] = []
    missing = [name for name in expected if name not in actual_map]
    extra = [name for name in actual if name not in expected_map]
    mismatched = [
        name
        for name in expected
        if name in actual_map and actual_map[name] != expected_map[name]
    ]
    for name in missing:
        lines.append(f"missing column: {name}")
    for name in extra:
        lines.append(f"unexpected column: {name}")
    for name in mismatched:
        lines.append(
            f"type mismatch for {name}: expected {expected_map[name]}, got {actual_map[name]}"
        )
    return False, tuple(lines)


def merge_schemas(per_file_schemas: list[dict[str, pl.DataType]]) -> pl.Schema:
    counts: dict[str, collections.Counter] = {}
    for file_schema in per_file_schemas:
        for column_name, dtype in file_schema.items():
            counts.setdefault(column_name, collections.Counter())[dtype] += 1

    merged = pl.Schema()
    for column_name, counter in counts.items():
        merged[column_name] = counter.most_common(1)[0][0]
    return merged


def resolve_worker_count(requested: int, available: int, auto_default: int) -> int:
    if available <= 1:
        return 1
    if requested <= 0:
        return min(auto_default, available)
    return max(1, min(requested, available))


def infer_index_depth(column_name: str, all_names: set[str]) -> int:
    depth = 0
    current = f"{column_name}_index"
    while current in all_names:
        depth += 1
        current = f"{current}_index"
    return depth


def build_dynamic_table_column_order(
    base_names: list[str],
    available_names: set[str],
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        if name in available_names and name not in seen:
            ordered.append(name)
            seen.add(name)

    for name in base_names:
        add_name(name)
    add_name("id")

    current_level = tuple(ordered)
    while current_level:
        next_level: list[str] = []
        for name in current_level:
            index_name = f"{name}_index"
            if index_name in available_names and index_name not in seen:
                ordered.append(index_name)
                seen.add(index_name)
                next_level.append(index_name)
        current_level = tuple(next_level)

    for name in sorted(available_names):
        add_name(name)
    return tuple(ordered)


def numpy_dtype_to_polars(dtype: object) -> pl.DataType:
    kind = getattr(dtype, "kind", None)
    if kind in {"O", "S", "U"}:
        return pl.String
    return polars.datatypes.convert.numpy_char_code_to_dtype(dtype)


def descriptor_to_dtype(
    descriptor: ColumnDescriptor,
    all_names: set[str],
) -> pl.DataType:
    dtype = descriptor.base_dtype
    if len(descriptor.shape) > 1:
        dtype = pl.Array(dtype, shape=descriptor.shape[1:])
    for _ in range(infer_index_depth(descriptor.name, all_names)):
        dtype = pl.List(dtype)
    return dtype


def is_h5py_reference_dataset(dataset: h5py.Dataset) -> bool:
    return h5py.check_dtype(ref=dataset.dtype) is not None


def select_columns_for_description(
    column_order: tuple[str, ...],
    excluded_columns: set[str] | frozenset[str],
    requested_names: set[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in column_order
        if not name.endswith("_index")
        and name not in excluded_columns
        and (requested_names is None or name in requested_names)
    )


def build_dynamic_table_schema(
    column_order: tuple[str, ...],
    columns: dict[str, ColumnDescriptor],
) -> dict[str, pl.DataType]:
    all_names = set(column_order)
    schema: dict[str, pl.DataType] = {}
    for name in column_order:
        if name.endswith("_index"):
            continue
        descriptor = columns.get(name)
        if descriptor is None:
            continue
        schema[name] = descriptor_to_dtype(descriptor, all_names)
    return schema


def h5py_describe_columns(
    group: h5py.Group,
    excluded_columns: set[str] | frozenset[str],
    column_workers: int,
    requested_names: set[str] | None = None,
) -> tuple[tuple[str, ...], dict[str, ColumnDescriptor]]:
    colnames = group.attrs.get("colnames")
    if colnames is None:
        base_names = sorted(name for name in group.keys() if not name.endswith("_index"))
    else:
        base_names = [decode_name(name) for name in colnames]
    available_names = set(group.keys())
    column_order = build_dynamic_table_column_order(base_names, available_names)
    names_to_describe = select_columns_for_description(
        column_order,
        excluded_columns=excluded_columns,
        requested_names=requested_names,
    )

    columns: dict[str, ColumnDescriptor] = {}
    max_workers = resolve_worker_count(
        requested=column_workers,
        available=len(names_to_describe),
        auto_default=DEFAULT_AUTO_COLUMN_WORKERS,
    )
    if max_workers > 1 and len(names_to_describe) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="schema_cols",
        ) as executor:
            future_to_name = {
                executor.submit(_describe_h5py_column, group, name): name
                for name in names_to_describe
            }
            for future in concurrent.futures.as_completed(future_to_name):
                descriptor = future.result()
                if descriptor is not None:
                    columns[descriptor.name] = descriptor
    else:
        for name in names_to_describe:
            descriptor = _describe_h5py_column(group, name)
            if descriptor is not None:
                columns[descriptor.name] = descriptor
    return column_order, columns


def _describe_h5py_column(
    group: h5py.Group,
    name: str,
) -> ColumnDescriptor | None:
    obj = group.get(name)
    if not isinstance(obj, h5py.Dataset):
        return None
    if is_h5py_reference_dataset(obj):
        return None
    return ColumnDescriptor(
        name=name,
        shape=tuple(int(dim) for dim in obj.shape),
        base_dtype=numpy_dtype_to_polars(obj.dtype),
    )


def h5coro_meta_to_polars(meta: H5Metadata) -> pl.DataType:
    if meta.type in {
        H5Metadata.STRING_TYPE,
        H5Metadata.VL_STRING_TYPE,
        H5Metadata.REFERENCE_TYPE,
        H5Metadata.ENUMERATED_TYPE,
        H5Metadata.COMPOUND_TYPE,
        H5Metadata.UNKNOWN_TYPE,
    }:
        return pl.String
    if meta.type in {
        H5Metadata.FIXED_POINT_TYPE,
        H5Metadata.FLOATING_POINT_TYPE,
    }:
        return numpy_dtype_to_polars(meta.getNumpyType())
    raise TypeError(f"unsupported h5coro metadata type: {meta.type}")


class H5pySchemaBackend:
    def __init__(
        self,
        name: str,
        region: str,
        column_workers: int,
        cache_type: str | None = None,
        block_size: int | None = None,
    ) -> None:
        self.name = name
        self.column_workers = column_workers
        client_kwargs = {"region_name": region} if region else {}
        self._fs = s3fs.S3FileSystem(anon=True, client_kwargs=client_kwargs)
        self._cache_type = cache_type
        self._block_size = block_size

    def _open_binary(self, path: str) -> BinaryIO:
        kwargs = {}
        if self._cache_type is not None:
            kwargs["cache_type"] = self._cache_type
        if self._block_size is not None:
            kwargs["block_size"] = self._block_size
        return self._fs.open(path, mode="rb", **kwargs)

    @contextlib.contextmanager
    def _open_h5(self, path: str):
        binary = self._open_binary(path)
        try:
            with h5py.File(binary, mode="r") as h5_file:
                yield h5_file
        finally:
            with contextlib.suppress(Exception):
                binary.close()

    def describe_columns(
        self,
        path: str,
        table_path: str,
        excluded_columns: set[str] | frozenset[str],
        requested_names: set[str] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, ColumnDescriptor]]:
        with self._open_h5(path) as h5_file:
            group = h5_file[table_path]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"{table_path!r} is not a group in {path!r}")
            return h5py_describe_columns(
                group,
                excluded_columns=excluded_columns,
                column_workers=self.column_workers,
                requested_names=requested_names,
            )

    def infer_file_schema(self, path: str, task: SchemaTask) -> FileSchemaResult:
        column_order, columns = self.describe_columns(
            path,
            task.table_path,
            excluded_columns=task.excluded_columns,
        )
        return FileSchemaResult(schema=build_dynamic_table_schema(column_order, columns))


class ObstoreSchemaBackend:
    def __init__(self, name: str, region: str, column_workers: int) -> None:
        self.name = name
        self.column_workers = column_workers
        self._store = obstore.fsspec.FsspecStore(
            "s3",
            region=region,
            skip_signature=True,
        )

    @contextlib.contextmanager
    def _open_h5(self, path: str):
        binary = obstore.fsspec.BufferedFile(fs=self._store, path=path)
        try:
            with h5py.File(binary, mode="r") as h5_file:
                yield h5_file
        finally:
            with contextlib.suppress(Exception):
                binary.close()

    def describe_columns(
        self,
        path: str,
        table_path: str,
        excluded_columns: set[str] | frozenset[str],
        requested_names: set[str] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, ColumnDescriptor]]:
        with self._open_h5(path) as h5_file:
            group = h5_file[table_path]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"{table_path!r} is not a group in {path!r}")
            return h5py_describe_columns(
                group,
                excluded_columns=excluded_columns,
                column_workers=self.column_workers,
                requested_names=requested_names,
            )

    def infer_file_schema(self, path: str, task: SchemaTask) -> FileSchemaResult:
        column_order, columns = self.describe_columns(
            path,
            task.table_path,
            excluded_columns=task.excluded_columns,
        )
        return FileSchemaResult(schema=build_dynamic_table_schema(column_order, columns))


class H5CoroSchemaBackend:
    def __init__(self, name: str, column_workers: int) -> None:
        self.name = name
        self.column_workers = column_workers

    def infer_file_schema(self, path: str, task: SchemaTask) -> FileSchemaResult:
        h5 = H5Coro(
            s3_to_https(path),
            HTTPDriver,
            credentials={},
            errorChecking=False,
            verbose=False,
        )
        try:
            table_path = task.table_path.strip("/")
            links, _attributes, metadata = h5.inspectPath(task.table_path, w_attr=False)
            if metadata is not None:
                raise TypeError(f"{task.table_path!r} is not a group in {path!r}")

            base_names = sorted(name for name in links if not name.endswith("_index"))
            available_names = set(links)
            column_order = build_dynamic_table_column_order(base_names, available_names)
            names_to_describe = select_columns_for_description(
                column_order,
                excluded_columns=task.excluded_columns,
            )

            columns: dict[str, ColumnDescriptor] = {}
            unsupported_columns: set[str] = set()
            max_workers = resolve_worker_count(
                requested=self.column_workers,
                available=len(names_to_describe),
                auto_default=DEFAULT_AUTO_COLUMN_WORKERS,
            )
            if max_workers > 1 and len(names_to_describe) > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="schema_cols",
                ) as executor:
                    future_to_name = {
                        executor.submit(
                            _inspect_h5coro_column,
                            h5,
                            table_path,
                            name,
                        ): name
                        for name in names_to_describe
                    }
                    for future in concurrent.futures.as_completed(future_to_name):
                        name, descriptor, is_unsupported = future.result()
                        if descriptor is not None:
                            columns[name] = descriptor
                        elif is_unsupported:
                            unsupported_columns.add(name)
            else:
                for name in names_to_describe:
                    name, descriptor, is_unsupported = _inspect_h5coro_column(
                        h5,
                        table_path,
                        name,
                    )
                    if descriptor is not None:
                        columns[name] = descriptor
                    elif is_unsupported:
                        unsupported_columns.add(name)

            return FileSchemaResult(
                schema=build_dynamic_table_schema(column_order, columns),
                unsupported_columns=unsupported_columns,
            )
        finally:
            with contextlib.suppress(Exception):
                h5.close()


def _inspect_h5coro_column(
    h5: H5Coro,
    table_path: str,
    name: str,
) -> tuple[str, ColumnDescriptor | None, bool]:
    try:
        _child_links, _child_attributes, meta = h5.inspectPath(
            f"/{table_path}/{name}",
            w_attr=False,
        )
    except Exception:
        return name, None, True
    if meta is None or meta.type == H5Metadata.REFERENCE_TYPE:
        return name, None, False
    try:
        return (
            name,
            ColumnDescriptor(
                name=name,
                shape=tuple(int(dim) for dim in meta.dimensions),
                base_dtype=h5coro_meta_to_polars(meta),
            ),
            False,
        )
    except Exception:
        return name, None, True


def make_backends(region: str, column_workers: int) -> dict[str, object]:
    return {
        "fsspec_h5py": H5pySchemaBackend(
            name="fsspec_h5py",
            region=region,
            column_workers=column_workers,
            cache_type="none",
            block_size=DEFAULT_S3FS_BLOCK_SIZE,
        ),
        "obstore_h5py": ObstoreSchemaBackend(
            name="obstore_h5py",
            region=region,
            column_workers=column_workers,
        ),
        "h5coro": H5CoroSchemaBackend(name="h5coro", column_workers=column_workers),
    }


def run_backend(
    backend: object,
    paths: tuple[str, ...],
    task: SchemaTask,
    file_workers: int,
    per_file_timeout_s: float | None = None,
) -> BenchmarkRun:
    try:
        per_file_s: list[float | None] = [None] * len(paths)
        per_file_schemas: list[dict[str, pl.DataType] | None] = [None] * len(paths)
        unsupported_columns: set[str] = set()
        notes: list[str] = []

        started = time.perf_counter()
        print(
            f"\n[{backend.name}] starting "
            f"(file_workers={file_workers}, "
            f"column_workers={'auto' if backend.column_workers <= 0 else backend.column_workers})",
            flush=True,
        )
        if file_workers > 1 and per_file_timeout_s and per_file_timeout_s > 0:
            notes.append(
                "per-file timeout is only enforced in serial file mode"
            )
        if file_workers > 1 and len(paths) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=file_workers,
                thread_name_prefix=f"{backend.name}_files",
            ) as executor:
                future_to_index = {
                    executor.submit(
                        _run_file_schema_task,
                        backend=backend,
                        path=path,
                        task=task,
                        file_index=file_index,
                        total_files=len(paths),
                        per_file_timeout_s=per_file_timeout_s,
                    ): file_index
                    for file_index, path in enumerate(paths)
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    file_index, elapsed_s, result = future.result()
                    per_file_s[file_index] = elapsed_s
                    per_file_schemas[file_index] = result.schema
                    unsupported_columns.update(result.unsupported_columns)
                    notes.extend(result.notes)
        else:
            for file_index, path in enumerate(paths):
                file_index, elapsed_s, result = _run_file_schema_task(
                    backend=backend,
                    path=path,
                    task=task,
                    file_index=file_index,
                    total_files=len(paths),
                    per_file_timeout_s=per_file_timeout_s,
                )
                per_file_s[file_index] = elapsed_s
                per_file_schemas[file_index] = result.schema
                unsupported_columns.update(result.unsupported_columns)
                notes.extend(result.notes)
        elapsed_s = time.perf_counter() - started

        merged_schema = reorder_schema_like_expected(
            schema_to_dict(
                merge_schemas(
                    [schema for schema in per_file_schemas if schema is not None]
                )
            ),
            task.expected_schema,
        )
        valid, diff_lines = compare_schemas(merged_schema, task.expected_schema)
        if unsupported_columns:
            valid = False
            diff_lines = diff_lines + tuple(
                f"unsupported column metadata: {name}" for name in sorted(unsupported_columns)
            )

        return BenchmarkRun(
            backend=backend.name,
            elapsed_s=elapsed_s,
            per_file_s=[value for value in per_file_s if value is not None],
            schema=merged_schema,
            valid=valid,
            unsupported_columns=tuple(sorted(unsupported_columns)),
            notes=unique_preserve_order(notes),
            diff_lines=diff_lines,
        )
    except Exception as exc:
        return BenchmarkRun(
            backend=backend.name,
            elapsed_s=0.0,
            per_file_s=[],
            schema=None,
            valid=False,
            error=repr(exc),
        )


def _run_file_schema_task(
    backend: object,
    path: str,
    task: SchemaTask,
    file_index: int,
    total_files: int,
    per_file_timeout_s: float | None,
) -> tuple[int, float, FileSchemaResult]:
    file_label = f"[{backend.name}] file {file_index + 1}/{total_files}"
    print(f"{file_label}: {path}", flush=True)
    file_started = time.perf_counter()
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_print_heartbeat,
        args=(file_label, file_started, stop_event),
        daemon=True,
    )
    heartbeat.start()
    try:
        with _file_timeout(per_file_timeout_s):
            result = backend.infer_file_schema(path, task)
    except Exception as exc:
        elapsed_s = time.perf_counter() - file_started
        print(f"{file_label}: failed in {elapsed_s:.2f}s: {exc!r}", flush=True)
        raise
    finally:
        stop_event.set()
        heartbeat.join(timeout=0.1)
    elapsed_s = time.perf_counter() - file_started
    print(f"{file_label}: completed in {elapsed_s:.2f}s", flush=True)
    return file_index, elapsed_s, result


def _print_heartbeat(
    label: str,
    started: float,
    stop_event: threading.Event,
    interval_s: float = 10.0,
) -> None:
    while not stop_event.wait(interval_s):
        print(
            f"{label}: still running after {time.perf_counter() - started:.1f}s",
            flush=True,
        )


@contextlib.contextmanager
def _file_timeout(timeout_s: float | None):
    if (
        timeout_s is None
        or timeout_s <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"timed out after {timeout_s:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark schema inference over the same NWB paths used by the existing "
            "get_table_schema benchmark, but using backend-native metadata walks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths-file",
        type=pathlib.Path,
        default=DEFAULT_PATHS_FILE,
        help="Text file containing one NWB S3 path per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of paths to benchmark from the top of the file.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of full benchmark repeats per backend.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Region to pin for public S3 access.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="units",
        help="Schema benchmark task to run.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=sorted((*DEFAULT_BACKENDS, "all")),
        help="Backend(s) to run. May be repeated.",
    )
    parser.add_argument(
        "--per-file-timeout",
        type=float,
        default=120.0,
        help="Fail a backend if a single file takes longer than this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=0,
        help="Worker threads across files. Use 0 to pick an auto-parallel default.",
    )
    parser.add_argument(
        "--column-workers",
        type=int,
        default=0,
        help="Worker threads across columns within each file. Use 0 to pick an auto-parallel default.",
    )
    return parser


def resolve_backend_names(selected: list[str] | None) -> tuple[str, ...]:
    if not selected or "all" in selected:
        return DEFAULT_BACKENDS
    return tuple(selected)


def print_run_summary(run: BenchmarkRun, repeat: int, total_repeats: int) -> None:
    label = f"[{run.backend}] repeat {repeat}/{total_repeats}"
    if run.error is not None:
        print(f"{label}: ERROR in {run.elapsed_s:.2f}s: {run.error}", flush=True)
        return

    status = "OK" if run.valid else "INVALID"
    print(
        f"{label}: {status} in {run.elapsed_s:.2f}s "
        f"(per-file: {', '.join(f'{value:.2f}s' for value in run.per_file_s)})",
        flush=True,
    )
    for note in run.notes:
        print(f"  note: {note}", flush=True)
    for line in run.diff_lines:
        print(f"  {line}", flush=True)


def print_final_summary(all_runs: dict[str, list[BenchmarkRun]]) -> None:
    print("\nSummary", flush=True)
    header = f"{'backend':<16} {'status':<8} {'mean_s':>8} {'best_s':>8} {'runs':>4}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for backend_name, runs in all_runs.items():
        elapsed = [run.elapsed_s for run in runs if run.error is None]
        status = "OK" if runs and all(run.valid for run in runs) else "FAIL"
        mean_s = statistics.mean(elapsed) if elapsed else float("nan")
        best_s = min(elapsed) if elapsed else float("nan")
        print(
            f"{backend_name:<16} {status:<8} {mean_s:>8.2f} {best_s:>8.2f} {len(runs):>4}",
            flush=True,
        )


def main() -> int:
    args = build_argument_parser().parse_args()
    task = TASKS[args.task]
    backend_names = resolve_backend_names(args.backend)
    paths = load_paths(args.paths_file, args.limit)
    file_workers = resolve_worker_count(
        requested=args.file_workers,
        available=len(paths),
        auto_default=DEFAULT_AUTO_FILE_WORKERS,
    )
    backends = make_backends(region=args.region, column_workers=args.column_workers)

    print(
        f"Benchmarking task={task.name!r} over {len(paths)} file(s) from {args.paths_file}",
        flush=True,
    )
    print(
        "Worker settings: "
        f"file_workers={file_workers}, "
        f"column_workers={'auto' if args.column_workers <= 0 else args.column_workers}",
        flush=True,
    )
    print(f"Selected backends: {', '.join(backend_names)}", flush=True)
    for path in paths:
        print(f"  - {path}", flush=True)

    all_runs: dict[str, list[BenchmarkRun]] = {name: [] for name in backend_names}
    for backend_name in backend_names:
        backend = backends[backend_name]
        for repeat in range(1, args.repeat + 1):
            run = run_backend(
                backend,
                paths,
                task,
                file_workers=file_workers,
                per_file_timeout_s=args.per_file_timeout,
            )
            all_runs[backend_name].append(run)
            print_run_summary(run, repeat=repeat, total_repeats=args.repeat)

    print_final_summary(all_runs)
    return 0 if all(run.valid for runs in all_runs.values() for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
