"""Run NeuroDataBench against two lazynwb checkouts and gate regressions.

The script deliberately runs each checkout in a fresh ``uv`` script environment.
That keeps the package under test tied to the requested checkout while the
benchmark implementation and its runner come from the pinned NeuroDataBench
checkout supplied by the caller.
"""

from __future__ import annotations

import argparse
import collections.abc
import dataclasses
import json
import logging
import math
import os
import pathlib
import signal
import statistics
import subprocess
import sys

_LOGGER = logging.getLogger(__name__)

_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"
_DEFAULT_BACKEND = "obstore"
_DEFAULT_REPEATS = 3
_DEFAULT_MAX_REGRESSION_PERCENT = 15.0
_DEFAULT_MIN_REGRESSION_SECONDS = 0.25
_DEFAULT_TIMEOUT_SECONDS = 300.0
_IMPLEMENTATION = "implementations/lazynwb_template.py"


@dataclasses.dataclass(frozen=True, slots=True)
class _RunResult:
    """Metrics and process status for one benchmark invocation."""

    label: str
    repeat: int
    output_dir: pathlib.Path
    returncode: int | None
    timed_out: bool
    total_seconds: float | None
    setup_seconds: float | None
    submit_answers_seconds: float | None
    peak_rss_delta_bytes: float | None
    network_received_bytes: float | None
    correct: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this result."""
        result = dataclasses.asdict(self)
        result["output_dir"] = self.output_dir.as_posix()
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class _MetricComparison:
    """Median baseline/candidate values and the resulting regression decision."""

    metric: str
    baseline_median: float
    candidate_median: float
    regression_percent: float
    max_regression_percent: float
    min_regression_seconds: float
    regressed: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this comparison."""
        return dataclasses.asdict(self)


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run repeated baseline/candidate benchmarks and return a process status."""
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    _validate_args(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    neurodatabench_dir = args.neurodatabench_dir.resolve()

    revisions = {
        "baseline": _git_revision(baseline_dir),
        "candidate": _git_revision(candidate_dir),
        "neurodatabench": _git_revision(neurodatabench_dir),
    }
    _LOGGER.info(
        "Running NeuroDataBench regression benchmark=%s backend=%s repeats=%d "
        "baseline=%s candidate=%s.",
        args.benchmark,
        args.backend,
        args.repeats,
        revisions["baseline"],
        revisions["candidate"],
    )

    results: list[_RunResult] = []
    for repeat in range(1, args.repeats + 1):
        # Alternate the order to reduce systematic bias from remote object-store
        # connection and server-side caches without sharing a local lazynwb cache.
        labels = ("baseline", "candidate") if repeat % 2 else ("candidate", "baseline")
        for label in labels:
            checkout = baseline_dir if label == "baseline" else candidate_dir
            result = _run_once(
                label=label,
                repeat=repeat,
                checkout=checkout,
                neurodatabench_dir=neurodatabench_dir,
                benchmark=args.benchmark,
                backend=args.backend,
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
                python_executable=args.python_executable,
            )
            results.append(result)

    comparisons = _compare_results(
        results,
        max_regression_percent=args.max_regression_percent,
        min_regression_seconds=args.min_regression_seconds,
    )
    passed = _all_results_valid(results) and not any(
        comparison.regressed for comparison in comparisons
    )
    summary = _summary_payload(
        args=args,
        revisions=revisions,
        results=results,
        comparisons=comparisons,
        passed=passed,
    )
    _write_summary(output_dir, summary)
    _print_summary(summary)
    return 0 if passed else 1


def _parse_args(argv: collections.abc.Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for a regression run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurodatabench-dir", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-dir", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", default=_DEFAULT_BENCHMARK)
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    parser.add_argument(
        "--python-executable",
        type=pathlib.Path,
        help="Use one existing Python environment for both checkouts.",
    )
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=_DEFAULT_MAX_REGRESSION_PERCENT,
        help="Fail when a candidate median exceeds a baseline median by this percentage.",
    )
    parser.add_argument(
        "--min-regression-seconds",
        type=float,
        default=_DEFAULT_MIN_REGRESSION_SECONDS,
        help="Require this absolute slowdown as well as the percentage threshold.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Maximum wall time for each uv benchmark invocation.",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("results")
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _configure_logging(log_level: str) -> None:
    """Configure concise console logging for the orchestration process."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Validate paths and numeric regression settings before starting work."""
    for name in ("neurodatabench_dir", "baseline_dir", "candidate_dir"):
        path = getattr(args, name)
        if not path.is_dir():
            raise ValueError(f"{name} is not a directory: {path}")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.max_regression_percent < 0:
        raise ValueError("--max-regression-percent must be non-negative")
    if args.min_regression_seconds < 0:
        raise ValueError("--min-regression-seconds must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    implementation = _implementation_path(args.neurodatabench_dir)
    if not implementation.is_file():
        raise ValueError(f"NeuroDataBench implementation is missing: {implementation}")
    benchmark = (
        args.neurodatabench_dir
        / "src"
        / "neurodatabench"
        / "benchmarks"
        / f"{args.benchmark}.json"
    )
    if not benchmark.is_file():
        raise ValueError(f"NeuroDataBench benchmark is missing: {benchmark}")


def _implementation_path(checkout: pathlib.Path) -> pathlib.Path:
    """Support the current per-benchmark script and the original pinned template."""
    current = checkout / "implementations/lazynwb/dynamic_routing.py"
    return current if current.is_file() else checkout / _IMPLEMENTATION


def _git_revision(checkout: pathlib.Path) -> str:
    """Return the checked-out commit, or a useful marker for non-git sources."""
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def _run_once(
    *,
    label: str,
    repeat: int,
    checkout: pathlib.Path,
    neurodatabench_dir: pathlib.Path,
    benchmark: str,
    backend: str,
    output_dir: pathlib.Path,
    timeout_seconds: float,
    python_executable: pathlib.Path | None = None,
) -> _RunResult:
    """Run one checkout and collect its NeuroDataBench result artifacts."""
    run_dir = output_dir / label / f"repeat-{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = (
        neurodatabench_dir
        / "src"
        / "neurodatabench"
        / "benchmarks"
        / f"{benchmark}.json"
    )
    implementation_path = _implementation_path(neurodatabench_dir)
    cache_path = run_dir / "catalog.sqlite"
    command = [
        "uv",
        "run",
        "--with-editable",
        str(checkout),
        "--with-editable",
        str(neurodatabench_dir),
        "--script",
        str(implementation_path),
        "--benchmark",
        str(benchmark_path),
        "--out",
        str(run_dir),
        "--log-level",
        "DEBUG",
    ]
    if python_executable is not None:
        # Keep dependencies identical while selecting lazynwb from the checkout.
        # The printed import path is evidence that the installed distribution did
        # not accidentally replace the requested source tree.
        bootstrap = (
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
            "import lazynwb; print('CHECKOUT: '+lazynwb.__file__,flush=True); "
            "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        )
        command = [
            str(python_executable.resolve()),
            "-c",
            bootstrap,
            str(checkout / "src"),
            str(implementation_path),
            "--benchmark",
            str(benchmark_path),
            "--out",
            str(run_dir),
            "--log-level",
            "DEBUG",
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "AWS_REGION": environment.get("AWS_REGION", "us-west-2"),
            "AWS_DEFAULT_REGION": environment.get("AWS_DEFAULT_REGION", "us-west-2"),
            "AWS_EC2_METADATA_DISABLED": "true",
            "NDB_BENCHMARK": str(benchmark_path),
            "NDB_IMPLEMENTATION_ID": f"lazynwb-{label}-{repeat:02d}",
            "NDB_LOCAL_CACHE": "cold",
            "NDB_LAZYNWB_CACHE_PATH": str(cache_path),
            "NDB_OBJECT_STORE_BACKEND": backend,
        }
    )
    _LOGGER.info("Starting %s repeat %d: %s", label, repeat, " ".join(command))
    completed, timed_out = _run_command(
        command,
        cwd=neurodatabench_dir,
        environment=environment,
        timeout_seconds=timeout_seconds,
        log_path=run_dir / "orchestrator.log",
    )
    if completed is not None:
        _LOGGER.info(
            "Finished %s repeat %d with exit code %d.",
            label,
            repeat,
            completed.returncode,
        )
    else:
        _LOGGER.error(
            "Timed out %s repeat %d after %.1f seconds.", label, repeat, timeout_seconds
        )

    result = _read_run_result(
        label=label,
        repeat=repeat,
        output_dir=run_dir,
        returncode=None if completed is None else completed.returncode,
        timed_out=timed_out,
    )
    _LOGGER.info(
        "%s repeat %d metrics: total=%s setup=%s submit=%s rss_delta=%s correct=%s",
        label,
        repeat,
        result.total_seconds,
        result.setup_seconds,
        result.submit_answers_seconds,
        result.peak_rss_delta_bytes,
        result.correct,
    )
    return result


def _run_command(
    command: list[str],
    *,
    cwd: pathlib.Path,
    environment: dict[str, str],
    timeout_seconds: float,
    log_path: pathlib.Path,
) -> tuple[subprocess.CompletedProcess[str] | None, bool]:
    """Run a benchmark command, preserving output and terminating timed-out groups."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _LOGGER.error("Benchmark command timed out: %s", " ".join(command))
        _terminate_process(process)
        try:
            stdout, _ = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            _LOGGER.error("Timed-out benchmark process did not exit after termination.")
            _kill_process(process)
            stdout, _ = process.communicate()
        partial_output = error.output or ""
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode("utf-8", errors="replace")
        stdout = str(partial_output) + stdout
        _write_text(log_path, stdout)
        _emit_command_output(stdout)
        return None, True

    _write_text(log_path, stdout)
    _emit_command_output(stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout, None), False


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out command and its children where the platform allows."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )


def _kill_process(process: subprocess.Popen[str]) -> None:
    """Force-kill a timed-out command and its process group."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()


def _emit_command_output(output: str) -> None:
    """Forward child output to CI logs without adding another log format."""
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()


def _read_run_result(
    *,
    label: str,
    repeat: int,
    output_dir: pathlib.Path,
    returncode: int | None,
    timed_out: bool,
) -> _RunResult:
    """Read NeuroDataBench's timing, validation, and profile artifacts."""
    timings = _read_json(output_dir / "timings.json")
    validation = _read_json(output_dir / "validation.json")
    profile = _read_json(output_dir / "profile_summary.json")
    return _RunResult(
        label=label,
        repeat=repeat,
        output_dir=output_dir,
        returncode=returncode,
        timed_out=timed_out,
        total_seconds=_seconds_from_nanoseconds(timings, "total_duration_ns"),
        setup_seconds=_seconds_from_nanoseconds(timings, "setup_duration_ns"),
        submit_answers_seconds=_seconds_from_nanoseconds(
            timings,
            "submit_answers_duration_ns",
        ),
        peak_rss_delta_bytes=_number(
            profile, "peak_process_plus_children_rss_delta_bytes"
        ),
        network_received_bytes=_nested_number(profile, "network_delta", "bytes_recv"),
        correct=returncode == 0 and not timed_out and validation.get("correct") is True,
    )


def _read_json(path: pathlib.Path) -> dict[str, object]:
    """Read one JSON object, returning an empty object when an invocation failed."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _seconds_from_nanoseconds(data: dict[str, object], key: str) -> float | None:
    """Convert a NeuroDataBench nanosecond timing field to seconds."""
    value = _number(data, key)
    return None if value is None else value / 1_000_000_000


def _number(data: dict[str, object], key: str) -> float | None:
    """Return a finite numeric JSON field, if present."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _nested_number(data: dict[str, object], parent: str, key: str) -> float | None:
    """Return a finite numeric field from a nested JSON object."""
    value = data.get(parent)
    return _number(value, key) if isinstance(value, dict) else None


def _compare_results(
    results: collections.abc.Iterable[_RunResult],
    *,
    max_regression_percent: float,
    min_regression_seconds: float = _DEFAULT_MIN_REGRESSION_SECONDS,
) -> tuple[_MetricComparison, ...]:
    """Compare median timing metrics for valid baseline and candidate runs."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for metric in ("total_seconds", "setup_seconds", "submit_answers_seconds"):
            value = getattr(result, metric)
            if value is not None and result.correct:
                grouped.setdefault(metric, {}).setdefault(result.label, []).append(
                    value
                )

    comparisons: list[_MetricComparison] = []
    for metric in ("total_seconds", "setup_seconds", "submit_answers_seconds"):
        values = grouped.get(metric, {})
        baseline = values.get("baseline", [])
        candidate = values.get("candidate", [])
        if not baseline or not candidate:
            continue
        baseline_median = statistics.median(baseline)
        candidate_median = statistics.median(candidate)
        regression_percent = _regression_percent(baseline_median, candidate_median)
        comparisons.append(
            _MetricComparison(
                metric=metric,
                baseline_median=baseline_median,
                candidate_median=candidate_median,
                regression_percent=regression_percent,
                max_regression_percent=max_regression_percent,
                min_regression_seconds=min_regression_seconds,
                regressed=(
                    regression_percent > max_regression_percent
                    and candidate_median - baseline_median > min_regression_seconds
                ),
            )
        )
    return tuple(comparisons)


def _regression_percent(baseline: float, candidate: float) -> float:
    """Return the candidate increase over baseline as a percentage."""
    if baseline <= 0:
        return 0.0 if candidate <= baseline else math.inf
    return (candidate / baseline - 1.0) * 100.0


def _all_results_valid(results: collections.abc.Iterable[_RunResult]) -> bool:
    """Return whether every expected baseline and candidate run completed correctly."""
    grouped: dict[str, list[_RunResult]] = {}
    for result in results:
        grouped.setdefault(result.label, []).append(result)
    return all(
        len(grouped.get(label, [])) > 0
        and all(result.correct for result in grouped[label])
        for label in ("baseline", "candidate")
    )


def _summary_payload(
    *,
    args: argparse.Namespace,
    revisions: dict[str, str],
    results: collections.abc.Iterable[_RunResult],
    comparisons: collections.abc.Iterable[_MetricComparison],
    passed: bool,
) -> dict[str, object]:
    """Build the durable JSON summary consumed by CI and humans."""
    return {
        "passed": passed,
        "benchmark": args.benchmark,
        "backend": args.backend,
        "repeats": args.repeats,
        "max_regression_percent": args.max_regression_percent,
        "min_regression_seconds": args.min_regression_seconds,
        "timeout_seconds": args.timeout_seconds,
        "revisions": revisions,
        "results": [result.to_json_dict() for result in results],
        "comparisons": [comparison.to_json_dict() for comparison in comparisons],
    }


def _write_summary(output_dir: pathlib.Path, summary: dict[str, object]) -> None:
    """Write JSON and Markdown summaries into the artifact directory."""
    _write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
    )
    lines = [
        "# NeuroDataBench lazynwb regression",
        "",
        f"**Result:** {'PASS' if summary['passed'] else 'FAIL'}",
        f"**Benchmark:** `{summary['benchmark']}`  ",
        f"**Backend:** `{summary['backend']}`  ",
        f"**Allowed regression:** `{summary['max_regression_percent']:.2f}%`",
        f"**Minimum absolute slowdown:** `{summary['min_regression_seconds']:.3f}s`",
        "",
        "| Metric | Baseline median | Candidate median | Change | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    comparisons = summary["comparisons"]
    if isinstance(comparisons, list) and comparisons:
        for comparison in comparisons:
            assert isinstance(comparison, dict)
            status = "FAIL" if comparison["regressed"] else "PASS"
            lines.append(
                f"| `{comparison['metric']}` | {comparison['baseline_median']:.3f}s | "
                f"{comparison['candidate_median']:.3f}s | "
                f"{comparison['regression_percent']:.2f}% | {status} |"
            )
    else:
        lines.append("| timing metrics | unavailable | unavailable | n/a | FAIL |")
    lines.extend(("", "Raw run directories are under `baseline/` and `candidate/`."))
    _write_text(output_dir / "summary.md", "\n".join(lines) + "\n")


def _print_summary(summary: dict[str, object]) -> None:
    """Print the compact CI summary and explain a failed gate."""
    print(f"NeuroDataBench regression: {'PASS' if summary['passed'] else 'FAIL'}")
    comparisons = summary["comparisons"]
    if isinstance(comparisons, list):
        for comparison in comparisons:
            assert isinstance(comparison, dict)
            print(
                f"  {comparison['metric']}: "
                f"{comparison['baseline_median']:.3f}s -> "
                f"{comparison['candidate_median']:.3f}s "
                f"({comparison['regression_percent']:.2f}%)"
            )
    if not summary["passed"]:
        print("See summary.md and the raw NeuroDataBench artifacts for diagnostics.")


def _write_text(path: pathlib.Path, text: str) -> None:
    """Write UTF-8 text after creating the containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
