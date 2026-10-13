"""Unit tests for the NeuroDataBench regression gate."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "run_neurodatabench_regression.py"
_SPEC = importlib.util.spec_from_file_location("neurodatabench_regression", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load {_SCRIPT_PATH}")
_REGRESSION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REGRESSION
_SPEC.loader.exec_module(_REGRESSION)


def _result(label: str, repeat: int, total: float) -> object:
    """Build a valid synthetic run for comparison tests."""
    return _REGRESSION._RunResult(
        label=label,
        repeat=repeat,
        output_dir=pathlib.Path("results") / label / str(repeat),
        returncode=0,
        timed_out=False,
        total_seconds=total,
        setup_seconds=total / 2,
        submit_answers_seconds=total / 2,
        peak_rss_delta_bytes=None,
        network_received_bytes=None,
        correct=True,
    )


def test_compare_results_uses_medians_and_tolerance() -> None:
    """A slowdown beyond the configured tolerance fails the timing gate."""
    results = (
        _result("baseline", 1, 10.0),
        _result("baseline", 2, 12.0),
        _result("baseline", 3, 11.0),
        _result("candidate", 1, 12.0),
        _result("candidate", 2, 14.0),
        _result("candidate", 3, 13.0),
    )

    comparisons = _REGRESSION._compare_results(results, max_regression_percent=10.0)

    total = next(comparison for comparison in comparisons if comparison.metric == "total_seconds")
    assert total.baseline_median == pytest.approx(11.0)
    assert total.candidate_median == pytest.approx(13.0)
    assert total.regression_percent == pytest.approx(18.181818, rel=1e-5)
    assert total.regressed is True


def test_invalid_run_is_not_used_as_a_baseline() -> None:
    """Incorrect benchmark answers do not produce a false performance pass."""
    results = (
        _result("baseline", 1, 10.0),
        _result("candidate", 1, 1.0),
        _REGRESSION._RunResult(
            label="candidate",
            repeat=2,
            output_dir=pathlib.Path("results/candidate/2"),
            returncode=1,
            timed_out=False,
            total_seconds=1.0,
            setup_seconds=0.5,
            submit_answers_seconds=0.5,
            peak_rss_delta_bytes=None,
            network_received_bytes=None,
            correct=False,
        ),
    )

    assert _REGRESSION._all_results_valid(results) is False
    comparisons = _REGRESSION._compare_results(results, max_regression_percent=100.0)
    total = next(comparison for comparison in comparisons if comparison.metric == "total_seconds")
    assert total.candidate_median == pytest.approx(1.0)


def test_small_absolute_slowdown_requires_both_thresholds() -> None:
    """Percentage noise in short setup phases must not fail a full benchmark."""
    results = (_result("baseline", 1, 0.1), _result("candidate", 1, 0.2))
    comparisons = _REGRESSION._compare_results(results, max_regression_percent=15.0)
    assert not any(comparison.regressed for comparison in comparisons)
    strict = _REGRESSION._compare_results(
        results, max_regression_percent=15.0, min_regression_seconds=0.0
    )
    assert all(comparison.regressed for comparison in strict)
