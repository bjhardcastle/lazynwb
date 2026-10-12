# dev3 to dev6 performance investigation

The `dev6` branch at `2bbcf8d` identifies as `1.0.0.dev8` and includes changes
after the published dev6 release. Compare these separately:

| Source | Revision |
| --- | --- |
| Published dev3 | `01b51473aa2fb5b2562579efbca1514a35cbcd49` |
| Published dev6 | `15f4792` |
| Current dev6 branch before this investigation | `2bbcf8d` |

## Reproduced causes

`53a40d2` fixed disabling SQLite caching but also removed accessor reuse.
`LazyNWB` and `Subject` still ask `_get_accessor` for every metadata property.
A single `get_metadata_df` call therefore opened its file 28 times instead of
once. This is reproducible without network timing: the regression test reads
real fixture metadata and counts backend opens. It failed with 27 opens for
session plus subject fields before the fix.

Restore reuse in the private helper while preserving independently constructed
`FileAccessor` handles and the SQLite `disable_cache` fix. Bound the cache to
128 entries, distinguish backend/authentication configuration, reopen closed
handles, and release cached references without closing borrowed datasets.
Catalog readers that close their own handles must construct independent
accessors rather than borrow from this cache.

`af6947e` introduced bounded scan fetching. Its correctness fixes include
source order, duplicate sources, and limits after filtering. However, length
planning read catalog snapshots without retaining them for materialization,
particularly on Zarr. The new regression test reproduces two catalog reads
for one local Zarr scan. Retaining the snapshot in the scan's existing mapping
reduces this to one while preserving explicit batch limits.

## Experiment audit

Keep credentials (`2af2524`), path pruning (`3b11b4c`, `de49e2b`), sparse schema
alignment (`e6ab9f9`), the lazy-scan correctness fixes (`af6947e`), the opt-in
single-file batching feature (`c8eb777`), and compound Zarr dtype parsing
(`2bbcf8d`). Preserve the pre-existing uncommitted DANDI authentication and
array-schema promotion fixes.

The broad HDF5 experiment sequence starts at `0583705`: direct dataset reads,
span-native indexed reads, sparse ragged indexes and timestamps, shared range
reads, sparse chunk indexes, Arrow materialization, and range-backed
TimeSeries/exact discovery. An isolated candidate restored the six affected
implementation modules to `c8eb777` while retaining later unrelated fixes.
With accessor reuse restored in both candidates, two alternating HDF5 runs
measured 11.71 s median with this stack and 14.84 s without it. All answers
were correct. Blanket removal is therefore not supported by this workload.
Restoring the older TimeSeries implementation also changes its returned data
accessors back to h5py; that requires the interface check specified in AGENTS.md.

## Measurement conditions

Measurements were taken on Windows using the local NeuroDataBench checkout
at `8dec896` with its pre-existing uncommitted implementation/runner changes.
Both lazynwb sources used the same existing NeuroDataBench Python environment
(Polars 1.38.1, obstore 0.9.4, pydantic 2.13.4). The bootstrap printed each
imported lazynwb source path. These are checkout comparisons, not comparisons
of independently resolved PyPI dependency sets.

Each NeuroDataBench run used a fresh process and catalog path, anonymous S3,
the five packaged dynamic routing sessions, and validated all four answers.
Paired repeats alternate baseline/candidate order. Small setup times are
noisy: the original percentage-only gate failed on a change of roughly
20 milliseconds even when total runtime was within its 15% threshold. The
runner now requires both a 15% increase and a 250 ms absolute increase by
default. Both thresholds are explicit and configurable, with raw timings
preserved. This policy change is independent of the library fixes.

Initial paired runs reproduced HDF5 dev3 versus published dev6 at 12.83 versus
19.22 s, and Zarr dev3 versus current HEAD at 39.27 versus 52.92 s. These
single pairs establish symptoms rather than precise performance ratios.

The remote HDF5 metadata workload returned identical content hashes for all
three sources, across cold and repeated reads:

| Source | First read (s) | Repeated read (s) | Opens, first/repeated |
| --- | ---: | ---: | ---: |
| dev3 | 5.26 | 0.081 | 5 / 0 |
| Unmodified branch HEAD | 56.43 | 57.88 | 140 / 140 |
| Accessor reuse fix | 2.93 | 0.124 | 5 / 0 |

The equivalent Zarr metadata read also returned identical hashes:

| Source | First read (s) | Repeated read (s) | Opens, first/repeated |
| --- | ---: | ---: | ---: |
| dev3 | 5.28 | 2.47 | 5 / 0 |
| Unmodified branch HEAD | 10.19 | 6.75 | 140 / 140 |
| Accessor reuse fix | 5.41 | 2.59 | 5 / 0 |

This metadata timing sample is diagnostic, not a statistical performance gate;
the local correctness suite overlapped part of the unmodified HEAD run. The
file-open counts are deterministic, and its slow repeated read completed after
the correctness suite. Some later Zarr repeats overlapped merge validation;
their medians are descriptive. A separate smoke comparison of the merged
candidate runs after all tests/builds, without competing agent work.

In a controlled Zarr pair, retaining scan snapshots reduced total time from
45.54 to 26.27 s, with correct answers. Both variants in that pair used the
same 100,000-row default batch prototype, isolating snapshot retention. The
initial fix preserved the existing 1,000-row default and explicit batch-size
behavior; changing the default alone did not resolve the regression. The first
three-pair validation then isolated residual HDF5 time to those small default
batches (trials-query medians 2.66 versus 1.62 s). Normal scans now use the usual
100,000-row Polars hint when no hint is provided; `low_memory=True` retains the
1,000-row default, and explicit batch-size hints are unchanged.

Raw logs, timing, validation, memory, network, and comparison artifacts are
under `.scratch/perf-results/`. See `docs/neurodatabench-regression.md` for
repeatable checkout comparisons and `benchmarks/metadata_benchmark.py` for
metadata timing, file-open counts, and result hashes.

The portable record in `docs/performance/dev3-comparison.json` retains individual
final-run timings, answer validation, metadata hashes, and source hashes.

## Validation

The final library changes pass 386 local tests (7 skipped, 6 integration tests
deselected, 4 expected failures). The three regression-harness tests also pass,
including the new absolute-noise-floor check. Focused lint checks pass; the
existing unrelated annotation/complexity findings in `file_io.py` remain.

Three alternating Zarr pairs before the default-batch adjustment passed with
all answers correct: 32.404 s
baseline median versus 29.858 s candidate median (7.86% faster). This also passed
the original percentage-only gate, without using the added absolute floor.

The same checkpoint's HDF5 median was 11.777 s for dev3 versus 13.053 s for
the candidate (10.83% slower), within the 15% gate but motivating the final
default-batch adjustment. The 27 scan/catalog tests passed after that change.

After adjusting the default, three HDF5 pairs passed: 11.446 s baseline median
versus 10.672 s candidate median (6.76% faster). Three Zarr pairs also passed
at 42.583 versus 25.101 s; merge validation overlapped the later baseline runs,
so do not interpret that 41% difference as an isolated speedup estimate.

The merged release candidate passes 389 tests (7 skipped, 6 deselected,
4 expected failures), including main's anonymous-Zarr regression coverage.
`uv lock --offline --check` passes. Both the 1.0.0 wheel and source distribution
build, and an import from the wheel reports version 1.0.0.

The final smoke checks used the exact merged release candidate, after all
tests/builds completed. Both passed with all answers correct:

| Benchmark | dev3 (s) | Merged 1.0.0 candidate (s) |
| --- | ---: | ---: |
| Dynamic routing HDF5 | 11.497 | 9.624 |
| Dynamic routing Zarr | 37.053 | 29.244 |

These are single pairs, supplementary to the repeated comparisons above.

## Release checkpoint

`codex/release-1.0.0` is prepared in `.scratch/release-candidate`. It merges
dev6 with current origin/main, resolves all four conflicts, preserves main's
anonymous-Zarr fix using the obstore fsspec adapter, and declares version 1.0.0.
It includes the pre-existing DANDI authentication/schema fixes and benchmark
tooling, while leaving the original working tree and unrelated files intact.

The previous workflow automatically bumped and published on pushes to main.
The prepared workflow runs tests on main and publishes only version tags,
preventing a merge from unexpectedly bumping/publishing a different version.
After review and merge, tag the intended commit as `v1.0.0` to invoke publishing.
No PyPI publication or release tag was created during this investigation.
