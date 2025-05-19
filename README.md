# lazynwb


[![PyPI](https://img.shields.io/pypi/v/lazynwb.svg?label=PyPI&color=blue)](https://pypi.org/project/lazynwb/)
[![Python version](https://img.shields.io/pypi/pyversions/lazynwb)](https://pypi.org/project/lazynwb/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/lazynwb?logo=codecov)](https://app.codecov.io/github/AllenInstitute/lazynwb)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/bjhardcastle/lazynwb/publish.yml?label=CI/CD&logo=github)](https://github.com/bjhardcastle/lazynwb/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/bjhardcastle/lazynwb?logo=github)](https://github.com/bjhardcastle/lazynwb/issues)


# Purpose

## 1. Make NWB table access faster and/or consume less memory by reading only the data required, when it's needed

As of 2025 and `pynwb==3.0`, there are a couple of ways to access data stored in an NWB file as a
`DynamicTable` (e.g. `trials`, `units`):
-  get the `pandas` dataframe for the table and access the desired column
-  or access specific columns as arrays from disk

The schema for the  `units` table includes columns for list or nested-list type data, including
`spike_times`, and `waveform_mean` and `waveform_sd` which can be large for Neuropixels probes and
often not needed for analysis. Reading the entire table into memory may be unnecessary and,
especially when reading from NWBs stored in the cloud, can be slow.

Accessing individual columns as arrays, on the other hand, means we no longer have the convenience
of a DataFrame.

Ideally, we would filter our table based on metrics in some columns, then access the larger
columns for the filtered subset of rows, seemlessly with a single command.

To this end, `lazynwb.scan_nwb()` provides a
[`polars.LazyFrame()`](https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html)
interface to NWB tables, which
supports both predicate pushdown and
projection of columns. 

It also supports reading multiple NWB files in one operation, producing a
concatenated table:

```python
import lazynwb
import polars as pl

(
  lazynwb.scan_nwb(
    [nwb_path_0, nwb_path_1, ...],  # single path or iterable
    table_path='/units',             # or '/intervals/trials' etc
  )
  .filter(
    pl.col('activity_drift') <= 0.2,
    pl.col('amplitude_cutoff') <= 0.1,
    pl.col('presence_ratio') >= 0.7,
    pl.col('isi_violations_ratio') <= 0.5,
    pl.col('decoder_label') != 'noise',
  )
  .select('unit_id', 'location', 'spike_times', '_nwb_path', '_table_row_index')
  # _nwb_path and _table_row_index are not columns in the NWB table: they're added to identify source of each row in a table that spans multiple NWBs
)
>>> shape: (101, 4)
┌─────────┬─────────────────────────────────┬─────────────────────────────────┬──────────────┐
│ unit_id ┆ spike_times                     ┆ _nwb_path                       ┆ _table_index │
│ ---     ┆ ---                             ┆ ---                             ┆ ---          │
│ i64     ┆ list[f64]                       ┆ str                             ┆ u32          │
╞═════════╪═════════════════════════════════╪═════════════════════════════════╪══════════════╡
│ 193     ┆ [2722.628735, 2723.620493, … 4… ┆ /data/ecephys_702960_2024-03-1… ┆ 5            │
│ 23      ┆ [1784.801304, 1784.804037, … 3… ┆ /data/ecephys_725805_2024-07-1… ┆ 4            │
│ 0       ┆ [9.2712e6, 9.2712e6, … 9.2731e… ┆ /data/ecephys_737812_2024-08-0… ┆ 0            │
│ 300     ┆ [9.2713e6, 9.2714e6, … 9.2731e… ┆ /data/ecephys_737812_2024-08-0… ┆ 6            │
│ 19      ┆ [6115.424355, 6116.428649, … 7… ┆ /data/ecephys_702960_2024-03-1… ┆ 5            │
│ …       ┆ …                               ┆ …                               ┆ …            │
│ 437     ┆ [581.476385, 598.829113, … 331… ┆ /data/ecephys_666859_2023-06-1… ┆ 40           │
│ 439     ┆ [929.656482, 1134.993272, … 33… ┆ /data/ecephys_666859_2023-06-1… ┆ 41           │
│ 446     ┆ [626.940861, 661.785209, … 331… ┆ /data/ecephys_666859_2023-06-1… ┆ 42           │
│ 449     ┆ [618.939192, 618.991564, … 331… ┆ /data/ecephys_666859_2023-06-1… ┆ 43           │
│ 609     ┆ [594.415999, 646.51812, … 3312… ┆ /data/ecephys_666859_2023-06-1… ┆ 44           │
└─────────┴─────────────────────────────────┴─────────────────────────────────┴──────────────┘
```

## 2. Quickly provide a summary of the metadata for all NWB files in a project
```python
lazynwb.get_metadata_df(nwb_paths, as_polars=True)
>>>
```Getting metadata: 100%|█████████████████████| 252/252 [00:17<00:00, 14.51file/s]
shape: (252, 28)
┌────────────┬────────────┬───────────┬───────────┬───┬────────┬───────────┬───────────┬───────────┐
│ identifier ┆ session_st ┆ session_i ┆ session_d ┆ … ┆ weight ┆ strain    ┆ date_of_b ┆ _nwb_path │
│ ---        ┆ art_time   ┆ d         ┆ escriptio ┆   ┆ ---    ┆ ---       ┆ irth      ┆ ---       │
│ str        ┆ ---        ┆ ---       ┆ n         ┆   ┆ null   ┆ str       ┆ ---       ┆ str       │
│            ┆ datetime[μ ┆ str       ┆ ---       ┆   ┆        ┆           ┆ datetime[ ┆           │
│            ┆ s, UTC]    ┆           ┆ str       ┆   ┆        ┆           ┆ μs, UTC]  ┆           │
╞════════════╪════════════╪═══════════╪═══════════╪═══╪════════╪═══════════╪═══════════╪═══════════╡
│ 0514cf12-2 ┆ 2024-08-07 ┆ 713655_20 ┆ ecephys   ┆ … ┆ null   ┆ Sst-IRES- ┆ 2023-11-2 ┆ /data/dyn │
│ 41f-4ab2-a ┆ 19:03:44   ┆ 24-08-07  ┆ session   ┆   ┆        ┆ Cre;Ai32  ┆ 3         ┆ amicrouti │
│ ce9-1c2619 ┆ UTC        ┆           ┆ (day 3)   ┆   ┆        ┆           ┆ 08:00:00  ┆ ng_datacu │
│ …          ┆            ┆           ┆ with b…   ┆   ┆        ┆           ┆ UTC       ┆ be_…      │
│ 5c032dff-e ┆ 2024-12-06 ┆ 743199_20 ┆ ecephys   ┆ … ┆ null   ┆ VGAT-ChR2 ┆ 2024-05-1 ┆ /data/dyn │
│ 04f-4884-9 ┆ 19:06:17   ┆ 24-12-06  ┆ session   ┆   ┆        ┆ -YFP      ┆ 8         ┆ amicrouti │
│ 85d-055ac7 ┆ UTC        ┆           ┆ (day 4)   ┆   ┆        ┆           ┆ 07:00:00  ┆ ng_datacu │
│ …          ┆            ┆           ┆ with b…   ┆   ┆        ┆           ┆ UTC       ┆ be_…      │
│ 4a7e9fdb-4 ┆ 2022-09-27 ┆ 636397_20 ┆ ecephys   ┆ … ┆ null   ┆ C57BL6J(N ┆ 2022-06-0 ┆ /data/dyn │
│ fab-4052-a ┆ 18:36:50   ┆ 22-09-27  ┆ session   ┆   ┆        ┆ P)        ┆ 2         ┆ amicrouti │
│ 7fc-f2d109 ┆ UTC        ┆           ┆ (day 2)   ┆   ┆        ┆           ┆ 07:00:00  ┆ ng_datacu │
│ …          ┆            ┆           ┆ with b…   ┆   ┆        ┆           ┆ UTC       ┆ be_…      │
│ 9b4aab77-5 ┆ 2025-01-16 ┆ 744279_20 ┆ ecephys   ┆ … ┆ null   ┆ Sst-IRES- ┆ 2024-05-2 ┆ /data/dyn │
│ 021-43f3-9 ┆ 22:01:37   ┆ 25-01-16  ┆ session   ┆   ┆        ┆ Cre;Ai32  ┆ 5         ┆ amicrouti │
│ f18-b13291 ┆ UTC        ┆           ┆ (day 4)   ┆   ┆        ┆           ┆ 07:00:00  ┆ ng_datacu │
│ …          ┆            ┆           ┆ with b…   ┆   ┆        ┆           ┆ UTC       ┆ be_…      │
│ b0ba34cb-4 ┆ 2024-04-22 ┆ 706401_20 ┆ ecephys   ┆ … ┆ null   ┆ Sst-IRES- ┆ 2023-10-0 ┆ /data/dyn │
...
│ 971-495d-b ┆ 19:18:59   ┆ 25-03-18  ┆ session   ┆   ┆        ┆ -YFP      ┆ 6         ┆ amicrouti │
│ 6ed-dc7b08 ┆ UTC        ┆           ┆ (day 1)   ┆   ┆        ┆           ┆ 07:00:00  ┆ ng_datacu │
│ …          ┆            ┆           ┆ withou…   ┆   ┆        ┆           ┆ UTC       ┆ be_…      │
└────────────┴────────────┴───────────┴───────────┴───┴────────┴───────────┴───────────┴───────────┘
```

## 3. Quickly provide a summary of the contents of a single NWB file
```python
from lazynwb.tables import _get_internal_file_paths

_get_internal_file_paths(lazynwb.FileAccessor(nwb_paths[0]))
>>> {
  '/acquisition/frametimes_eye_camera/timestamps': <HDF5 dataset "timestamps": shape (267399,), type "<f8">,
  '/acquisition/frametimes_front_camera/timestamps': <HDF5 dataset "timestamps": shape (267204,), type "<f8">,
  '/acquisition/frametimes_side_camera/timestamps': <HDF5 dataset "timestamps": shape (267374,), type "<f8">,
  '/acquisition/lick_sensor_events/data': <HDF5 dataset "data": shape (2734,), type "<f8">,
  '/acquisition/lick_sensor_events/timestamps': <HDF5 dataset "timestamps": shape (2734,), type "<f8">,
  '/intervals/aud_rf_mapping_trials': <HDF5 group "/intervals/aud_rf_mapping_trials" (10 members)>,
  '/intervals/epochs': <HDF5 group "/intervals/epochs" (9 members)>,
  '/intervals/performance': <HDF5 group "/intervals/performance" (21 members)>,
  '/intervals/trials': <HDF5 group "/intervals/trials" (48 members)>,
  '/intervals/vis_rf_mapping_trials': <HDF5 group "/intervals/vis_rf_mapping_trials" (12 members)>,
  '/processing/behavior/dlc_eye_camera': <HDF5 group "/processing/behavior/dlc_eye_camera" (110 members)>,
  '/processing/behavior/eye_tracking': <HDF5 group "/processing/behavior/eye_tracking" (26 members)>,
  '/processing/behavior/facemap_front_camera/data': <HDF5 dataset "data": shape (267204, 500), type "<f4">,
  '/processing/behavior/facemap_front_camera/timestamps': <HDF5 dataset "timestamps": shape (267204,), type "<f8">,
  '/processing/behavior/facemap_side_camera/data': <HDF5 dataset "data": shape (267374, 500), type "<f4">,
  '/processing/behavior/facemap_side_camera/timestamps': <HDF5 dataset "timestamps": shape (267374,), type "<f8">,
  '/processing/behavior/licks/data': <HDF5 dataset "data": shape (2707,), type "<f8">,
  '/processing/behavior/licks/timestamps': <HDF5 dataset "timestamps": shape (2707,), type "<f8">,
  '/processing/behavior/lp_front_camera': <HDF5 group "/processing/behavior/lp_front_camera" (57 members)>,
  '/processing/behavior/lp_side_camera': <HDF5 group "/processing/behavior/lp_side_camera" (57 members)>,
  '/processing/behavior/quiescent_interval_violations/timestamps': <HDF5 dataset "timestamps": shape (131,), type "<f8">,
  '/processing/behavior/rewards/timestamps': <HDF5 dataset "timestamps": shape (130,), type "<f8">,
  '/processing/behavior/running_speed/data': <HDF5 dataset "data": shape (251998,), type "<f8">,
  '/processing/behavior/running_speed/timestamps': <HDF5 dataset "timestamps": shape (251998,), type "<f8">
 }
```

---

# Development
See instructions in https://github.com/bjhardcastle/lazynwb/CONTRIBUTING.md and the original template: https://github.com/bjhardcastle/copier-pdm-npc/blob/main/README.md

## notes

- hdf5 access seems to have a mutex lock that threads spend a long time waiting to
  acquire (with remfile)
- seems to slow down over time in single-threaded loop
    - on laptop, first 5 are fast (2-3 s per iteration) - successive iterations
      are much slower (>60 s)
