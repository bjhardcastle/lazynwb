# lazynwb


[![PyPI](https://img.shields.io/pypi/v/lazynwb.svg?label=PyPI&color=blue)](https://pypi.org/project/lazynwb/)
[![Python version](https://img.shields.io/pypi/pyversions/lazynwb)](https://pypi.org/project/lazynwb/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/lazynwb?logo=codecov)](https://app.codecov.io/github/AllenInstitute/lazynwb)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/bjhardcastle/lazynwb/publish.yml?label=CI/CD&logo=github)](https://github.com/bjhardcastle/lazynwb/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/bjhardcastle/lazynwb?logo=github)](https://github.com/bjhardcastle/lazynwb/issues)

Efficient read-only access to tables, time series and metadata across multiple local or cloud-hosted NWB files simultaneously, without loading entire files into memory.

```
pip install lazynwb
```

---

## Why lazynwb

### Work with a project's worth of NWB files

Seamlessly read and concatenate tables across sessions:

```python
import lazynwb

# read the units table from every session into a single DataFrame
df = lazynwb.get_df(
    ['session_1.nwb', 'session_2.nwb', 'session_3.nwb'],
    '/intervals/trials',
)
# each row keeps its source file in the `_nwb_path` column
```

### Efficient dataframe access with projection and predicate pushdown

NWB tables like `/units` mix single-value metric columns with large array
columns (`spike_times`, `waveform_mean`). With `pynwb`, accessing a dataframe means
loading everything.

`lazynwb` provides a Polars plugin that returns a LazyFrame backed by the NWB
file. Only the columns and rows you actually use are loaded:

```python
import lazynwb
import polars as pl

lf = lazynwb.scan_nwb('s3://bucket/session.nwb', '/units')

df = (
    lf
    .filter(
        pl.col('presence_ratio') >= 0.95,    # predicate pushdown: skip non-matching rows
        pl.col('location') == 'VISp',
    )
    .select('unit_id', 'spike_times')        # projection pushdown: only fetch these columns
    .collect()
)
```

For queries that don't need all columns on data that's stored in the cloud, `lazynwb` can turn an operations that takes minutes into one that takes seconds.

For more details, see the [lazy API guide in the Polars documentation](https://docs.pola.rs/user-guide/concepts/lazy-api/).

### A simple, consistent API

One interface for local files, S3/GCS/Azure, HDF5 and Zarr: no extra imports
or backend-specific code:

```python
import lazynwb

# local HDF5
lazynwb.get_df('my_file.nwb', '/trials')

# remote Zarr
lazynwb.get_df('s3://bucket/session.nwb', '/trials')

# DANDI archive
lf = lazynwb.scan_dandiset('000363', '/trials')
```

### Basic benchmarks

Streaming a single NWB file over HTTPS
([Steinmetz 2019](https://dandiarchive.org/dandiset/000017), 312 MB HDF5) with a laptop on a typical home internet connection:

**Tables**: reading `/intervals/trials` (214 rows, no array columns):

| Method | Time |
|---|---|
| `pynwb` `.to_dataframe()` | 8.3 s |
| `lazynwb.get_df` | 5.1 s |
| `lazynwb.scan_nwb` | 4.1 s |

For a table with no array columns (relatively quick to load), all approaches read the same volume of data.

**Tables**: reading `/units` (1085 rows, includes large `spike_times` and `waveform_mean` arrays):

| Method | Time | What it reads |
|---|---|---|
| `pynwb` `.to_dataframe()` | 231 s | all columns (no choice) |
| `lazynwb.get_df(..., exclude_array_columns=False)`  | 282 s | all columns (equivalent) |
| `lazynwb.get_df(..., exclude_array_columns=True)` | 6 s | scalar columns only |
| `lazynwb.scan_nwb` (filter + select) | 10 s | filter on scalar columns, then fetch `spike_times` |

When reading all columns, `lazynwb` and `pynwb` take
roughly the same time: if you need all data in memory, there's no reason to use `lazynwb` here. The difference is
that `pynwb` always reads everything, while `lazynwb` lets you choose.

**TimeSeries**: `lick_times` (3190 samples, full download):

| Method | Time |
|---|---|
| `pynwb` | 7.3 s |
| `lazynwb.get_timeseries` | 6.1 s |
| `lazynwb.get_timeseries` (metadata only) | 6.2 s |

Both download the same data, and `pynwb` also supports lazy access to time series data. The only advantage here is the consistent API.

See [benchmarks/streaming_benchmark.py](benchmarks/streaming_benchmark.py)
to reproduce or run against your own files:
```
python benchmarks/streaming_benchmark.py [NWB_PATH]
```

## Why not to use lazynwb

- some convenience features of `pynwb` will not be available, for example object references in tables
- incomplete coverage of the NWB spec. Focussed on the core metadata, `TimeSeries` and `DynamicTable`, and tested primarily on ecephys files. Please file an issue if you need support for a particular container.
- you need to write NWB files

---

## Quick start

```python
import lazynwb

# read the trials table as a pandas DataFrame
df = lazynwb.get_df('my_file.nwb', '/intervals/trials')
```

Use `get_internal_paths` to find available paths if you're not sure what's in a file:
```python
lazynwb.get_internal_paths('my_file.nwb')
```

---

## Reading tables

### As a pandas or polars DataFrame (`get_df`)

Returns a pandas DataFrame by default:
```python
df = lazynwb.get_df('my_file.nwb', '/units')
```

Return a polars DataFrame instead:
```python
df = lazynwb.get_df('my_file.nwb', '/units', as_polars=True)
```

Select specific columns:
```python
df = lazynwb.get_df('my_file.nwb', '/units', include_column_names=['unit_id', 'location'])
```

Exclude specific columns:
```python
df = lazynwb.get_df('my_file.nwb', '/units', exclude_column_names=['waveform_mean'])
```

Large array columns like `spike_times` and `waveform_mean` are excluded by default
(`exclude_array_columns=True`). Include them explicitly:
```python
df = lazynwb.get_df('my_file.nwb', '/units', exclude_array_columns=False)
```

Read a table across multiple files into a single DataFrame:
```python
df = lazynwb.get_df(
    ['file_1.nwb', 'file_2.nwb', 'file_3.nwb'],
    '/intervals/trials',
)
```

Each row gets `_nwb_path`, `_table_path` and `_table_index` columns to identify its
source file and original row index.

### As a Polars LazyFrame (`scan_nwb`)

`scan_nwb` returns a `polars.LazyFrame` that reads data on demand. Only the
columns and rows you actually use are fetched from disk or the network, which
makes it useful for large files or files on cloud storage.

```python
import lazynwb
import polars as pl

lf = lazynwb.scan_nwb('my_file.nwb', '/units')

# filter rows and select columns - only the needed data is read
df = (
    lf
    .filter(pl.col('presence_ratio') >= 0.9)
    .select('unit_id', 'location', 'spike_times')
    .collect()
)
```

Read across multiple files:
```python
lf = lazynwb.scan_nwb(
    ['file_1.nwb', 'file_2.nwb'],
    '/units',
)
df = (
    lf
    .filter(
        pl.col('amplitude_cutoff') <= 0.1,
        pl.col('isi_violations_ratio') <= 0.5,
    )
    .select('unit_id', 'location', 'spike_times', '_nwb_path')
    .collect()
)
```

Control schema inference when files have slightly different column types:
```python
lf = lazynwb.scan_nwb(
    nwb_paths,
    '/units',
    infer_schema_length=5,               # only read first 5 files for schema
    schema_overrides={'unit_id': pl.Int64},  # force a column type
)
```

There's also `read_nwb`, which is the same as `scan_nwb(...).collect()`:
```python
df = lazynwb.read_nwb(nwb_paths, '/units')  # returns pl.DataFrame
```

Note: `pl.DataFrame` has a `.to_pandas()` method.

### Using `LazyNWB` (PyNWB-like interface)

Access tables and metadata from a single file with familiar attribute names:
```python
nwb = lazynwb.LazyNWB('my_file.nwb')

# tables (returned as pandas DataFrames)
nwb.trials
nwb.units
nwb.epochs
nwb.electrodes

# metadata
nwb.session_id
nwb.session_start_time
nwb.session_description
nwb.identifier
nwb.experiment_description
nwb.experimenter
nwb.lab
nwb.institution
nwb.keywords
```

Subject metadata:
```python
nwb.subject.age
nwb.subject.sex
nwb.subject.species
nwb.subject.genotype
nwb.subject.subject_id
nwb.subject.strain
nwb.subject.date_of_birth
```

Get a table as polars:
```python
df = nwb.get_df('/units', as_polars=True)
```

Get a summary of everything in the file:
```python
nwb.describe()
# {'identifier': '...', 'session_id': '...', ..., 'paths': ['/acquisition/...', '/units', ...]}
```

---

## Time series

Get a single time series by searching for a name:
```python
ts = lazynwb.get_timeseries('my_file.nwb', search_term='running_speed')

ts.data          # h5py.Dataset or zarr.Array (lazy - not loaded until sliced)
ts.timestamps    # h5py.Dataset or zarr.Array
ts.unit          # e.g. 'cm/s'
ts.rate          # sampling rate, if available
ts.description
```

Get a time series by exact internal path:
```python
ts = lazynwb.get_timeseries('my_file.nwb', exact_path=True, search_term='/acquisition/lick_sensor_events')
```

Get all time series in the file:
```python
all_ts = lazynwb.get_timeseries('my_file.nwb', match_all=True)
# dict: {'/acquisition/lick_sensor_events': TimeSeries(...), '/processing/behavior/running_speed': TimeSeries(...), ...}
```

Also available on a `LazyNWB` object:
```python
nwb = lazynwb.LazyNWB('my_file.nwb')
ts = nwb.get_timeseries('running_speed')
```

---

## Metadata across files

Get session and subject metadata for many files at once:
```python
df = lazynwb.get_metadata_df(nwb_paths)  # pandas DataFrame
```
```python
df = lazynwb.get_metadata_df(nwb_paths, as_polars=True)  # polars DataFrame
```

Returns columns including `identifier`, `session_id`, `session_start_time`,
`session_description`, `subject_id`, `age`, `sex`, `species`, `genotype`,
`strain`, `date_of_birth`, `_nwb_path`, and more.

---

## File contents and schema

### Discover internal paths

See what's inside an NWB file:
```python
paths = lazynwb.get_internal_paths('my_file.nwb')
# {'/acquisition/lick_sensor_events/data': <HDF5 dataset ...>,
#  '/intervals/trials': <HDF5 group ...>,
#  '/units': <HDF5 group ...>,
#  ...}
```

### Get table schema

Get the unified column names and types for a table across multiple files:
```python
schema = lazynwb.get_table_schema(nwb_paths, '/intervals/trials')
# OrderedDict([('condition', String), ('id', Int64), ('start_time', Float64), ...])
```

Uses polars (Arrow) data types.

---

## Format conversion

Export NWB tables to other file formats with `convert_nwb_tables`.

Supported formats: `parquet`, `csv`, `json`, `excel`, `feather`, `arrow`, `avro`, `delta`.

```python
output_paths = lazynwb.convert_nwb_tables(
    nwb_paths,
    output_dir='./output',
    output_format='parquet',
)
# {'/intervals/trials': PosixPath('./output/trials.parquet'),
#  '/units': PosixPath('./output/units.parquet')}
```

Pass format-specific options via keyword arguments:
```python
# parquet with zstd compression
lazynwb.convert_nwb_tables(nwb_paths, './output', output_format='parquet', compression='zstd')

# csv with custom separator
lazynwb.convert_nwb_tables(nwb_paths, './output', output_format='csv', separator='\t')

# json, pretty-printed
lazynwb.convert_nwb_tables(nwb_paths, './output', output_format='json', pretty=True)
```

Only export tables present in all files:
```python
lazynwb.convert_nwb_tables(nwb_paths, './output', min_file_count=len(nwb_paths))
```

Use full internal paths as filenames (e.g. `intervals_trials.parquet` instead of `trials.parquet`):
```python
lazynwb.convert_nwb_tables(nwb_paths, './output', full_path=True)
```

---

## SQL queries

Register all tables from NWB files as a Polars SQL context:
```python
ctx = lazynwb.get_sql_context(nwb_paths)
df = ctx.execute("SELECT unit_id, location FROM units WHERE presence_ratio > 0.9").collect()
```

---

## Cloud and remote files

All functions accept S3, GCS, Azure Blob Storage and HTTP/HTTPS paths
in addition to local file paths:

```python
# S3
df = lazynwb.get_df('s3://my-bucket/data/file.nwb', '/units')

# Google Cloud Storage
df = lazynwb.get_df('gs://my-bucket/data/file.nwb', '/units')

# Azure Blob Storage
df = lazynwb.get_df('az://my-container/data/file.nwb', '/units')

# HTTP/HTTPS
df = lazynwb.get_df('https://example.com/data/file.nwb', '/units')
```

Configure cloud access via `lazynwb.file_io.config`:
```python
from lazynwb.file_io import config

config.use_obstore = True                         # use obstore for S3/GCS/Azure (default: False)
config.use_remfile = False                        # use remfile for HTTP byte-range requests (default: True)
config.anon = True                                # anonymous access across backends
config.fsspec_storage_options = {"region": "us-west-2"}  # backend-specific extras if needed
config.disable_cache = False                      # disable FileAccessor caching (default: False)
```

---

## DANDI archive

Scan a table across all NWB files in a DANDI dandiset:
```python
lf = lazynwb.scan_dandiset(
    dandiset_id='000363',
    table_path='/units',
    version='0.231012.2129',
    max_assets=10,  # limit number of files (useful for testing)
)
df = lf.collect()
```

Filter which assets to include:
```python
lf = lazynwb.scan_dandiset(
    '000363',
    '/units',
    asset_filter=lambda asset: 'probe' in asset['path'],
)
```

Get S3 URLs for all NWB files in a dandiset:
```python
urls = lazynwb.get_dandiset_s3_urls('000363')
```

Open a single DANDI asset:
```python
accessor = lazynwb.from_dandi_asset(
    dandiset_id='000363',
    asset_id='21c622b7-6d8e-459b-98e8-b968a97a1585',
)
```

---

## Internal columns

When reading tables from multiple files, three columns are added automatically:

| Column | Description |
|---|---|
| `_nwb_path` | Path to the source NWB file |
| `_table_path` | Internal path of the table (e.g. `/units`) |
| `_table_index` | Row index in the original table |

These are available as constants: `lazynwb.NWB_PATH_COLUMN_NAME`,
`lazynwb.TABLE_PATH_COLUMN_NAME`, `lazynwb.TABLE_INDEX_COLUMN_NAME`.

---
