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

With `pynwb`, each container type has its own access pattern - dot attributes,
dictionary keys, and method calls get mixed together depending on where the data
lives in the file:

```python
# pynwb
from pynwb import NWBHDF5IO
io = NWBHDF5IO('my_file.nwb', 'r')
nwb = io.read()

nwb.units.to_dataframe()
nwb.trials.to_dataframe()
nwb.processing['behavior']['eye_tracking'].to_dataframe()
nwb.processing['ophys']['Fluorescence']['RoiResponseSeries'].data[:]
```

You need to know whether something is a property, a dict-like container,
or a DynamicTable, and chain the right combination for each.

With `lazynwb`, every table is accessed the same way - by its internal path:

```python
# lazynwb
import lazynwb

lazynwb.get_df('my_file.nwb', '/units')
lazynwb.get_df('my_file.nwb', '/intervals/trials')
lazynwb.get_df('my_file.nwb', '/processing/behavior/eye_tracking')
```

And for time series data:
```python
ts = lazynwb.get_timeseries('my_file.nwb', '/processing/ophys/Fluorescence/RoiResponseSeries')
ts.data[:]
```

No need to know the container class or chain attribute lookups. The same path
works for any file, any backend (HDF5 or Zarr), local or remote, and extends to
reading across multiple files in one call.

`lazynwb` can also read only the columns and rows you request, rather than loading the entire table into memory first. This matters for tables with list- or array- like columns, like the `units` table, 
where `spike_times` and `waveform_mean` can be very large compared to other single-value metrics columns.

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

config.use_obstore = True                          # use obstore for S3/GCS/Azure (default: True)
config.use_remfile = False                         # use remfile for HTTP byte-range requests (default: False)
config.fsspec_storage_options = {"anon": True}     # e.g. anonymous S3 access
config.disable_cache = False                       # disable FileAccessor caching (default: False)
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
