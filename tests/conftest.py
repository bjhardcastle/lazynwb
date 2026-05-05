from __future__ import annotations

import collections.abc
import gc
import os
import pathlib
import random
import shutil
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

# Units class is implicitly used by nwbfile.add_unit, but not directly instantiated here.
from pynwb.misc import Units  # Uncommented: Ensure Units is imported

import lazynwb
import tests._dandi_sample as dandi_sample

RESET_FILES = False
CLEANUP_FILES = False
OVERRIDE_DIR: None | pathlib.Path = (
    None if CLEANUP_FILES else pathlib.Path(__file__).parent / "files" / "nwb_files"
)


def _reset_nwb_files(dir_path: pathlib.Path) -> None:
    lazynwb.clear_cache()
    gc.collect()
    if dir_path.exists():
        shutil.rmtree(dir_path)

if OVERRIDE_DIR is not None and RESET_FILES:
    _reset_nwb_files(OVERRIDE_DIR)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("lazynwb")
    group.addoption(
        dandi_sample._DANDI_SAMPLE_OPTION,
        action="store_true",
        default=False,
        help=(
            "run opt-in DANDI:001637 sample integration tests; alternatively set "
            f"{dandi_sample._DANDI_SAMPLE_ENV_VAR}=1"
        ),
    )


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Function],
) -> None:
    """Modify the order of tests"""
    _skip_dandi_sample_tests_if_disabled(config, items)
    # Run this test last because it closes FileAccessor instances reused for the session.
    cache_clearing_test = next(
        (i for i in items if i.name == "test_file_accessor_clearing"), None
    )
    if cache_clearing_test is not None:
        items.remove(cache_clearing_test)
        items[:] = [*items, cache_clearing_test]


def _skip_dandi_sample_tests_if_disabled(
    config: pytest.Config,
    items: list[pytest.Function],
) -> None:
    if _dandi_sample_tests_enabled(config):
        return
    skip_marker = pytest.mark.skip(
        reason=(
            f"set {dandi_sample._DANDI_SAMPLE_ENV_VAR}=1 or pass "
            f"{dandi_sample._DANDI_SAMPLE_OPTION} to run DANDI:001637 sample "
            "integration tests"
        )
    )
    for item in items:
        if "dandi_sample" in item.keywords:
            item.add_marker(skip_marker)


def _dandi_sample_tests_enabled(config: pytest.Config) -> bool:
    return bool(config.getoption(dandi_sample._DANDI_SAMPLE_OPTION)) or (
        os.environ.get(dandi_sample._DANDI_SAMPLE_ENV_VAR) == "1"
    )


def _add_nwb_file_content(nwbfile: NWBFile, unique_id_suffix: str = "") -> NWBFile:
    """
    Populates an NWBFile object with predefined content.
    """
    nwbfile.subject = Subject(
        subject_id=f"sub001_{unique_id_suffix}",
        species="Mus musculus",
        sex="M",
        age="P90D",
        description="Test subject",
    )

    # Processing module for running data
    nwbfile.create_processing_module(
        name="behavior", description="processed behavioral data"
    )
    num_samples = 120 + random.randint(1, 10)  # Randomize number of samples
    timestamps = np.linspace(0, 12, num_samples)  # 12 seconds of data
    running_speed_data = np.cos(timestamps) * 0.5 + 0.5  # Dummy speed data (0 to 1 m/s)
    running_speed_ts_0 = TimeSeries(
        name="running_speed_with_timestamps",
        data=running_speed_data,
        unit="m/s",
        timestamps=timestamps,
        description="forward running speed on wheel",
    )
    nwbfile.processing["behavior"].add(running_speed_ts_0)

    # Add a second timeseries with start time and rate
    running_speed_ts_1 = TimeSeries(
        name="running_speed_with_rate",
        data=running_speed_data,
        unit="m/s",
        starting_time=2.0,
        rate=60.0,  # 1000 Hz
        description="forward running speed on wheel",
    )
    nwbfile.processing["behavior"].add(running_speed_ts_1)

    # Units table
    # Create the units table with description first, before adding any units.
    nwbfile.units = Units(
        name="units",
    )
    # Columns 'spike_times', 'waveform_mean', 'obs_intervals' are standard in pynwb.misc.Units.
    # nwbfile.add_unit will add data to the nwbfile.units table created above.
    num_units = 4
    nwbfile.units.add_column(name="structure", description="brain region")
    for i in range(num_units):
        # Spike times for this unit (ragged array)
        spike_times_data = np.sort(np.random.uniform(0, 12, np.random.randint(30, 60)))
        # Waveform mean for this unit (assuming fixed length for simplicity)
        waveform_mean_data = np.random.randn(25, 384)
        # Observation intervals for this unit (ragged array of [start, stop] pairs)
        if i % 2 == 0:
            obs_intervals_data = np.array([[0.0, 5.5], [6.5, 12.0]])
        else:
            obs_intervals_data = np.array([[2.1, 7.5]])

        nwbfile.add_unit(
            structure='VISp',
            spike_times=spike_times_data,
            waveform_mean=waveform_mean_data,
            obs_intervals=obs_intervals_data,
            # id is managed automatically
        )

    # Trials table - set description during creation
    trials_table = TimeIntervals(name="trials")
    trials_table.add_column(name="condition", description="experimental condition")

    num_trials = 6
    for i in range(num_trials):
        start_time = i * 2.0 + 0.05  # e.g., 0.05, 2.05, 4.05, ...
        stop_time = start_time + 1.8  # e.g., 1.85, 3.85, 5.85, ...
        trials_table.add_row(
            start_time=start_time,
            stop_time=stop_time,
            condition=f"{chr(65+i)}",  # A, B, C...
        )
    nwbfile.trials = trials_table

    # Epochs table - set description during creation
    epochs_table = TimeIntervals(
        name="epochs",
        description="experimental epochs",
    )
    # Add epochs to the table
    num_epochs = 3
    for i in range(num_epochs):
        start_time = i * 4.0 + 0.1  # e.g., 0.1, 4.1, 8.1
        stop_time = start_time + 3.5  # e.g., 3.6, 7.6, 11.6
        epochs_table.add_row(
            start_time=start_time,
            stop_time=stop_time,
            tags=[f"tag_{i+1}", "task"],
        )
    nwbfile.epochs = epochs_table

    return nwbfile


@pytest.fixture(scope="session")
def local_hdf5_path(
    local_hdf5_paths: list[pathlib.Path],
) -> collections.abc.Iterator[pathlib.Path]:
    """Provides a path to a single HDF5 NWB file."""
    # Use the first file in the list of HDF5 files
    yield local_hdf5_paths[0]


@pytest.fixture(scope="session")
def local_hdf5_paths() -> collections.abc.Iterator[list[pathlib.Path]]:
    """Provides a path to a directory with multiple HDF5 NWB files."""
    test_dir = OVERRIDE_DIR or (
        pathlib.Path(tempfile.gettempdir()) / f"lazynwb_test_dir_{uuid.uuid4().hex}"
    )
    test_dir.mkdir(exist_ok=True, parents=True)

    # Create 2 NWB files
    returned_paths = []
    for i in range(2):
        file_path = test_dir / f"test_hdf5_{i}.nwb"
        returned_paths.append(file_path)
        if file_path.exists() and not RESET_FILES:
            continue

        nwbfile_obj = NWBFile(
            session_description=f"Test hdf5 NWB file {i}",
            identifier=str(uuid.uuid4()),
            session_start_time=datetime.now(tz=timezone.utc),
            keywords=["test", "experiment", "lazynwb"],
            experimenter=["User A", "User B"],
            institution="Test Institution",
            file_create_date=datetime.now(tz=timezone.utc),
        )
        # Populate the NWBFile with content using the helper function
        nwbfile_obj = _add_nwb_file_content(
            nwbfile_obj, unique_id_suffix=f"hdf5_dir_{i}"
        )

        with NWBHDF5IO(str(file_path.absolute()), "w") as io:
            io.write(nwbfile_obj)

    yield returned_paths

    if CLEANUP_FILES:
        _reset_nwb_files(test_dir)

@pytest.fixture(scope="session")
def local_zarr_path(
    local_zarr_paths: list[pathlib.Path],
) -> collections.abc.Iterator[pathlib.Path]:
    """Provides a path to a single Zarr NWB store (directory)."""
    yield local_zarr_paths[0]


@pytest.fixture(scope="session")
def local_zarr_paths() -> collections.abc.Iterator[list[pathlib.Path]]:
    """Provides a path to a directory with multiple Zarr NWB files."""
    test_dir = OVERRIDE_DIR or (
        pathlib.Path(tempfile.gettempdir()) / f"lazynwb_zarr_dir_{uuid.uuid4().hex}"
    )
    test_dir.mkdir(exist_ok=True, parents=True)

    # Create 2 Zarr files
    returned_paths = []
    for i in range(2):
        zarr_store_path = test_dir / f"test_zarr_{i}.nwb.zarr"
        returned_paths.append(zarr_store_path)
        if zarr_store_path.exists() and not RESET_FILES:
            continue

        nwbfile_obj = NWBFile(
            session_description=f"Test Zarr NWB file {i}",
            identifier=str(uuid.uuid4()),
            session_start_time=datetime.now(tz=timezone.utc),
        )
        # Populate the NWBFile with content using the helper function
        nwbfile_obj = _add_nwb_file_content(
            nwbfile_obj, unique_id_suffix=f"zarr_dir_{i}"
        )
        from hdmf_zarr import NWBZarrIO
        with NWBZarrIO(str(zarr_store_path.absolute()), "w") as io:
            io.write(nwbfile_obj)

    yield returned_paths

    if CLEANUP_FILES:
        _reset_nwb_files(test_dir)


@pytest.fixture(scope="session")
def dandi_001637_sample_assets() -> tuple[dandi_sample._DandiSampleAsset, ...]:
    return dandi_sample._DANDI_SAMPLE_ASSETS


@pytest.fixture(scope="session")
def _dandi_001637_resolved_sample_assets_cache() -> tuple[
    dandi_sample._DandiSampleResolvedAsset, ...
]:
    return dandi_sample._resolve_sample_asset_urls()


@pytest.fixture
def dandi_001637_resolved_sample_assets(
    _dandi_001637_resolved_sample_assets_cache: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
) -> tuple[dandi_sample._DandiSampleResolvedAsset, ...]:
    dandi_sample._log_resolved_sample_assets(
        _dandi_001637_resolved_sample_assets_cache,
        cache_scope="session",
    )
    return _dandi_001637_resolved_sample_assets_cache
