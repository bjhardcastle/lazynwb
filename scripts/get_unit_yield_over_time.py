from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime
import logging
import os
from collections.abc import Iterable
import threading
import time
from typing import Callable

import dandi.dandiapi
import numpy as np
import pandas as pd
import tqdm

import lazynwb

logger = logging.getLogger(__name__)

CHEN_BRAINWIDE_DANDISET_ID = '000363'
IBL_BRAINWIDE_DANDISET_ID = '000409'

results_file_lock = threading.Lock()

@dataclasses.dataclass
class Result:
    nwb_path: str
    session_start_time: datetime.datetime
    subject_id: str
    num_good_units: int
    num_units: int
    device: str
    is_single_shank: bool
    is_v1_probe: bool
    most_common_area: str
    brain_region: str | None = None

def append_to_csv(csv_name: str, results: Result | Iterable[Result]) -> None:
    if not isinstance(results, Iterable):
        results = [results]
    if not results:
        return
    try:
        df = pd.read_csv(csv_name)
    except (FileNotFoundError, ValueError):
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame.from_records(dataclasses.asdict(result) for result in results)])
    # use lock to accommodate multi-threading
    with results_file_lock:
        df.to_csv(csv_name, index=False)

def get_assets(dandiset_id: str) -> tuple[dandi.dandiapi.BaseRemoteAsset, ...]:
    if dandiset_id == CHEN_BRAINWIDE_DANDISET_ID:
        return lazynwb.get_dandiset_assets(CHEN_BRAINWIDE_DANDISET_ID)
    elif dandiset_id == IBL_BRAINWIDE_DANDISET_ID:
        return get_ibl_brainwide_multiday_ephys_assets()
    else:
        raise ValueError(f"Unrecognized dandiset_id: {dandiset_id}")

def chen_helper(asset: dandi.dandiapi.BaseRemoteAsset) -> list[Result]:
    """Return one Result for each device in the nwb file (may be empty if no good units)."""
    nwb_path = asset.path
    with lazynwb.get_lazynwb_from_dandiset_asset(asset) as nwb:
        session_start_time = datetime.datetime.fromisoformat(nwb.session_start_time[()].decode())
        if 'units' not in nwb:
            return []
        subject_id = nwb.general.subject.subject_id.asstr()[()]
        classification_dtype = nwb.units.classification.dtype
        if classification_dtype == float:
            assert nwb.units.anno_name.dtype == float
            return [] # some sessions (e.g. 2019-02-16) are in dataset but apparently have no good units or area labels
        if classification_dtype.name != 'object':
            raise ValueError(f"Unrecognized classification dtype: {classification_dtype}")
        classification = nwb.units.classification.asstr()[:]
        # ----------------------------------------------------------------- #
        #! this part is slow!
        # - reading electrode_group for each unit (100s/1000s units) via hdf5 object references
        # - there are only a few electrode_groups (one per shank): ideally we
        #   would access each one only once
        # - but every reference object is unique even
        #   if they point to the same location in the file and there's no way to
        #   know what the object reference points to, other than to read it, which
        #   is slow 
        # - if we could convert <HDF5 object reference> to a string path, like in
        #   zarr files, this would be much faster
        t0 = time.time()
        devices = np.array([nwb._nwb[group]['device'].name.split('/')[-1] for group in nwb.units.electrode_group[:]])
        t = time.time() - t0
        logger.info(f"Time to get devices: {t:.2f} seconds")
        # ----------------------------------------------------------------- #
        results = []
        for device in np.unique(devices):
            device_units = devices == device
            good_units = (classification == 'good') & device_units
            unit_electrode_idx = nwb.units.electrodes[good_units]
            locations = []
            for electrode_idx in unit_electrode_idx:
                location: dict[str, str] = eval(nwb.general.extracellular_ephys.electrodes.location.asstr()[electrode_idx])
                locations.append(location)
            brain_region = locations[0]['brain_regions']
            assert all(loc['brain_regions'] == brain_region for loc in locations)
            num_good_units = len(np.argwhere(good_units).flatten())
            assert num_good_units == len(locations)
            most_common_area = next(name for name, count in collections.Counter(nwb.units.anno_name.asstr()[good_units]).most_common() if name)
            results.append(
                Result(
                    nwb_path=nwb_path,
                    session_start_time=session_start_time,
                    subject_id=subject_id,
                    num_units=len(np.argwhere(device_units).flatten()),
                    num_good_units=num_good_units,
                    device=device,
                    is_single_shank='MS' not in device, # MS = multi-shank
                    is_v1_probe='1.0' in device,
                    brain_region=brain_region,
                    most_common_area=most_common_area,
                    )
                )
        return results

def get_ibl_brainwide_multiday_ephys_assets() -> tuple[dandi.dandiapi.RemoteBlobAsset, ...]:
    assets = lazynwb.get_dandiset_assets(IBL_BRAINWIDE_DANDISET_ID)
    assets = tuple(asset for asset in assets if all(label in asset.path for label in ('ecephys', '.nwb')))
    assert not any(path.endswith('.mp4') for path in (asset.path for asset in assets))
    subjects = {asset.path.split('/')[0] for asset in assets}
    assets_for_subjects_with_multiple_days = []
    for subject in subjects:
        sessions = [asset for asset in assets if subject in asset.path]
        if len(sessions) > 1:
            assets_for_subjects_with_multiple_days.extend(sessions)
    return tuple(assets_for_subjects_with_multiple_days)

def ibl_helper(asset: dandi.dandiapi.BaseRemoteAsset) -> list[Result]:
    """Return one Result for each device in the nwb file (may be empty if no good units)."""
    nwb_path = asset.path
    with lazynwb.get_lazynwb_from_dandiset_asset(asset) as nwb:
        if 'units' not in nwb:
            return []
        session_start_time = datetime.datetime.fromisoformat(nwb.session_start_time[()].decode())
        subject_id = nwb.general.subject.subject_id.asstr()[()]
        label = nwb.units.label
        if label.dtype != float:
            raise ValueError(f"Unrecognized label dtype: {label.dtype}")
        devices = nwb.units.probe_name.asstr()[:]
        results = []
        for device in np.unique(devices):
            device_units = devices == device
            good_units = (label[:] == 1) & device_units
            locations = nwb.units.allen_location.asstr()[good_units]
            num_good_units = len(np.argwhere(good_units).flatten())
            assert num_good_units == len(locations)
            most_common_area = next(name for name, count in collections.Counter(locations).most_common() if name)
            results.append(
                Result(
                    nwb_path=nwb_path,
                    session_start_time=session_start_time,
                    subject_id=subject_id,
                    num_units=len(np.argwhere(device_units).flatten()),
                    num_good_units=num_good_units,
                    device=device,
                    is_single_shank=True,
                    is_v1_probe=True,
                    brain_region=None,
                    most_common_area=most_common_area,
                    )
                )
        return results

def save_results(dandiset_id: str, csv_name: str, helper: Callable, use_threadpool: bool = False) -> None:
    """Threadpool currently encounters a deadlock - probably due to hdf5 file"""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(csv_name)
    assets = get_assets(dandiset_id)
    tqdm_kwargs = dict(total=len(assets), desc=f"Processing dandiset {dandiset_id}", unit="nwb file", ncols=80)
    if use_threadpool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(helper, asset) for asset in assets]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), **tqdm_kwargs):
                results = future.result()
                append_to_csv(csv_name, results)
    else:
        for asset in tqdm.tqdm(assets, **tqdm_kwargs):
            results = helper(asset)
            append_to_csv(csv_name, results)
            
def main() -> None:
    t0 = time.time()
    save_results(dandiset_id=IBL_BRAINWIDE_DANDISET_ID, csv_name='ibl_results.csv', helper=ibl_helper, use_threadpool=True)
    logging.info(f"Total time: {time.time() - t0:.2f} seconds")
    
    t0 = time.time()
    save_results(dandiset_id=CHEN_BRAINWIDE_DANDISET_ID, csv_name='chen_results.csv', helper=chen_helper, use_threadpool=True)
    logging.info(f"Total time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
