from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime
import os
from collections.abc import Iterable
from typing import Callable

import dandi.dandiapi
import numpy as np
import pandas as pd
import tqdm

import lazynwb


@dataclasses.dataclass
class Result:
    nwb_path: str
    session_start_time: datetime.datetime
    subject_id: str
    num_good_units: int
    num_units: int
    brain_region: str
    device: str
    is_single_shank: bool 
    is_v1_probe: bool
    most_common_area: str


def chen_helper(asset: dandi.dandiapi.BaseRemoteAsset) -> list[Result]:
    """Return one Result for each device in the nwb file (may be empty if no good units)."""
    nwb_path = asset.path
    nwb = lazynwb.get_lazynwb_from_dandiset_asset(asset)
    session_start_time = datetime.datetime.fromisoformat(nwb.session_start_time[()].decode())
    subject_id = nwb.general.subject.subject_id.asstr()[()]
    classification = nwb.units.classification
    if classification.dtype == float:
        num_good_units = len(np.argwhere(~np.isnan(classification[:])).flatten())
        if num_good_units == 0:
            assert nwb.units.anno_name.dtype == float
            return [] # some sessions (e.g. 2019-02-16) are in dataset but apparently have no good units or area labels
    if classification.dtype.name != 'object':
        raise ValueError(f"Unrecognized classification dtype: {classification.dtype}")
    devices = np.array([nwb[group]['device'].name.split('/')[-1] for group in nwb.units.electrode_group])
    results = []
    for device in np.unique(devices):
        device_units = devices == device
        good_units = (classification.asstr()[:] == 'good') & device_units
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

def append_to_csv(csv_name: str, results: Result | Iterable[Result]) -> None:
    if not isinstance(results, Iterable):
        results = [results]
    try:
        df = pd.read_csv(csv_name)
    except FileNotFoundError:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame.from_records(dataclasses.asdict(result) for result in results)])
    df.to_csv(csv_name, index=False)

def save_results(dandiset_id: str, csv_name: str, helper: Callable, use_threadpool: bool = False) -> None:
    """Threadpool currently encounters a deadlock - probably due to hdf5 file"""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(csv_name)
    assets = lazynwb.get_dandiset_assets(dandiset_id)
    if use_threadpool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(helper, asset) for asset in assets]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    append_to_csv(csv_name, result)
    else:
        for asset in tqdm.tqdm(assets):
            append_to_csv(csv_name, helper(asset))

def main() -> None:
    save_results(dandiset_id='000363', csv_name='chen_results.csv', helper=chen_helper, use_threadpool=False)

if __name__ == "__main__":
    main()
