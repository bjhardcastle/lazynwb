import concurrent.futures
import dataclasses
import datetime
from typing import Callable

import dandi.dandiapi
import numpy as np
import tqdm
import pandas as pd

import lazynwb

@dataclasses.dataclass
class Result:
    nwb_path: str
    session_start_time: datetime.datetime
    subject_id: str
    num_good_units: int
    device: str


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
    devices = [nwb[group]['device'].name.split('/')[-1] for group in nwb.units.electrode_group]
    results = []
    for device in set(devices):
        selected_units = np.argwhere(devices == device).flatten()
        num_good_units = len(np.argwhere(classification.asstr()[selected_units] == 'good').flatten())
        results.append(Result(nwb_path=nwb_path, session_start_time=session_start_time, subject_id=subject_id, num_good_units=num_good_units, device=device))
    return results

def save_results(dandiset_id: str, helper: Callable, use_threadpool: bool = True) -> None:
    results = []
    assets = lazynwb.get_dandiset_assets(dandiset_id)
    if use_threadpool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [pool.submit(helper, asset) for asset in assets]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    results.extend(result)
    else:
        for asset in tqdm.tqdm(assets):
            results.extend(helper(asset))
    pd.DataFrame.from_records(dataclasses.asdict(result) for result in results).to_csv('results.csv', index=False)
    
def main() -> None:
    save_results(dandiset_id='000363', helper=chen_helper, use_threadpool=True)
    
if __name__ == "__main__":
    main()
