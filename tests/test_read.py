import time
import dandi.dandiapi

import cloudnwb 

MIN_OPEN_TIME_SECONDS = 1

def test_open_large_hdf5_time() -> None:
    dandiset_id = '000363'  # ephys dataset from the Svoboda Lab
    filepath = 'sub-440957/sub-440957_ses-20190211T143614_behavior+ecephys+image+ogen.nwb' # 437 GB file
    with dandi.dandiapi.DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    t0 = time.time()
    cloudnwb.lazy_open_nwb(s3_url)
    if (t := time.time() - t0) > MIN_OPEN_TIME_SECONDS:
        raise AssertionError(f'Opening {s3_url} took too long: {t} seconds')
    
test_open_large_hdf5_time()