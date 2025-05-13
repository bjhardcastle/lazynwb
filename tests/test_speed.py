import logging
import time

import pytest

import lazynwb

logger = logging.getLogger()

MIN_OPEN_TIME_SECONDS = 2.5

def get_large_hdf5_url() -> str:
    dandiset_id = '000363'  # ephys dataset from the Svoboda Lab
    filepath = 'sub-440957/sub-440957_ses-20190211T143614_behavior+ecephys+image+ogen.nwb' # 437 GB file
    with lazynwb.get_dandi_client() as client:
        asset = client.get_dandiset(
            dandiset_id=dandiset_id, version_id="draft"
        ).get_asset_by_path(filepath)
        return asset.get_content_url(follow_redirects=1, strip_query=True)

def get_small_zarr_url() -> str:
    return 's3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb'

@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.fixture
def url(request: pytest.FixtureRequest) -> str:
    if request.param == 'large_hdf5':
        return get_large_hdf5_url()
    elif request.param == 'small_zarr':
        return get_small_zarr_url()
    else:
        raise ValueError(f'Unknown url fixture value: {request.param}')

@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.mark.parametrize('url', ['large_hdf5', 'small_zarr'], indirect=True)
def test_open_time(url: str) -> None:
    # may need to try this more than once: S3 storage can be slow on first request in a while
    t0 = time.time()
    nwb = lazynwb.FileAccessor(url)
    t = time.time() - t0
    logger.info(f'Opened {url} with {nwb.__class__.__name__} in {t:.2f} seconds')
    assert t < MIN_OPEN_TIME_SECONDS, f'Opening {url} with {nwb.__class__.__name__} took too long: {t:.1f} seconds (expected < {MIN_OPEN_TIME_SECONDS})'

@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.mark.parametrize('url', ['large_hdf5', 'small_zarr'], indirect=True)
def test_metadata_df(url: str) -> None:
    t0 = time.time()
    df = lazynwb.get_metadata_df(url, disable_progress=True)
    t    = time.time() - t0
    assert t < MIN_OPEN_TIME_SECONDS, f'Fetching summary dataframe took too long: {t:.1f} seconds (expected < {MIN_OPEN_TIME_SECONDS})'
    logger.info(f'Fetched summary dataframe for {url} in {t:.2f} seconds')
    
@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.mark.parametrize('url', ['large_hdf5'], indirect=True)
def test_remfile_vs_h5py(url: str) -> None:
    times = []
    for use_remfile in [True, False]: # the first S3 access of data is typically slower than subseqeuent ones, so this is biased against remfile
        t0 = time.time()
        _ = lazynwb.open(url, use_remfile=use_remfile)
        times.append( t:= time.time() - t0)
        logger.info(f'Opened {url} with {use_remfile=} in {t:.2f} seconds')
    assert times[0] < times[1], f'Opening {url} with remfile {times[0]=} was not faster than h5py {times[1]=}: default to remfile=False in open()'

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    pytest.main([__file__])
