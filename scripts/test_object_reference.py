import time
import fsspec
import h5py
import psutil
import remfile


LARGE_HDF5_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c'
SMALL_HDF5_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/56c/31a/56c31a1f-a6fb-4b73-ab7d-98fb5ef9a553' 

url = SMALL_HDF5_URL    #for quicker testing use small file 

use_remfile = False
if use_remfile:
    nwb = h5py.File(remfile.File(url, verbose=True, _max_threads=10))
else:
    fsspec.get_filesystem_class("https").clear_instance_cache()
    filesystem = fsspec.filesystem("https")
    byte_stream = filesystem.open(path=url, mode="rb", cache_type="first")
    nwb = h5py.File(name=byte_stream)
    
# this is an instance of <HDF5 object reference>:
object_reference = nwb['units/electrode_group'][0]

# the location that `object_reference` points to (which currently can't be
# determined from the opaque object reference in python)
url_to_actual_location = {
    LARGE_HDF5_URL: '/general/extracellular_ephys/17216703352 1-281',
    SMALL_HDF5_URL: '/general/extracellular_ephys/18005110031 1-281',
}

def get_time_and_memory():
    m0 = psutil.Process().memory_info().rss
    t0 = time.time()
    yield
    t1 = time.time()
    m1 = psutil.Process().memory_info().rss
    yield f"{t1 - t0:.2f} s, {(m1 - m0) / 1024**2:.2f} MB"

# 1. accessing the location directly and reading metadata is fast:
tm = get_time_and_memory()
next(tm)
_ = nwb[url_to_actual_location[url]].name
print(f"1. Got referenced object data directly: {next(tm)}")

# 2. when using the object reference, a lazy accessor seems to be returned initially
# (which is fast):
tm = get_time_and_memory()
next(tm)
lazy_object_data = nwb[object_reference]
print(f"2. Got lazy object reference: {next(tm)}")

# 3'. de-reference the lazy object to get location and use directly:
tm = get_time_and_memory()
next(tm)
loc = h5py.h5r.get_name(object_reference, nwb.id)
print(f"3''. Got de-referenced location: {next(tm)}")

tm = get_time_and_memory()
next(tm)
reference_path = nwb[loc].name
print(f"3'. Got de-referenced object data: {next(tm)}")

# 3. when the same component is accessed, it is much slower than in 1. - suggests
#    more data than necessary is being read
tm = get_time_and_memory()
next(tm)
reference_path = lazy_object_data.name
print(f"3. Got referenced object data: {next(tm)}")
assert reference_path == url_to_actual_location[url]      

# 4. subsequent access of a different component is fast - supporting the idea that 
#    more data than necessary is being read (and cached) in 3. 
second_object_reference = nwb['units/electrode_group'][-1]
tm = get_time_and_memory()
next(tm)
second_reference_path = nwb[second_object_reference].name
print(f"4. Got second de-referenced object data: {next(tm)}")
assert second_reference_path != url_to_actual_location[url]
