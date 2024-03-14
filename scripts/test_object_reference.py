import time
import h5py
import remfile

LARGE_HDF5_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c'
SMALL_HDF5_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/56c/31a/56c31a1f-a6fb-4b73-ab7d-98fb5ef9a553' 

url = LARGE_HDF5_URL    #for quicker testing use small file 

nwb = h5py.File(remfile.File(url), mode="r")

# this is an instance of <HDF5 object reference>:
object_reference = nwb['units/electrode_group'][0]

# the location `object_reference` points to (which can't be determined from the object reference itself)
url_to_actual_location = {
    LARGE_HDF5_URL: '/general/extracellular_ephys/17216703352 1-281',
    SMALL_HDF5_URL: '/general/extracellular_ephys/18005110031 1-281',
}

# 1. accessing the location directly and reading metadata is fast:
t0 = time.time()
_ = nwb[url_to_actual_location[url]].name
print(f"1. Time to get referenced object data directly: {time.time() - t0:.2f} s")

# 2. when using the object reference, a lazy accessor seems to be returned initially
# (which is fast):
t0 = time.time()
lazy_object_data = nwb[object_reference]
print(f"2. Time to get lazy object reference: {time.time() - t0:.2f} s")

# 3. when the same component is accessed, it is much slower than in 1. - suggests
#    more data than necessary is being read
t0 = time.time()
reference_path = lazy_object_data.name
print(f"3. Time to get referenced object data: {time.time() - t0:.2f} s")
assert reference_path == url_to_actual_location[url]      

# 4. subsequent access of a different component is fast - supporting the idea that 
#    more data than necessary is being read (and cached) in 3. 
t0 = time.time()
second_object_reference = nwb['units/electrode_group'][-1]
second_reference_path = nwb[second_object_reference].name
print(f"4. Time to get second referenced object data: {time.time() - t0:.2f} s")
assert second_reference_path != url_to_actual_location[url]
