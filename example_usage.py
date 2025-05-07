import time

import h5py
import numpy as np

# Update import paths
from src.lazynwb.file_handlers import FileAccessor, auto_file_open


# Example 1: Simple file access using default opener (from file_io.py)
@auto_file_open()
def process_nwb(file):
    """
    Process data from either a file path or an already open file.
    Uses the default opener from file_io.py which handles both HDF5 and Zarr formats.

    Args:
        file: Path to an NWB file or an open file object
    """
    data = file["processing"]["behavior"]["timestamps"][:]
    return data


# Example 2: Working with multiple files using h5py explicitly
@auto_file_open(h5py.File)
def copy_dataset(src_file, dest_file, src_path, dest_path=None):
    """Copy a dataset from one file to another."""
    if dest_path is None:
        dest_path = src_path
    dest_file[dest_path] = src_file[src_path][:]
    return True


# Example 3: Working with a list of files using default opener
@auto_file_open()
def compute_average(nwb_files, dataset_path):
    """Compute average across multiple files."""
    datasets = []
    for file in nwb_files:
        datasets.append(file[dataset_path][:])
    return np.mean(np.concatenate(datasets, axis=0))


# Example 4: Using thread pool to open files in parallel with default opener
@auto_file_open(use_thread_pool=True)
def parallel_process_files(files, output_file, dataset_path):
    """
    Process multiple files in parallel and save results to output file.
    Files will be opened in parallel using a thread pool.
    """
    datasets = []
    for file in files:
        datasets.append(file[dataset_path][:])

    if isinstance(output_file, FileAccessor):
        # Handle FileAccessor object differently than h5py.File
        # This demonstrates compatibility with different file objects
        output_file._accessor.create_dataset(
            dataset_path, data=np.concatenate(datasets, axis=0)
        )
    else:
        # Regular h5py.File handling
        output_file.create_dataset(dataset_path, data=np.concatenate(datasets, axis=0))

    return datasets


# Example with default opener
result_nwb = process_nwb(file="path/to/data.nwb")

# Example with parallel file opening using default opener
nwb_files = [f"path/to/data{i}.nwb" for i in range(10)]
start_time = time.time()
results = parallel_process_files(
    files=nwb_files,
    output_file="path/to/output.nwb",
    dataset_path="processing/behavior/timestamps",
)
print(f"Processed 10 files in {time.time() - start_time:.2f} seconds")
