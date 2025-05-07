"""
Examples demonstrating how to use the file I/O utilities in lazynwb.
"""

import time

import h5py
import numpy as np

# Update import to use file_handlers instead of file_io
from src.lazynwb.file_handlers import FileAccessor, Safe, auto_file_close


# Example 1: Simple file access using default opener
@auto_file_close()
def process_nwb(file):
    """
    Process data from either a file path or an already open file.
    Uses the default opener which handles both HDF5 and Zarr formats.

    Args:
        file: Path to an NWB file or an open file object
    """
    data = file["processing"]["behavior"]["timestamps"][:]
    return data


# Example 2: Working with multiple files using h5py explicitly
@auto_file_close(h5py.File)  # Using the top-level alias
def copy_dataset(src_file, dest_file, src_path, dest_path=None):
    """Copy a dataset from one file to another."""
    if dest_path is None:
        dest_path = src_path
    dest_file[dest_path] = src_file[src_path][:]
    return True


# Example 3: Working with a list of files
@Safe.auto_file_open()
def compute_average(nwb_files, dataset_path):
    """Compute average across multiple files."""
    datasets = []
    for file in nwb_files:
        datasets.append(file[dataset_path][:])
    return np.mean(np.concatenate(datasets, axis=0))


# Example 4: Using thread pool to open files in parallel
@Safe.auto_file_open(use_thread_pool=True)
def parallel_process_files(files, output_file, dataset_path):
    """
    Process multiple files in parallel and save results to output file.
    Files will be opened in parallel using a thread pool.
    """
    datasets = []
    for file in files:
        datasets.append(file[dataset_path][:])

    # Since all files are wrapped in FileAccessor, we can access the underlying object
    output_file.create_dataset(dataset_path, data=np.concatenate(datasets, axis=0))
    return datasets


def demonstrate_usage():
    """Run through examples of various file I/O usage patterns."""

    # Example with explicit file opening using FileAccessor
    print("Example: Using FileAccessor directly")
    with FileAccessor("path/to/data.nwb") as file:
        units = file["units"]
        print(f"Number of units: {len(units)}")

    # Example using the decorator with a path
    print("\nExample: Using Safe.auto_file_open decorator with path")
    result = process_nwb(file="path/to/data.nwb")
    print(f"Got data with shape: {result.shape}")

    # Example with already open file
    print("\nExample: Using Safe.auto_file_open with already open file")
    with FileAccessor("path/to/data.nwb") as file:
        # Will use the already open file, not try to re-open it
        result = process_nwb(file=file)
        print(f"Got data with shape: {result.shape}")

    # Example with multiple files
    print("\nExample: Working with multiple files")
    copy_result = copy_dataset(
        src_file="path/to/data1.nwb",
        dest_file="path/to/output.nwb",
        src_path="processing/behavior/timestamps",
    )
    print(f"Copy successful: {copy_result}")

    # Example with a list of files
    print("\nExample: Working with a list of files")
    file_paths = ["path/to/data1.nwb", "path/to/data2.nwb", "path/to/data3.nwb"]
    avg_value = compute_average(
        nwb_files=file_paths, dataset_path="processing/behavior/timestamps"
    )
    print(f"Average value: {avg_value}")

    # Example with parallel file opening
    print("\nExample: Parallel file opening")
    many_files = [f"path/to/data{i}.nwb" for i in range(10)]
    start_time = time.time()
    results = parallel_process_files(
        files=many_files,
        output_file="path/to/output.nwb",
        dataset_path="processing/behavior/timestamps",
    )
    print(f"Processed 10 files in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    demonstrate_usage()
