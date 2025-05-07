"""
Examples showing how to use type hints with the file_io module's auto_file_close decorator.
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

# Update import to use file_handlers instead of file_io
from src.lazynwb.file_handlers import (
    FileAccessor,
    FileArgument,
    HasGetItem,
    auto_file_close,
)


# Example 1: Basic type hints
@auto_file_close()
def read_dataset(file: FileArgument, dataset_path: str) -> np.ndarray:
    """
    Read a dataset from a file.

    Args:
        file: Either a file path or an open file object
              (will be a DataAccessor inside the function)
        dataset_path: Path to the dataset within the file

    Returns:
        The dataset as a numpy array
    """
    # At this point, file is guaranteed to be an open DataAccessor
    # You can add an assertion for extra clarity and runtime checking
    assert isinstance(file, HasGetItem), "File must support __getitem__"

    return file[dataset_path][:]


# Example 2: Working with multiple files
@auto_file_close()
def merge_datasets(
    files: Sequence[FileArgument],
    dataset_path: str,
    output_file: Optional[FileArgument] = None,
) -> np.ndarray:
    """
    Merge datasets from multiple files.

    Args:
        files: Sequence of file paths or open file objects
        dataset_path: Path to the dataset within each file
        output_file: Optional output file to save the merged dataset

    Returns:
        The merged dataset as a numpy array
    """
    # Within the function, all files will be open DataAccessors
    datasets = []

    for file in files:
        # Runtime check for clarity
        assert isinstance(file, HasGetItem), "Each file must support __getitem__"
        datasets.append(file[dataset_path][:])

    merged = np.concatenate(datasets, axis=0)

    if output_file is not None:
        assert isinstance(
            output_file, HasGetItem
        ), "Output file must support __getitem__"
        output_file.create_dataset(dataset_path, data=merged)

    return merged


# Example 3: File argument with explicit type checking using isinstance
@auto_file_close()
def extract_data(file: FileArgument, group_path: str, dataset: str) -> np.ndarray:
    """
    Extract data from a specific group and dataset.

    Args:
        file: Either a file path or an open file object
        group_path: Path to the group within the file
        dataset: Name of the dataset within the group

    Returns:
        The dataset as a numpy array
    """
    # Different ways to do runtime type checks
    if isinstance(file, FileAccessor):
        # FileAccessor specific handling
        return file[f"{group_path}/{dataset}"][:]
    elif hasattr(file, "__getitem__"):
        # Generic handling for any object with __getitem__
        return file[group_path][dataset][:]
    else:
        raise TypeError(f"Expected DataAccessor, got {type(file)}")


def main():
    # These examples show the type annotations
    # In actual use, you'd provide real file paths

    # Example 1: Basic usage
    data = read_dataset(file="path/to/file.nwb", dataset_path="timestamps")
    print(f"Read data with shape {data.shape}")

    # Example 2: Multiple files
    files = ["path/to/file1.nwb", "path/to/file2.nwb"]
    merged = merge_datasets(files=files, dataset_path="timestamps")
    print(f"Merged data with shape {merged.shape}")

    # Example 3: Explicit type checking
    data = extract_data(
        file="path/to/file.nwb", group_path="processing/behavior", dataset="timestamps"
    )
    print(f"Extracted data with shape {data.shape}")


if __name__ == "__main__":
    main()
