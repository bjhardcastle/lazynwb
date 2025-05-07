"""
Examples demonstrating how to work with file paths extracted from various objects.
"""

import os
from pathlib import Path

import upath
from src.lazynwb.file_handlers import FileAccessor, get_file_path, is_same_file


def demonstrate_path_extraction():
    """Show how to extract paths from different types of objects."""

    # Example file paths - both local and cloud
    local_path = "path/to/data.nwb"
    cloud_path = "s3://my-bucket/path/to/data.nwb"

    # Extract from string path
    path1 = get_file_path(local_path)
    print(f"UPath from local string: {path1}")

    path2 = get_file_path(cloud_path)
    print(f"UPath from cloud string: {path2}")
    print(f"Cloud protocol: {path2.protocol}")

    # Extract from Path object
    path_obj = Path(local_path)
    path3 = get_file_path(path_obj)
    print(f"UPath from Path object: {path3}")

    # Extract from UPath object
    upath_obj = upath.UPath(cloud_path)
    path4 = get_file_path(upath_obj)
    print(f"UPath from UPath object: {path4}")

    # Extract from FileAccessor
    file_accessor = FileAccessor(cloud_path)
    path5 = get_file_path(file_accessor)
    print(f"UPath from FileAccessor: {path5}")


def use_path_in_workflow():
    """Example of using file path extraction in a workflow."""

    def process_file_with_checks(file_arg):
        """Process a file with path-based checks."""

        # Get the file path without opening the file
        file_path = get_file_path(file_arg)

        # Check if it's a cloud path
        if file_path.protocol:
            print(f"Cloud file detected on {file_path.protocol}: {file_path}")
            # For cloud paths, we might not want to do certain checks
            return

        # For local paths, we can do existence and size checks
        if not file_path.exists():
            print(f"Warning: File {file_path} doesn't exist")
            return

        # Check file size (without opening it)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # If file is larger than 100MB
            print(f"Large file detected ({size_mb:.1f}MB), processing may take time")

        # Check file modification time
        mod_time = os.path.getmtime(str(file_path))
        print(f"File was last modified at: {mod_time}")

        # Now process with auto_file_close decorator or directly
        print(f"Processing file: {file_path}")

    # Use it with different types
    process_file_with_checks("path/to/data.nwb")
    process_file_with_checks("s3://my-bucket/data.nwb")

    with FileAccessor("path/to/another_data.nwb") as file:
        process_file_with_checks(file)


def check_if_same_file():
    """Example showing how to check if two file arguments refer to same file."""

    # Different representations of the same file
    file_path1 = "path/to/data.nwb"
    file_path2 = Path("path/to/data.nwb")
    file_accessor = FileAccessor("path/to/data.nwb")

    # Check if they're the same file
    same_file1 = is_same_file(file_path1, file_path2)
    print(f"String path and Path object refer to same file: {same_file1}")

    same_file2 = is_same_file(file_path1, file_accessor)
    print(f"String path and FileAccessor refer to same file: {same_file2}")

    # Different files
    different_file = is_same_file(file_path1, "path/to/other_data.nwb")
    print(f"Different paths refer to same file: {different_file}")

    # Cloud paths
    cloud_path1 = "s3://my-bucket/data.nwb"
    cloud_path2 = "s3://my-bucket/data.nwb"
    cloud_path3 = "s3://other-bucket/data.nwb"

    print(f"Same S3 paths: {is_same_file(cloud_path1, cloud_path2)}")
    print(f"Different S3 paths: {is_same_file(cloud_path1, cloud_path3)}")
    print(f"Local vs S3: {is_same_file(file_path1, cloud_path1)}")


if __name__ == "__main__":
    print("=== Path Extraction Demo ===")
    demonstrate_path_extraction()

    print("\n=== Using Paths in Workflows ===")
    use_path_in_workflow()

    print("\n=== File Comparison Demo ===")
    check_if_same_file()
