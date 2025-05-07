"""
Integration tests for file_io and file_access modules together.
"""

import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to ensure the imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import pytest
from src.lazynwb.file_handlers import (
    FileAccessor,
    auto_file_open,
    get_file_path,
    is_same_file,
)


class TestIntegration:
    """Integration tests between file_io and file_access."""

    def setup_method(self):
        """Set up test data."""
        # Create temp file with test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name) / "test_file.h5"
        self.second_path = Path(self.temp_dir.name) / "test_file2.h5"
        self.output_paths = []  # Track any additional files created during tests

        # Create files with different data
        with h5py.File(self.temp_path, "w") as f:
            f.create_dataset("data", data=np.arange(10))

        with h5py.File(self.second_path, "w") as f:
            f.create_dataset("data", data=np.arange(5))

    def teardown_method(self):
        """Clean up test data."""
        # First try to remove any additional output files created during tests
        for path in self.output_paths:
            try:
                if Path(path).exists():
                    Path(path).unlink()
            except Exception as e:
                print(f"Warning: Failed to remove output file {path}: {e}")

        # Then clean up the temp directory
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")

    def test_auto_file_open_with_get_file_path(self):
        """Test combining auto_file_open with get_file_path."""

        @auto_file_open(h5py.File)
        def process_file(file):
            path = get_file_path(file)
            # Read the data
            data = file["data"][:]
            return {"path": path, "data": data}

        # Process a path
        result1 = process_file(file=str(self.temp_path))
        assert str(self.temp_path) in str(result1["path"])
        assert np.array_equal(result1["data"], np.arange(10))

        # Process with already open file
        with h5py.File(self.second_path, "r") as f:
            result2 = process_file(file=f)
            assert str(self.second_path) in str(result2["path"])
            assert np.array_equal(result2["data"], np.arange(5))

    def test_is_same_file_with_auto_file_open(self):
        """Test using is_same_file inside a function decorated with auto_file_open."""

        @auto_file_open(h5py.File)
        def check_files_match(file1, file2):
            # Compare the paths
            return is_same_file(file1, file2)

        # Check with same path in different formats
        assert check_files_match(file1=str(self.temp_path), file2=self.temp_path)

        # Check with different paths
        assert not check_files_match(
            file1=str(self.temp_path), file2=str(self.second_path)
        )

        # Check with one open file and one path
        with h5py.File(self.temp_path, "r") as f:
            assert check_files_match(file1=f, file2=str(self.temp_path))

    def test_file_accessor_with_auto_file_open(self):
        """Test using FileAccessor with auto_file_open."""

        @auto_file_open(h5py.File)
        def process_with_accessor(file):
            # Create a FileAccessor from the file
            accessor = FileAccessor(file)
            return accessor["data"][:]

        result = process_with_accessor(file=str(self.temp_path))
        assert np.array_equal(result, np.arange(10))

    def test_complex_workflow(self):
        """Test a more complex workflow combining multiple file operations."""

        # Create a function that reads from one file and writes to another
        @auto_file_open(h5py.File)
        def transform_and_save(input_files, output_file):
            # Extract paths for logging
            input_paths = [get_file_path(f) for f in input_files]
            output_path = get_file_path(output_file)

            # Process data
            all_data = []
            for i, f in enumerate(input_files):
                data = f["data"][:]
                all_data.append(data * (i + 1))  # Apply a transformation

            # Combine and save
            combined = np.concatenate(all_data)
            output_file.create_dataset("processed_data", data=combined)

            return {
                "input_paths": input_paths,
                "output_path": output_path,
                "data_shape": combined.shape,
            }

        # Create a third file for output
        output_path = Path(self.temp_dir.name) / "output.h5"
        self.output_paths.append(output_path)  # Track for cleanup

        # Run the workflow
        result = transform_and_save(
            input_files=[str(self.temp_path), str(self.second_path)],
            output_file=str(output_path),
        )

        # Verify the results
        assert len(result["input_paths"]) == 2
        assert str(self.temp_path) in str(result["input_paths"][0])
        assert str(self.second_path) in str(result["input_paths"][1])
        assert str(output_path) in str(result["output_path"])

        # Verify the data was written correctly and ensure file is closed
        with h5py.File(output_path, "r") as f:
            data = f["processed_data"][:]
            expected = np.concatenate([np.arange(10) * 1, np.arange(5) * 2])
            assert np.array_equal(data, expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
