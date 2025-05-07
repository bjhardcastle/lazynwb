"""
Unit tests for the file_access module.
"""

import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to ensure the imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import pytest

# Update imports to use file_handlers instead of file_access
from src.lazynwb.file_handlers import FileAccessWrapper, auto_file_open


class TestFileAccessWrapper:
    """Tests for FileAccessWrapper class."""

    def setup_method(self):
        """Set up test data."""
        # Create temp file with test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name) / "test_file.h5"

        with h5py.File(self.temp_path, "w") as f:
            f.create_dataset("data", data=np.arange(10))

    def teardown_method(self):
        """Clean up test data."""
        # Use try-except to ensure cleanup even if an error occurs
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")

    def test_file_access_wrapper_with_path(self):
        """Test FileAccessWrapper with a path."""
        wrapper = FileAccessWrapper(str(self.temp_path), h5py.File)

        with wrapper.access() as f:
            assert isinstance(f, h5py.File)
            assert np.array_equal(f["data"][:], np.arange(10))

        # File should be closed after context exit
        assert not f.id.valid, "File should be closed after context exit"

    def test_file_access_wrapper_with_open_file(self):
        """Test FileAccessWrapper with an already open file."""
        with h5py.File(self.temp_path, "r") as f:
            wrapper = FileAccessWrapper(f)

            with wrapper.access() as wrapped_f:
                assert wrapped_f is f  # Should be the same object
                assert np.array_equal(wrapped_f["data"][:], np.arange(10))

            # File should still be open as we didn't open it
            assert f.id.valid, "File should still be open"


class TestAutoFileOpen:
    """Tests for auto_file_open decorator."""

    def setup_method(self):
        """Set up test data."""
        # Create temp file with test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name) / "test_file.h5"

        with h5py.File(self.temp_path, "w") as f:
            f.create_dataset("data", data=np.arange(10))

    def teardown_method(self):
        """Clean up test data."""
        # Use try-except to ensure cleanup even if an error occurs
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")

    def test_auto_file_open_with_path(self):
        """Test auto_file_open with a path argument."""

        @auto_file_open(h5py.File)
        def read_data(file):
            return file["data"][:]

        result = read_data(file=str(self.temp_path))
        assert np.array_equal(result, np.arange(10))

    def test_auto_file_open_with_open_file(self):
        """Test auto_file_open with an already open file."""

        @auto_file_open(h5py.File)
        def read_data(file):
            return file["data"][:]

        with h5py.File(self.temp_path, "r") as f:
            result = read_data(file=f)
            assert np.array_equal(result, np.arange(10))
            # File should still be open
            assert f.id.valid, "File should remain open after function call"

    def test_auto_file_open_with_multiple_paths(self):
        """Test auto_file_open with multiple file path arguments."""
        # Create a second test file
        second_path = Path(self.temp_dir.name) / "test_file2.h5"
        with h5py.File(second_path, "w") as f:
            f.create_dataset("data", data=np.arange(5))

        @auto_file_open(h5py.File)
        def combine_data(file1, file2):
            data1 = file1["data"][:]
            data2 = file2["data"][:]
            return np.concatenate([data1, data2])

        result = combine_data(file1=str(self.temp_path), file2=str(second_path))
        assert np.array_equal(result, np.concatenate([np.arange(10), np.arange(5)]))

    def test_auto_file_open_with_file_list(self):
        """Test auto_file_open with a list of file paths."""
        # Create multiple test files
        paths = []
        try:
            for i in range(3):
                path = Path(self.temp_dir.name) / f"test_list_{i}.h5"
                with h5py.File(path, "w") as f:
                    f.create_dataset("data", data=np.ones(5) * i)
                paths.append(str(path))

            @auto_file_open(h5py.File)
            def process_file_list(files):
                return [np.sum(f["data"][:]) for f in files]

            results = process_file_list(files=paths)
            assert results == [0, 5, 10]  # Sum of each dataset

            # Verify files are properly closed by trying to open them again
            for path in paths:
                with h5py.File(path, "r") as f:
                    assert f["data"].shape == (
                        5,
                    ), f"File {path} was not properly closed"
        except Exception:
            # Re-raise the exception after cleanup attempts
            raise

    def test_auto_file_open_with_thread_pool(self):
        """Test auto_file_open with thread pool for parallel file opening."""
        # Create multiple test files
        paths = []
        for i in range(5):
            path = Path(self.temp_dir.name) / f"test_pool_{i}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.ones(5) * i)
            paths.append(str(path))

        @auto_file_open(h5py.File, use_thread_pool=True)
        def process_files_parallel(files):
            return [np.sum(f["data"][:]) for f in files]

        results = process_files_parallel(files=paths)
        assert results == [0, 5, 10, 15, 20]  # Sum of each dataset

    def test_default_opener(self):
        """Test auto_file_open with the default opener."""
        # This test will attempt to use the default opener which might be src.lazynwb.file_handlers.open
        # Mock the default_opener to avoid actual file operations
        import unittest.mock as mock

        # Fix the import path for the mock patch to target the correct module
        with mock.patch("src.lazynwb.file_handlers.open") as mock_opener:
            mock_file = mock.MagicMock()
            mock_file.__getitem__.return_value = {"data": np.arange(5)}
            mock_opener.return_value = mock_file

            @auto_file_open()
            def read_with_default_opener(file):
                return file["data"]

            result = read_with_default_opener(file="dummy_path")
            assert mock_opener.called
            assert result == mock_file["data"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
