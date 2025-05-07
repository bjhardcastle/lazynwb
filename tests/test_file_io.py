"""
Unit tests for the file_io module.
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

# Add the parent directory to sys.path to ensure the imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import pytest
import upath

# Update imports to use file_handlers instead of file_io
from src.lazynwb.file_handlers import (
    FileAccessor,
    auto_file_close,
    get_file_path,
    is_same_file,
)


class TestFilePathExtraction:
    """Tests for file path extraction utilities."""

    def test_get_file_path_from_string(self):
        """Test extracting path from string."""
        path_str = "/path/to/file.nwb"
        result = get_file_path(path_str)
        assert isinstance(result, str)
        assert result == path_str

    def test_get_file_path_from_path(self):
        """Test extracting path from Path object."""
        path_obj = Path("/path/to/file.nwb")
        result = get_file_path(path_obj)
        assert isinstance(result, str)
        assert result == str(path_obj)

    def test_get_file_path_from_upath(self):
        """Test extracting path from UPath object."""
        path_obj = upath.UPath("/path/to/file.nwb")
        result = get_file_path(path_obj)
        assert isinstance(result, str)
        assert result == path_obj.as_posix()

    def test_get_file_path_from_file_accessor(self):
        """Test extracting path from FileAccessor."""
        with mock.patch("src.lazynwb.file_handlers.open") as mock_open:
            mock_open.return_value = mock.MagicMock()
            file_accessor = FileAccessor("/path/to/file.nwb")
            result = get_file_path(file_accessor)
            assert isinstance(result, str)
            assert result == "/path/to/file.nwb"

    def test_get_file_path_from_h5py_file(self):
        """Test extracting path from h5py.File."""
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as h5file:
                result = get_file_path(h5file)
                assert isinstance(result, upath.UPath)
                assert Path(result.as_posix()).name == Path(tmp.name).name

    def test_get_file_path_invalid_input(self):
        """Test extracting path from invalid input."""
        with pytest.raises(ValueError):
            get_file_path(123)  # Not a valid file argument

    def test_is_same_file_identical_paths(self):
        """Test is_same_file with identical paths."""
        path1 = "/path/to/file.nwb"
        path2 = "/path/to/file.nwb"
        assert is_same_file(path1, path2) is True

    def test_is_same_file_different_paths(self):
        """Test is_same_file with different paths."""
        path1 = "/path/to/file1.nwb"
        path2 = "/path/to/file2.nwb"
        assert is_same_file(path1, path2) is False

    def test_is_same_file_cloud_paths(self):
        """Test is_same_file with cloud paths."""
        path1 = "s3://bucket/file.nwb"
        path2 = "s3://bucket/file.nwb"
        path3 = "s3://other-bucket/file.nwb"
        assert is_same_file(path1, path2) is True
        assert is_same_file(path1, path3) is False

    def test_is_same_file_different_slashes(self):
        """Test is_same_file with different slash styles."""
        path1 = "/path/to/file.nwb"
        path2 = "\\path\\to\\file.nwb"  # Windows style
        assert is_same_file(path1, path2) is True


class TestFileAccessor:
    """Tests for FileAccessor class."""

    def test_file_accessor_init_with_path(self):
        """Test initializing FileAccessor with a path."""
        with mock.patch("src.lazynwb.file_io.open") as mock_open:
            mock_open.return_value = mock.MagicMock()
            accessor = FileAccessor("/path/to/file.nwb")
            assert accessor._path is not None
            mock_open.assert_called_once()
            # Ensure proper cleanup
            if hasattr(accessor, "__exit__"):
                accessor.__exit__(None, None, None)

    def test_file_accessor_init_with_accessor(self):
        """Test initializing FileAccessor with an accessor."""
        mock_accessor = mock.MagicMock(spec=h5py.File)
        accessor = FileAccessor(mock_accessor)
        assert accessor._path is None
        assert accessor._accessor is mock_accessor
        # Ensure proper cleanup
        if hasattr(accessor, "__exit__"):
            accessor.__exit__(None, None, None)

    def test_file_accessor_context_manager(self):
        """Test FileAccessor as context manager."""
        mock_h5file = mock.MagicMock(spec=h5py.File)

        with FileAccessor(mock_h5file) as accessor:
            assert accessor._accessor is mock_h5file

        # The close method should be called when exiting the context
        mock_h5file.close.assert_called_once()

    def test_file_accessor_getitem(self):
        """Test FileAccessor __getitem__ method."""
        mock_h5file = mock.MagicMock(spec=h5py.File)
        mock_h5file.__getitem__.return_value = "dataset_value"

        accessor = FileAccessor(mock_h5file)
        result = accessor["dataset"]

        assert result == "dataset_value"
        mock_h5file.__getitem__.assert_called_once_with("dataset")

    def test_file_accessor_getattr(self):
        """Test FileAccessor __getattr__ method."""
        mock_h5file = mock.MagicMock(spec=h5py.File)
        mock_h5file.attr_name = "attr_value"

        accessor = FileAccessor(mock_h5file)
        assert accessor.attr_name == "attr_value"


class TestAutoFileClose:
    """Tests for auto_file_close decorator."""

    def setup_method(self):
        """Set up test data."""
        # Create temp file with test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name) / "test_file.h5"

        with h5py.File(self.temp_path, "w") as f:
            f.create_dataset("data", data=np.arange(10))

    def teardown_method(self):
        """Clean up test data."""
    def test_auto_file_close_with_path(self):
        """Test auto_file_close with a path argument."""

        @auto_file_close(h5py.File)
        def read_data(file):
            return file["data"][:]

        result = read_data(file=str(self.temp_path))
        assert np.array_equal(result, np.arange(10))

    def test_auto_file_close_with_open_file(self):
        """Test auto_file_close with an already open file."""

        @auto_file_close(h5py.File)
        def read_data(file):
            return file["data"][:]

        with h5py.File(self.temp_path, "r") as f:
            result = read_data(file=f)
            # Verify file is still open
            assert not f.id.valid, "File should remain open after function call"

        assert np.array_equal(result, np.arange(10))

    def test_auto_file_close_with_multiple_paths(self):
        """Test auto_file_close with multiple file path arguments."""
        # Create a second test file
        second_path = Path(self.temp_dir.name) / "test_file2.h5"
        with h5py.File(second_path, "w") as f:
            f.create_dataset("data", data=np.arange(5))

        @auto_file_close(h5py.File)
        def combine_data(file1, file2):
            return np.concatenate([file1["data"][:], file2["data"][:]])

        result = combine_data(file1=str(self.temp_path), file2=str(second_path))
        assert np.array_equal(result, np.concatenate([np.arange(10), np.arange(5)]))

    def test_auto_file_close_with_file_list(self):
        """Test auto_file_close with a list of file paths."""
        # Create multiple test files
        paths = []
        for i in range(3):
            path = Path(self.temp_dir.name) / f"test_list_{i}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.ones(5) * i)
            paths.append(str(path))

        @auto_file_close(h5py.File)
        def process_file_list(files):
            return [np.sum(f["data"][:]) for f in files]

        results = process_file_list(files=paths)
        assert results == [0, 5, 10]  # Sum of each dataset

    def test_auto_file_close_with_thread_pool(self):
        """Test auto_file_close with thread pool for parallel file opening."""
        # Create multiple test files
        paths = []
        for i in range(5):
            path = Path(self.temp_dir.name) / f"test_pool_{i}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.ones(5) * i)
            paths.append(str(path))

        @auto_file_close(h5py.File, use_thread_pool=True)
        def process_files_parallel(files):
            return [np.sum(f["data"][:]) for f in files]

        results = process_files_parallel(files=paths)
        assert results == [0, 5, 10, 15, 20]  # Sum of each dataset

        # Verify files are properly closed
        for path in paths:
            # Try to open each file to ensure it was properly closed
            with h5py.File(path, "r") as f:
                assert f["data"].shape == (5,), f"File {path} was not properly closed"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
