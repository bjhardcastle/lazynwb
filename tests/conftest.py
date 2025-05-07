"""
Pytest configuration file for test suite.
"""

import glob
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to ensure the imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files():
    """Clean up any temporary files that might be left by tests."""
    # Set up - nothing to do here
    yield  # This is where the tests run

    # Tear down - clean up any leftover temporary files
    # Look for temp files in the system temp directory
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, "tmp*")
    for temp_file in glob.glob(pattern):
        if os.path.isfile(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                # Skip files we can't delete
                pass
        elif os.path.isdir(temp_file):
            try:
                # This is not recursive, only removes empty dirs
                os.rmdir(temp_file)
            except OSError:
                # Skip directories we can't delete
                pass
