"""
lazynwb: A library for lazy loading of NWB (Neurodata Without Borders) files.
"""

import doctest
import importlib.metadata
import logging

# Import other modules
from lazynwb.base import *
from lazynwb.lazyframe import *
from lazynwb.tables import *
from lazynwb.timeseries import *
from lazynwb.utils import *

from .file_handlers import (
    FileAccessor,
    FileAccessWrapper,
    auto_file_close,
    auto_file_open,
    get_file_path,
    is_same_file,
    open,
)

__all__ = [
    "FileAccessWrapper",
    "FileAccessor",
    "auto_file_close",
    "auto_file_open",
    "get_file_path",
    "is_same_file",
    "open",
]

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("lazynwb")
logger.debug(f"{__name__}.{__version__ = }")


def testmod(**testmod_kwargs) -> doctest.TestResults:
    """Run doctests for this package."""
    _ = testmod_kwargs.setdefault(
        "optionflags", doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    return doctest.testmod(**testmod_kwargs)


if __name__ == "__main__":
    testmod()
