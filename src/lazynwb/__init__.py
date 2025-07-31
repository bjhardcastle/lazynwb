"""
An attempt to speed-up access to large NWB (Neurodata Without Borders) files stored in the cloud.
"""

import doctest
import importlib.metadata
import logging

from lazynwb.base import *
from lazynwb.conversion import *
from lazynwb.file_io import *
from lazynwb.lazyframe import *
from lazynwb.tables import *
from lazynwb.timeseries import *
from lazynwb.utils import *

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("lazynwb")
logger.debug(f"{__name__}.{__version__ = }")

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
