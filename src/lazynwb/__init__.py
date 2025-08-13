"""
An attempt to speed-up access to large NWB (Neurodata Without Borders) files stored in the cloud.
"""

import doctest
import importlib.metadata
import logging

import dotenv

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


def load_dotenv() -> None:
    """
    Load environment variables from .env file in current working directory.

    >>> load_dotenv()
    """
    is_dotenv_used = dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    logger.debug(f"environment variables loaded from dotenv file: {is_dotenv_used}")


load_dotenv()

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
