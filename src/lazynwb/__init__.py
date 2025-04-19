"""
An attempt to speed-up access to large NWB (Neurodata Without Borders) files stored in the cloud.
"""

import doctest
import importlib.metadata
import logging

from lazynwb.base import *
from lazynwb.file_io import *
from lazynwb.lazyframe import *
from lazynwb.tables import *
from lazynwb.timeseries import *
from lazynwb.utils import *

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("lazynwb")
logger.debug(f"{__name__}.{__version__ = }")


def testmod(**testmod_kwargs) -> doctest.TestResults:
    """
    Run doctests for the module, configured to ignore exception details and
    normalize whitespace.

    Accepts kwargs to pass to doctest.testmod().

    Add to modules to run doctests when run as a script:
    .. code-block:: text
        if __name__ == "__main__":
            from npc_io import testmod
            testmod()

    """
    _ = testmod_kwargs.setdefault(
        "optionflags", doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    return doctest.testmod(**testmod_kwargs)


if __name__ == "__main__":
    testmod()
