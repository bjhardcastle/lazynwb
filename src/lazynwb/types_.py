from __future__ import annotations

import os
import pathlib
from typing import Union

import upath
from typing_extensions import TypeAlias

PathLike: TypeAlias = Union[str, bytes, os.PathLike, pathlib.Path, upath.UPath]
