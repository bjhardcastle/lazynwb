from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import inspect
import logging
import typing
from collections.abc import Iterable
from typing import Any, Literal, Protocol

import pandas as pd
import polars as pl
import tqdm
import upath

import lazynwb._config as lazynwb_config
import lazynwb.file_io
import lazynwb.tables
import lazynwb.timeseries
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)


def _metadata_read_scope(path: str) -> str:
    normalized_path = lazynwb.utils.normalize_internal_file_path(path)
    if normalized_path.startswith("general/subject"):
        return "subject"
    if normalized_path.startswith("general/"):
        return "session"
    return "top_level"


def _metadata_value_length(value: Any, shape: Any) -> int | None:
    if isinstance(shape, tuple | list):
        if len(shape) == 0:
            return 1
        return int(shape[0])
    with contextlib.suppress(TypeError):
        return len(value)
    return None


def _cast(accessor: lazynwb.file_io.FileAccessor, path: str) -> Any:
    """Read attribute from NWB file and interpret it as the appropriate Python object."""
    path = lazynwb.utils.normalize_internal_file_path(path)
    scope = _metadata_read_scope(path)
    logger.debug(
        "metadata read scope=%s source=%s backend=%s path=%s",
        scope,
        accessor._path.as_posix(),
        accessor._hdmf_backend.value,
        path,
    )
    v = accessor.get(path, None)
    if v is None:
        logger.debug(
            "metadata read scope=%s source=%s path=%s result=missing",
            scope,
            accessor._path.as_posix(),
            path,
        )
        return None
    shape = getattr(v, "shape", None)
    dtype = getattr(v, "dtype", None)
    if not getattr(v, "shape", True):
        v = [v[()]]
    value_length = _metadata_value_length(v, shape)
    if isinstance(v[0], bytes):
        s: str = v[0].decode()
        with contextlib.suppress(ValueError):
            result = datetime.datetime.fromisoformat(s)
            logger.debug(
                "metadata read scope=%s source=%s path=%s result=datetime "
                "shape=%s dtype=%s",
                scope,
                accessor._path.as_posix(),
                path,
                shape,
                dtype,
            )
            return result
        if s.startswith("[") and s.endswith("]") and s.count("[") == s.count("]") == 1:
            with contextlib.suppress(Exception):
                result = eval(s)
                logger.debug(
                    "metadata read scope=%s source=%s path=%s result=literal "
                    "shape=%s dtype=%s",
                    scope,
                    accessor._path.as_posix(),
                    path,
                    shape,
                    dtype,
                )
                return result
        if value_length is not None and value_length > 1:
            result = v.asstr()[:].tolist()
            logger.debug(
                "metadata read scope=%s source=%s path=%s result=list "
                "shape=%s dtype=%s",
                scope,
                accessor._path.as_posix(),
                path,
                shape,
                dtype,
            )
            return result
        logger.debug(
            "metadata read scope=%s source=%s path=%s result=scalar "
            "shape=%s dtype=%s",
            scope,
            accessor._path.as_posix(),
            path,
            shape,
            dtype,
        )
        return s
    if value_length is not None and value_length > 1:
        logger.debug(
            "metadata read scope=%s source=%s path=%s result=array "
            "shape=%s dtype=%s",
            scope,
            accessor._path.as_posix(),
            path,
            shape,
            dtype,
        )
        return v
    logger.debug(
        "metadata read scope=%s source=%s path=%s result=scalar " "shape=%s dtype=%s",
        scope,
        accessor._path.as_posix(),
        path,
        shape,
        dtype,
    )
    return v[0]


class LazyNWB:
    """
    PyNWB-like interface for accessing components of an NWB file.

    - initialize with a path to an NWB file or an open h5py.File, h5py.Group, or
    zarr.Group object

    - forwards attributes to the underlying NWB file accessor (h5py.File, h5py.Group), with
      intermediate objects used for convenient dot attr access. Will be slightly slower than
      accessing components directly with the NWB file accessor due to the overhead of creating
      python objects.

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb')
        >>> nwb.subject.date_of_birth
        datetime.datetime(2022, 2, 3, 0, 0, tzinfo=datetime.timezone.utc)
        >>> nwb.session_start_time
        datetime.datetime(2022, 8, 2, 15, 39, 59, tzinfo=datetime.timezone.utc)
    """

    _accessor: lazynwb.file_io.FileAccessor

    def __init__(
        self,
        path: lazynwb.types_.PathLike,
    ) -> None:
        self._file_path = lazynwb.file_io.from_pathlike(path)

    @property
    def _accessor(self) -> lazynwb.file_io.FileAccessor:
        """The underlying file accessor for this NWB file."""
        return lazynwb.file_io._get_accessor(self._file_path)

    def __repr__(self) -> str:
        return f"LazyNWB({self._file_path!r})"

    def _repr_html_(self) -> str:
        main_info = self._to_dict()
        subject_info = self.subject._to_dict()
        paths = self.describe().get("paths", [])

        html = f"""
        <h3>NWB file: {self._file_path}</h3>
        <ul>
        """
        for key, value in main_info.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value)) or "[]"
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"

        html += "<h4>Subject</h4><ul>"
        for key, value in subject_info.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"

        html += "<h4>Paths</h4><details><summary>Click to expand</summary><ul>"
        for path in paths:
            html += f"<li>{path}</li>"
        html += "</ul></details>"

        return html

    @property
    def identifier(self) -> str:
        return _cast(self._accessor, "identifier")

    @property
    def subject(self) -> Subject:
        return Subject(self._file_path)

    @property
    def session_start_time(self) -> datetime.datetime:
        return _cast(self._accessor, "session_start_time")

    @property
    def session_id(self) -> str:
        return _cast(self._accessor, "/general/session_id")

    @property
    def session_description(self) -> str:
        return _cast(self._accessor, "session_description")

    @property
    def trials(self) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            self._file_path, search_term="/intervals/trials", exact_path=True
        )

    @property
    def epochs(self) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            self._file_path, search_term="/intervals/epochs", exact_path=True
        )

    @property
    def electrodes(self) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            self._file_path,
            search_term="/general/extracellular_ephys/electrodes",
            exact_path=True,
        )

    @property
    def units(self) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            self._file_path,
            search_term="/units",
            exclude_array_columns=True,
            exact_path=True,
        ).pipe(lazynwb.tables.merge_array_column, "obs_intervals")

    @property
    def experiment_description(self) -> str:
        return _cast(self._accessor, "/general/experiment_description")

    @property
    def experimenter(self) -> str:
        return _cast(self._accessor, "/general/experimenter")

    @property
    def lab(self) -> str:
        return _cast(self._accessor, "/general/lab")

    @property
    def institution(self) -> str:
        return _cast(self._accessor, "/general/institution")

    @property
    def related_publications(self) -> str:
        return _cast(self._accessor, "/general/related_publications")

    @property
    def keywords(self) -> list[str]:
        k: str | Iterable[str] | None = _cast(self._accessor, "/general/keywords")
        if k is None:
            return []
        if isinstance(k, str):
            k = [k]
        return list(k)

    @property
    def notes(self) -> str:
        return _cast(self._accessor, "/general/notes")

    @property
    def data_collection(self) -> str:
        return _cast(self._accessor, "/general/data_collection")

    @property
    def surgery(self) -> str:
        return _cast(self._accessor, "/general/surgery")

    @property
    def pharmacology(self) -> str:
        return _cast(self._accessor, "/general/pharmacology")

    @property
    def virus(self) -> str:
        return _cast(self._accessor, "/general/virus")

    @property
    def source_script(self) -> str:
        return _cast(self._accessor, "/general/source_script")

    @property
    def source_script_file_name(self) -> str:
        return _cast(self._accessor, "/general/source_script_file_name")

    def _to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    def get_timeseries(
        self, search_term: str | None = None
    ) -> lazynwb.timeseries.TimeSeries:
        return lazynwb.timeseries.get_timeseries(
            self._file_path, search_term=search_term, match_all=False
        )

    def get(
        self,
        search_term: str,
        *,
        as_df: bool = False,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: bool | None = None,
    ) -> pd.DataFrame | pl.DataFrame | lazynwb.timeseries.TimeSeries:
        return get(
            nwb_data_sources=self._file_path,
            search_term=search_term,
            as_df=as_df,
            exact_path=exact_path,
            include_column_names=include_column_names,
            exclude_column_names=exclude_column_names,
            exclude_array_columns=exclude_array_columns,
            use_process_pool=use_process_pool,
            disable_progress=disable_progress,
            raise_on_missing=raise_on_missing,
            ignore_errors=ignore_errors,
            as_polars=as_polars,
        )

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: None = None,
    ) -> pd.DataFrame | pl.DataFrame: ...

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: Literal[True] = True,
    ) -> pl.DataFrame: ...

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: bool | None = None,
    ) -> pd.DataFrame | pl.DataFrame: ...

    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: bool | None = None,
    ) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            nwb_data_sources=self._file_path,
            search_term=search_term,
            exact_path=exact_path,
            include_column_names=include_column_names,
            exclude_column_names=exclude_column_names,
            exclude_array_columns=exclude_array_columns,
            use_process_pool=use_process_pool,
            disable_progress=disable_progress,
            raise_on_missing=raise_on_missing,
            ignore_errors=ignore_errors,
            as_polars=as_polars,
        )

    def describe(self) -> dict[str, Any]:
        return {
            **self._to_dict(),
            **self.subject._to_dict(),
            "paths": lazynwb.file_io.get_internal_paths(self._file_path),
        }


class NWBComponent(Protocol):
    @property
    def _accessor(self) -> lazynwb.file_io.FileAccessor: ...


def to_dict(obj: NWBComponent) -> dict[str, str | list[str] | datetime.datetime]:
    def _get_attr_names(obj: Any) -> list[str]:
        return [
            name
            for name, prop in obj.__class__.__dict__.items()
            if isinstance(prop, property)
            and any(t in inspect.signature(prop.fget).return_annotation for t in ("str", "list[str]", "datetime.datetime"))  # type: ignore[arg-type]
        ]

    results = {}
    for name in _get_attr_names(obj):
        results[name] = getattr(obj, name)
    return results


class Subject:
    _file_path: upath.UPath

    def __init__(
        self,
        path: lazynwb.types_.PathLike,
    ) -> None:
        self._file_path = lazynwb.file_io.from_pathlike(path)

    @property
    def _accessor(self) -> lazynwb.file_io.FileAccessor:
        """The underlying file accessor for this subject."""
        return lazynwb.file_io._get_accessor(self._file_path)

    def __repr__(self) -> str:
        return f"Subject({self._file_path!r})"

    @property
    def age(self) -> str | None:
        """The age of the subject. The ISO 8601 Duration format is recommended, e.g., “P90D” for 90 days old."""
        return _cast(self._accessor, "/general/subject/age")

    @property
    def age__reference(self) -> str | None:
        """Age is with reference to this event. Can be `birth` or `gestational`. If reference is omitted, then `birth` is implied. Value can be None when read from an NWB file with schema version 2.0 to 2.5 where age__reference is missing."""
        return _cast(self._accessor, "/general/subject/age__reference")

    @property
    def description(self) -> str | None:
        """A description of the subject, e.g., “mouse A10”."""
        return _cast(self._accessor, "/general/subject/description")

    @property
    def genotype(self) -> str | None:
        """The genotype of the subject, e.g., “Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP"""
        return _cast(self._accessor, "/general/subject/genotype")

    @property
    def sex(self) -> str | None:
        """The sex of the subject. Using “F” (female), “M” (male), “U” (unknown), or “O” (other) is recommended."""
        return _cast(self._accessor, "/general/subject/sex")

    @property
    def species(self) -> str | None:
        """The species of the subject. The formal latin binomal name is recommended, e.g., “Mus musculus”."""
        return _cast(self._accessor, "/general/subject/species")

    @property
    def subject_id(self) -> str | None:
        """A unique identifier for the subject, e.g., “A10”."""
        return _cast(self._accessor, "/general/subject/subject_id")

    @property
    def weight(self) -> str | None:
        """The weight of the subject, including units. Using kilograms is recommended. e.g., “0.02 kg”. If a float is provided, then the weight will be stored as “[value] kg”."""
        return _cast(self._accessor, "/general/subject/weight")

    @property
    def strain(self) -> str | None:
        """The strain of the subject, e.g., “C57BL/6J”."""
        return _cast(self._accessor, "/general/subject/strain")

    @property
    def date_of_birth(self) -> datetime.datetime | None:
        """The datetime of the date of birth. May be supplied instead of age."""
        return _cast(self._accessor, "/general/subject/date_of_birth")

    def _to_dict(self) -> dict[str, Any]:
        return to_dict(self)


def _source_paths(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
) -> tuple[str | lazynwb.types_.PathLike, ...]:
    if isinstance(nwb_data_sources, (str, bytes)) or not isinstance(
        nwb_data_sources, Iterable
    ):
        return (nwb_data_sources,)
    return tuple(nwb_data_sources)


def _path_info_key(path: str) -> str:
    normalized_path = lazynwb.utils.normalize_internal_file_path(path).rstrip("/")
    if normalized_path == "":
        return "/"
    return f"/{normalized_path.removeprefix('/')}"


def _get_matching_path(
    path_info: dict[str, dict[str, Any]],
    search_term: str,
    exact_path: bool,
) -> tuple[str, dict[str, Any]] | None:
    search_path = _path_info_key(search_term)
    match = path_info.get(search_path)
    if match is not None:
        return search_path, match
    if exact_path:
        return None

    timeseries_matches = tuple(
        (path, metadata)
        for path, metadata in path_info.items()
        if metadata["is_timeseries"] and search_term in path
    )
    if timeseries_matches:
        if len(timeseries_matches) > 1:
            logger.warning(
                "Found multiple timeseries matching %r: %s - returning first",
                search_term,
                [path for path, _ in timeseries_matches],
            )
        return timeseries_matches[0]
    return None


def _get_direct_container(
    nwb_data_source: lazynwb.types_.PathLike,
    search_term: str,
) -> tuple[str, dict[str, Any]] | None:
    search_path = _path_info_key(search_term)
    file = lazynwb.file_io._get_accessor(nwb_data_source)
    accessor = file.get(search_path, None)
    if accessor is None:
        return None
    return search_path, lazynwb.file_io._path_metadata_from_entry(
        search_path,
        accessor,
        {},
    )


def _get_inferred_container(
    nwb_data_source: lazynwb.types_.PathLike,
    search_term: str,
    exact_path: bool,
) -> tuple[str, dict[str, Any]] | None:
    direct_match = _get_direct_container(
        nwb_data_source=nwb_data_source,
        search_term=search_term,
    )
    if direct_match is not None or exact_path:
        logger.debug(
            "general get direct container inference source=%r search_term=%r "
            "exact_path=%s match=%s is_timeseries=%s",
            nwb_data_source,
            search_term,
            exact_path,
            None if direct_match is None else direct_match[0],
            None if direct_match is None else direct_match[1]["is_timeseries"],
        )
        return direct_match

    path_info = lazynwb.file_io.get_internal_path_info(nwb_data_source)
    match = _get_matching_path(
        path_info=path_info,
        search_term=search_term,
        exact_path=exact_path,
    )
    logger.debug(
        "general get container inference source=%r search_term=%r exact_path=%s "
        "match=%s is_timeseries=%s",
        nwb_data_source,
        search_term,
        exact_path,
        None if match is None else match[0],
        None if match is None else match[1]["is_timeseries"],
    )
    return match


def get(
    nwb_data_sources: (
        str | lazynwb.types_.PathLike | Iterable[str | lazynwb.types_.PathLike]
    ),
    search_term: str,
    *,
    as_df: bool = False,
    exact_path: bool = False,
    include_column_names: str | Iterable[str] | None = None,
    exclude_column_names: str | Iterable[str] | None = None,
    exclude_array_columns: bool = True,
    use_process_pool: bool = False,
    disable_progress: bool = False,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    as_polars: bool | None = None,
) -> pd.DataFrame | pl.DataFrame | lazynwb.timeseries.TimeSeries:
    """
    Return a DataFrame or TimeSeries from one or more NWB files.

    By default, TimeSeries containers return a :class:`lazynwb.TimeSeries` and
    table-like containers return a pandas or polars DataFrame. Set ``as_df=True``
    to force DataFrame materialization for any container, including TimeSeries.
    """
    paths = _source_paths(nwb_data_sources)
    if not paths:
        raise ValueError("At least one NWB source is required")
    data_sources: (
        str
        | lazynwb.types_.PathLike
        | tuple[
            str | lazynwb.types_.PathLike,
            ...,
        ]
    )
    data_sources = paths[0] if len(paths) == 1 else paths
    if as_df:
        logger.debug(
            "general get forced DataFrame path source_count=%d search_term=%r "
            "exact_path=%s as_polars=%s",
            len(paths),
            search_term,
            exact_path,
            as_polars,
        )
        return lazynwb.tables.get_df(
            nwb_data_sources=data_sources,
            search_term=search_term,
            exact_path=exact_path,
            include_column_names=include_column_names,
            exclude_column_names=exclude_column_names,
            exclude_array_columns=exclude_array_columns,
            use_process_pool=use_process_pool,
            disable_progress=disable_progress,
            raise_on_missing=raise_on_missing,
            ignore_errors=ignore_errors,
            as_polars=as_polars,
        )

    inferred_container = _get_inferred_container(
        nwb_data_source=paths[0],
        search_term=search_term,
        exact_path=exact_path,
    )
    if (
        inferred_container is not None
        and inferred_container[1]["is_timeseries"]
        and len(paths) > 1
    ):
        raise ValueError(
            "TimeSeries containers can only be returned for a single NWB source; "
            "set `as_df=True` to materialize matching TimeSeries from multiple "
            "sources as a DataFrame"
        )
    if inferred_container is not None and inferred_container[1]["is_timeseries"]:
        timeseries_path = inferred_container[0]
        logger.debug(
            "general get returning TimeSeries source=%r search_term=%r "
            "timeseries_path=%s",
            paths[0],
            search_term,
            timeseries_path,
        )
        return lazynwb.timeseries.TimeSeries(
            _file_path=paths[0],
            _table_path=timeseries_path,
        )

    table_search_term = (
        inferred_container[0] if inferred_container is not None else search_term
    )
    logger.debug(
        "general get returning DataFrame source_count=%d search_term=%r "
        "table_search_term=%r exact_path=%s as_polars=%s",
        len(paths),
        search_term,
        table_search_term,
        exact_path,
        as_polars,
    )
    return lazynwb.tables.get_df(
        nwb_data_sources=data_sources,
        search_term=table_search_term,
        exact_path=exact_path or inferred_container is not None,
        include_column_names=include_column_names,
        exclude_column_names=exclude_column_names,
        exclude_array_columns=exclude_array_columns,
        use_process_pool=use_process_pool,
        disable_progress=disable_progress,
        raise_on_missing=raise_on_missing,
        ignore_errors=ignore_errors,
        as_polars=as_polars,
    )


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    disable_progress: bool = False,
    as_polars: None = None,
) -> pd.DataFrame | pl.DataFrame: ...


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    disable_progress: bool = False,
    as_polars: Literal[False] = False,
) -> pd.DataFrame: ...


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    disable_progress: bool = False,
    as_polars: Literal[True] = True,
) -> pl.DataFrame: ...


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    disable_progress: bool = False,
    as_polars: bool | None = None,
) -> pd.DataFrame | pl.DataFrame: ...


def get_metadata_df(
    nwb_path_or_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    disable_progress: bool = False,
    as_polars: bool | None = None,
) -> pd.DataFrame | pl.DataFrame:
    as_polars = lazynwb_config._resolve_as_polars(as_polars)
    if isinstance(nwb_path_or_paths, str) or not isinstance(
        nwb_path_or_paths, Iterable
    ):
        paths = (nwb_path_or_paths,)
    else:
        paths = tuple(nwb_path_or_paths)

    def _get_metadata_df_helper(nwb_path: lazynwb.types_.PathLike) -> dict[str, Any]:
        nwb = LazyNWB(nwb_path)
        logger.debug(
            "metadata read scope=session+subject source=%s operation=get_metadata_df",
            nwb._file_path.as_posix(),
        )
        return {
            **nwb._to_dict(),
            **nwb.subject._to_dict(),
            lazynwb.tables.NWB_PATH_COLUMN_NAME: nwb._accessor._path.as_posix(),
        }

    logger.debug(
        "starting get_metadata_df metadata read scope=session+subject "
        "source_count=%d as_polars=%s",
        len(paths),
        as_polars,
    )
    future_to_path = {}
    for path in paths:
        future = lazynwb.utils.get_threadpool_executor().submit(
            _get_metadata_df_helper,
            nwb_path=path,
        )
        future_to_path[future] = path
    futures = concurrent.futures.as_completed(future_to_path)
    if not disable_progress:
        futures = tqdm.tqdm(
            futures,
            total=len(future_to_path),
            desc="Getting metadata",
            unit="file",
            ncols=80,
        )
    records = []
    for future in futures:
        path = future_to_path[future]
        try:
            records.append(future.result())
        except Exception:
            logger.error("Error processing %s:", path)
            raise
    logger.debug(
        "finished get_metadata_df metadata read scope=session+subject "
        "source_count=%d record_count=%d as_polars=%s",
        len(paths),
        len(records),
        as_polars,
    )
    if not as_polars:
        return pd.DataFrame.from_records(records)
    else:
        return pl.DataFrame(records)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
