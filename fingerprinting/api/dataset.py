import re
from typing import Optional, List, Union, Set, Callable
from os.path import join

import numpy as np
import pandas as pd

from fingerprinting.api.typing import SiteSelection


_FILTERS = {
    'drop_ack': lambda traces: traces[np.abs(traces["size"]) != 52].copy(),
    'time_as_int': lambda traces: traces.assign(time=traces['time'].astype(np.uint32))
}


class DatasetConfig:
    def __init__(self,
                 default_name: str,
                 name: Optional[str] = None,
                 path: Optional[str] = None,
                 sites: Union[List[int], str, range, Set[int], int] = None,
                 filters: Optional[List[str]] = None):
        if filters is None:
            filters = []
        if name is None:
            name = default_name

        if sites is not None and isinstance(sites, str):
            if sites.startswith("csv:"):
                sites = list(pd.read_csv(sites[4:])['site_id'].values)
            else:
                sites = re.sub(r'\s+', '', sites)

                pattern = re.compile(r'^(?P<start>0|[1-9][0-9]*)\.\.(?P<end>[1-9][0-9]*)(%(?P<step>[1-9][0-9]*))?$')
                match = pattern.fullmatch(sites)

                if match is None:
                    raise ValueError(f"Invalid range syntax: {sites}")

                start = int(match.group("start"))
                end = int(match.group("end"))
                step = int(match.group("step") or "1")

                if end <= start:
                    raise ValueError("The range end point must be greater than the start point, "
                                     f"but got {end} <= {start} for {sites}")

                sites = range(start, end, step)
        elif sites is not None and isinstance(sites, int):
            if sites < 2:
                raise ValueError(f"Must at least allow 2 sites, but got {sites}")
        else:
            if sites is not None and len(sites) < 2:
                raise ValueError(f"Must at least allow 2 sites, but got {sites}")

        if isinstance(sites, list):
            sites = set(sites)

        self.__name = name
        self.__path = path or join("data", self.name)

        self.__sites = sites

        self.__filters: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

        for f in set(f.lower() for f in filters):
            if f not in _FILTERS:
                raise ValueError(f"Unknown filter '{f}', possible values are {set(_FILTERS.keys())}")

            self.__filters.append(_FILTERS[f])


    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @property
    def sites(self) -> SiteSelection:
        return self.__sites

    @property
    def filters(self) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
        return self.__filters

