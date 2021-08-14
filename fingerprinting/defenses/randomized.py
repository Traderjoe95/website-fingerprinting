from abc import ABCMeta, abstractmethod
from typing import List, Type, Optional

import numpy as np
import numpy.random as rd
import pandas as pd

from ..api import Defense
from ..api.defense import StatelessDefense
from ..api.typing import Traces


def plugin_path() -> str:
    return 'core.defenses.randomized'


def defenses() -> List[Type[Defense]]:
    return [SessionRandom255, PacketRandom255, PacketRandomGaussian, SessionRandomGaussian, PacketRandomMTU]


_RNG = rd.default_rng()


class _RandomizedPadding(StatelessDefense, metaclass=ABCMeta):
    @abstractmethod
    def _sample(self, sizes: np.ndarray, site_ids: np.ndarray, trace_ids: np.ndarray) -> np.ndarray:
        pass

    def defend(self, traces: Traces) -> Traces:
        packets = traces["size"].values
        direction = np.clip(packets, -1, 1)

        sizes = np.abs(packets)
        need_padding = (sizes > 0) & (sizes < 1500)

        padding = np.zeros_like(sizes)
        padding[need_padding] = self._sample(sizes[need_padding], traces['site_id'].values[need_padding],
                                             traces['trace_id'].values[need_padding])
        traces['size'] = np.clip(traces['size'] + (direction * padding), -1500, 1500)

        return traces


class PacketRandomPadding(_RandomizedPadding, metaclass=ABCMeta):
    def _sample(self, sizes: np.ndarray, site_ids: np.ndarray, trace_ids: np.ndarray) -> np.ndarray:
        return self._sample_for_packets(sizes)

    @abstractmethod
    def _sample_for_packets(self, sizes: np.ndarray) -> np.ndarray:
        pass


class SessionRandomPadding(_RandomizedPadding, metaclass=ABCMeta):
    def _sample(self, sizes: np.ndarray, site_ids: np.ndarray, trace_ids: np.ndarray) -> np.ndarray:
        traces = pd.MultiIndex.from_arrays([site_ids, trace_ids])

        max_sizes = pd.Series(data=sizes, index=traces).groupby(level=[0, 1]).max()
        padding = pd.Series(data=self._sample_for_sessions(max_sizes.values), index=max_sizes.index)

        return padding[traces].values

    @abstractmethod
    def _sample_for_sessions(self, session_max_sizes: np.ndarray) -> np.ndarray:
        pass


class SessionRandom255(SessionRandomPadding):
    def __init__(self):
        self.__range = np.arange(0, 255, 8)

    def _sample_for_sessions(self, session_max_sizes: np.ndarray) -> np.ndarray:
        return _RNG.choice(self.__range, session_max_sizes.shape)

    @staticmethod
    def name() -> str:
        return "session-random-255"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass


class PacketRandom255(PacketRandomPadding):
    def __init__(self):
        self.__range = np.arange(0, 255, 8)

    def _sample_for_packets(self, sizes: np.ndarray) -> np.ndarray:
        return _RNG.choice(self.__range, sizes.shape)

    @staticmethod
    def name() -> str:
        return "packet-random-255"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass


class PacketRandomGaussian(PacketRandomPadding):
    def __init__(self, expected_padding: int = 128):
        self.expected_padding = expected_padding

    @staticmethod
    def name() -> str:
        return f'packet-random-gaussian'

    @property
    def expected_padding(self) -> int:
        return self.__expected_padding

    @expected_padding.setter
    def expected_padding(self, expected_padding: int):
        self.__expected_padding = expected_padding

    def get_params(self):
        return {'expected_padding': self.expected_padding}

    def set_params(self, **params):
        if 'expected_padding' in params:
            self.expected_padding = int(params['expected_padding'])

    def _sample_for_packets(self, sizes: np.ndarray) -> np.ndarray:
        loc = _loc(sizes, self.__expected_padding, upper_bound=1500)
        scale = _scale(loc)

        return _truncated_rounded_normal(sizes, loc, scale, upper_bound=1500)


class SessionRandomGaussian(SessionRandomPadding):
    def __init__(self, expected_padding: int = 128):
        self.expected_padding = expected_padding

    @staticmethod
    def name() -> str:
        return 'session-random-gaussian'

    @property
    def expected_padding(self) -> int:
        return self.__expected_padding

    @expected_padding.setter
    def expected_padding(self, expected_padding: int):
        self.__expected_padding = expected_padding

    def get_params(self):
        return {'expected_padding': self.expected_padding}

    def set_params(self, **params):
        if 'expected_padding' in params:
            self.expected_padding = int(params['expected_padding'])

    def _sample_for_sessions(self, session_max_sizes: np.ndarray) -> np.ndarray:
        loc = _loc(session_max_sizes, self.__expected_padding)
        scale = _scale(loc)

        return _truncated_rounded_normal(session_max_sizes, loc, scale)


class PacketRandomMTU(PacketRandomPadding):
    def __init__(self):
        self.__steps = np.arange(0, 1500, 8)

    def _sample_for_packets(self, sizes: np.ndarray) -> np.ndarray:
        high = 1500 - sizes
        # Make this adjustment for the padding to REALLY be uniformly selected
        high = np.where(high % 8 == 0, high, (high // 8 + 1) * 8)

        rand_padding = _RNG.integers(size=sizes.shape, low=0, high=high, endpoint=True, dtype=np.uint16)
        step_idx = np.digitize(rand_padding, self.__steps, right=True)

        return np.clip(self.__steps[step_idx], 0, 1500 - sizes)

    @staticmethod
    def name() -> str:
        return "packet-random-mtu"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass


def _truncated_rounded_normal(sizes: np.ndarray, loc: np.ndarray, scale: np.ndarray,
                              upper_bound: Optional[int] = None) -> np.ndarray:
    padding = _rounded_normal(sizes.shape, loc, scale)

    def cond(s: np.ndarray, p: np.ndarray):
        c = (p < 0)

        if upper_bound is not None:
            c |= (s + p) > upper_bound

        return c

    while np.any(cond(sizes, padding)):
        oob = cond(sizes, padding)
        remaining = np.count_nonzero(oob)
        padding[oob] = _rounded_normal((remaining,), loc[oob], scale[oob])

    return padding


def _rounded_normal(shape, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.rint(_RNG.normal(size=shape, loc=loc, scale=scale)).astype(np.uint16)


def _scale(loc: np.ndarray) -> np.ndarray:
    return loc / 3.


def _loc(sizes: np.ndarray, expected_padding: int, upper_bound: Optional[int] = None) -> np.ndarray:
    if expected_padding > 0 and upper_bound is not None:
        return np.minimum(expected_padding, (upper_bound - sizes) // 2)
    elif expected_padding > 0:
        return np.ones_like(sizes) * expected_padding
    else:
        return (1500 - sizes) // 2
