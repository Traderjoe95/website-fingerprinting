from abc import ABCMeta

import numpy as np
import numpy.random as rd
import pandas as pd

from fingerprinting.api.defense import StatelessDefense


class SessionRandom255(StatelessDefense):
    def __init__(self):
        self.__range = np.arange(0, 255, 8)

    async def defend(self, traces: pd.DataFrame) -> pd.DataFrame:
        trace_ids = pd.MultiIndex.from_frame(traces[["site_id", "trace_id"]]).unique()
        paddings = rd.choice(self.__range, trace_ids.shape[0])

        for idx, (site_id, trace_id) in enumerate(trace_ids):
            where = (traces["site_id"] == site_id) & (traces["trace_id"] == trace_id)
            direction = np.clip(traces.loc[where, "size"], -1, 1)
            traces.loc[where, "size"] += (direction * paddings[idx])

        traces["size"] = np.clip(traces["size"], -1500, 1500)

        return traces

    @property
    def name(self) -> str:
        return "session-random-255"


class PacketRandom255(StatelessDefense):
    def __init__(self):
        self.__range = np.arange(0, 255, 8)

    async def defend(self, traces: pd.DataFrame) -> pd.DataFrame:
        paddings = rd.choice(self.__range, traces.shape[0])
        direction = np.clip(traces["size"].values, -1, 1)

        traces["size"] = np.clip(traces["size"] + (paddings * direction), -1500, 1500)

        return traces

    @property
    def name(self) -> str:
        return "packet-random-255"


class DiscretizingPadding(StatelessDefense, metaclass=ABCMeta):
    def __init__(self, steps: np.ndarray):
        self.__steps = steps

    async def defend(self, traces: pd.DataFrame) -> pd.DataFrame:
        direction = np.clip(traces["size"].values, -1, 1)
        idxs = np.digitize(np.abs(traces["size"]), self.__steps, right=True)

        traces["size"] = self.__steps[idxs] * direction

        return traces


class LinearPadding(DiscretizingPadding):
    def __init__(self, increment: int = 128):
        if increment < 1 or increment > 1500:
            raise ValueError("The step size must be between 1 and 1500")

        super(LinearPadding, self).__init__(np.append(np.arange(0, 1500, increment), 1500))

    @property
    def name(self) -> str:
        return "linear-padding"


class ExponentialPadding(DiscretizingPadding):
    def __init__(self):
        super(ExponentialPadding, self).__init__(np.append(np.append(0, 2**np.arange(0, int(np.log2(1500)) + 1)), 1500))

    @property
    def name(self) -> str:
        return "exponential-padding"


class MiceElephantsPadding(DiscretizingPadding):
    def __init__(self):
        super(MiceElephantsPadding, self).__init__(np.array([0, 128, 1500]))

    @property
    def name(self) -> str:
        return "mice-elephants"


class PadToMTU(DiscretizingPadding):
    def __init__(self):
        super(PadToMTU, self).__init__(np.array([0, 1500]))

    @property
    def name(self) -> str:
        return "pad-to-mtu"


class PacketRandomMTU(StatelessDefense):
    def __init__(self):
        self.__pad = np.vectorize(_pad_upto_mtu)

    async def defend(self, traces: pd.DataFrame) -> pd.DataFrame:
        direction = np.clip(traces["size"].values, -1, 1)
        padded = self.__pad(np.abs(traces["size"]))

        traces["size"] = direction * padded

        return traces

    @property
    def name(self) -> str:
        return "packet-random-mtu"


def _pad_upto_mtu(s):
    choices = np.arange(0, 1500 - s, 8)

    if s == 0 or choices.size > 0:
        if choices[-1] != 1500 - s:
            choices = np.append(choices, 1500 - s)

        return rd.choice(choices, 1) + s
    else:
        return s
