from abc import ABCMeta, abstractmethod
from typing import Set

import numpy as np
import pandas as pd

from .util import fill_missing
from ..api import FeatureSet
from ..api.typing import TracesStream, TrainTestSplit, Traces
from ..util.pipeline import process_fenced


class SimpleFeatureSet(FeatureSet, metaclass=ABCMeta):
    def __init__(self, *, keep_ack: bool = True):
        self.__keep_ack = keep_ack
        self.__attributes: Set[str] = set()

    @abstractmethod
    def _aggregation(self):
        ...

    @abstractmethod
    def _dtype(self):
        ...

    def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = process_fenced(self.__extract_data, train_traces, test_traces)
        return fill_missing(train, self.__attributes), fill_missing(test, self.__attributes)

    def __extract_data(self, traces: Traces) -> Traces:
        traces["unit"] = 1

        pivot = pd.pivot_table(traces,
                               values="unit",
                               index=["site_id", "trace_id"],
                               columns="size",
                               fill_value=0,
                               aggfunc=self._aggregation())

        min_upstream = -52 if self.__keep_ack else -53
        min_downstream = 52 if self.__keep_ack else 53

        for size in range(min_upstream + 1, min_downstream):
            if size in pivot:
                del pivot[size]

        pivot = pivot.rename(lambda pkt: f"hist{pkt}", axis=1)

        for c in pivot:
            self.__attributes.add(c)

        return pivot.astype(self._dtype())

    def reset(self):
        self.__attributes = set()

    def get_params(self):
        return {'keep_ack': self.__keep_ack}

    def set_params(self, **params):
        if 'keep_ack' in params:
            self.__keep_ack = bool(params['keep_ack'])


class PacketHistogramFeatureSet(SimpleFeatureSet):
    def __init__(self, *, keep_ack: bool = True):
        super(PacketHistogramFeatureSet, self).__init__(keep_ack=keep_ack)

    def _aggregation(self):
        return "sum"

    def _dtype(self):
        return np.uint16


class PacketSetFeatureSet(SimpleFeatureSet):
    def __init__(self, *, keep_ack: bool = True):
        super(PacketSetFeatureSet, self).__init__(keep_ack=keep_ack)

    def _aggregation(self):
        return "max"

    def _dtype(self):
        return np.uint8
