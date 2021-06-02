from abc import ABCMeta, abstractmethod
from itertools import chain

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..api.feature_set import StatelessFeatureSet
from ..api.typing import LabelledExamples


class SimpleFeatureSet(StatelessFeatureSet, metaclass=ABCMeta):
    def __init__(self, *, keep_ack: bool = True):
        self.__keep_ack = keep_ack

    @abstractmethod
    def _aggregation(self):
        ...

    @abstractmethod
    def _dtype(self):
        ...

    async def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        traces["unit"] = 1

        pivot = pd.pivot_table(traces,
                               values="unit",
                               index=["site_id", "trace_id"],
                               columns="size",
                               fill_value=0,
                               aggfunc=self._aggregation())

        min_upstream = -52 if self.__keep_ack else -53
        min_downstream = 52 if self.__keep_ack else 53

        additional_cols = [
            size for size in chain(range(-1500, min_upstream + 1), range(min_downstream, 1501)) if size not in pivot
        ]

        if len(additional_cols) > 0:
            additional_df = pd.DataFrame.sparse.from_spmatrix(sp.csr_matrix((pivot.shape[0], len(additional_cols)),
                                                                            dtype=self._dtype()),
                                                              columns=additional_cols,
                                                              index=pivot.index)
            pivot = pd.concat([pivot, additional_df], axis=1)

        for size in range(min_upstream + 1, min_downstream):
            if size in pivot:
                del pivot[size]

        pivot = pivot[pivot.columns.sort_values()].astype(self._dtype()).reset_index()
        pivot.columns.name = None

        labels = pivot.pop("site_id").values
        del pivot["trace_id"]

        return sp.csr_matrix(pivot.values), labels


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
