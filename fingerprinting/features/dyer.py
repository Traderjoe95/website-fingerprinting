from typing import Set

import numpy as np
import pandas as pd

from .util import get_bursts, markers, fill_missing, round_to_increment
from ..api import StatelessFeatureSet, FeatureSet
from ..api.typing import LabelledExamples, TracesStream, TrainTestSplit, Traces
from ..util.pipeline import process_fenced


class Time(StatelessFeatureSet):
    async def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        time = traces[["site_id", "trace_id", "time"]].groupby(["site_id", "trace_id"], as_index=False).agg("max")
        del time["trace_id"]

        labels = time.pop("site_id")

        return time.values, labels.values


class Bandwidth(StatelessFeatureSet):
    async def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        down = traces[traces["size"] >= 0]
        up = traces[traces["size"] <= 0]

        bw_down = down[["site_id", "trace_id", "size"]].groupby(["site_id",
                                                                 "trace_id"]).agg("sum").rename({"size": "down"},
                                                                                                axis=1)
        bw_up = up[["site_id", "trace_id", "size"]].groupby(["site_id", "trace_id"]).agg("sum").rename({"size": "up"},
                                                                                                       axis=1)

        bw = bw_down.join(bw_up, how="outer").fillna(0).reset_index()
        del bw["trace_id"]

        labels = bw.pop("site_id")

        return np.abs(bw.values), labels.values


class VariableNGram(FeatureSet):
    def __init__(self):
        self.__attributes: Set[str] = set()

    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = await process_fenced(self.__extract_vng, train_traces, test_traces)
        return fill_missing(train, self.__attributes), fill_missing(test, self.__attributes)

    async def __extract_vng(self, traces: Traces) -> Traces:
        bursts = traces.groupby(["site_id", "trace_id"]).apply(get_bursts)

        direction = np.clip(bursts["burst_size"], -1, 1)
        bursts["vng_size"] = direction * round_to_increment(np.abs(bursts["burst_size"]), 600)

        features = markers(bursts, "vng_size", "S", self.__attributes)

        return features

    def reset(self):
        self.__attributes = set()