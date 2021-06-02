from logging import getLogger
from typing import Set

import numpy as np
import pandas as pd

from .util import get_bursts, round_to_increment, fill_missing, markers
from ..api import FeatureSet
from ..api.typing import TrainTestSplit, TracesStream, Traces
from ..util.config import ConfigList
from ..util.pipeline import process_fenced

idx = pd.IndexSlice

_logger = getLogger(__name__)


class MarkerConfig:
    def __init__(self,
                 burst_size: bool = True,
                 burst_size_inc: int = 600,
                 burst_length: bool = True,
                 burst_length_steps: ConfigList = ConfigList([1, 2, 3, 6, 9, 14]),
                 html_size: bool = True,
                 html_size_inc: int = 600,
                 download_pct: bool = True,
                 download_pct_inc: int = 5,
                 packet_size_count: bool = True,
                 packet_size_count_inc: int = 2,
                 total_size: bool = True,
                 total_size_inc: int = 10_000,
                 total_count: bool = True,
                 total_count_inc: int = 15):

        if burst_size_inc < 1 or html_size_inc < 1 or packet_size_count_inc < 1 or total_size_inc < 1 or total_count_inc < 1:
            raise ValueError("All increments must be positive")
        if len(burst_length_steps) <= 1 or any(step <= 0 for step in burst_length_steps):
            raise ValueError("There must only be positive (at least two) burst length steps")

        self.__burst_size = burst_size
        self.__burst_size_inc = burst_size_inc

        self.__burst_length = burst_length
        self.__burst_length_steps: np.ndarray = np.append(
            np.array(sorted((int(s) for s in burst_length_steps.as_list()), reverse=True)), 0)

        self.__html_size = html_size
        self.__html_size_inc = html_size_inc

        self.__download_pct = download_pct
        self.__download_pct_inc = download_pct_inc

        self.__packet_size_count = packet_size_count
        self.__packet_size_count_inc = packet_size_count_inc

        self.__total_size = total_size
        self.__total_size_inc = total_size_inc

        self.__total_count = total_count
        self.__total_count_inc = total_count_inc

    def round_burst_size(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__burst_size_inc)

    def round_burst_lengths(self, l: np.ndarray) -> int:
        cleaned = np.where(l > self.__burst_length_steps[0], self.__burst_length_steps[0], l)
        digitized = self.__burst_length_steps[np.digitize(cleaned, self.__burst_length_steps)]

        return np.where(l > self.__burst_length_steps[0], l, digitized)

    def round_html_size(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__html_size_inc)

    def round_packet_size_count(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__packet_size_count_inc)

    def round_packet_count(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__total_count_inc)

    def round_pct(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__download_pct_inc)

    def round_bandwidth(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__total_size_inc)


class Markers(FeatureSet):
    def __init__(self, config: MarkerConfig = MarkerConfig()):
        self.__config = config
        self.__attributes: Set[str] = {"UNQ+", "UNQ-", "PCT+", "PCT-", "N+", "N-"}

    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = await process_fenced(self.__extract_markers, train_traces, test_traces)
        return fill_missing(train, self.__attributes), fill_missing(test, self.__attributes)

    async def __extract_markers(self, traces: Traces) -> Traces:
        bursts = traces.groupby(["site_id", "trace_id"]).apply(get_bursts)

        direction = np.clip(bursts["burst_size"], -1, 1)
        bursts["burst_size_rd"] = direction * self.__config.round_burst_size(np.abs(bursts["burst_size"]))
        bursts["burst_length"] = direction * self.__config.round_burst_lengths(bursts["burst_length"])

        size_markers = markers(bursts, "burst_size_rd", "S", self.__attributes)
        number_markers = markers(bursts, "burst_length", "N", self.__attributes)

        html = bursts.loc[idx[:, :, 2], ["burst_size"]].rename({"burst_size": "html_size"}, axis=1)
        html["html_size"] = self.__config.round_html_size(html["html_size"])
        html_markers = markers(html, "html_size", "H", self.__attributes)

        traces["sign"] = np.clip(traces["size"], -1, 1)
        traces_agg = pd.pivot_table(traces,
                                    columns="sign",
                                    index=["site_id", "trace_id"],
                                    values="size",
                                    aggfunc=["nunique", "sum", "count"],
                                    fill_value=0)
        traces_agg.columns = ['/'.join(str(c) for c in col).strip() for col in traces_agg.columns.values]
        traces_agg = traces_agg.rename(
            {
                "nunique/-1": "unique_up",
                "sum/-1": "bw_up",
                "count/-1": "count_up",
                "nunique/1": "unique_dw",
                "sum/1": "bw_dw",
                "count/1": "count_dw"
            },
            axis=1)

        if "sum/0" in traces_agg:
            del traces_agg["sum/0"]
            del traces_agg["nunique/0"]
            del traces_agg["count/0"]

        for col in {"unique_up", "bw_up", "count_up", "unique_dw", "bw_dw", "count_dw"}:
            if col not in traces_agg:
                traces_agg[col] = 0

        traces_agg["bw_up"] = -1 * self.__config.round_bandwidth(-1 * traces_agg["bw_up"])
        traces_agg["bw_dw"] = self.__config.round_bandwidth(traces_agg["bw_dw"])

        bw_up_markers = markers(traces_agg, "bw_up", "B", self.__attributes)
        bw_dw_markers = markers(traces_agg, "bw_dw", "B", self.__attributes)

        total = traces_agg["count_up"] + traces_agg["count_dw"]
        non_zero = total > 0

        non_zero_total = total[non_zero]

        pct_up = np.zeros(traces_agg.shape[0])
        pct_up[non_zero.values] = ((100. * traces_agg.loc[non_zero, "count_up"]) / non_zero_total).values

        pct_dw = np.zeros(traces_agg.shape[0])
        pct_dw[non_zero.values] = (100. * traces_agg.loc[non_zero, "count_dw"] / non_zero_total).values

        additional = pd.DataFrame(
            {
                "UNQ+": self.__config.round_packet_size_count(traces_agg["unique_dw"]),
                "UNQ-": self.__config.round_packet_size_count(traces_agg["unique_up"]),
                "N+": self.__config.round_packet_count(traces_agg["count_dw"]),
                "N-": self.__config.round_packet_count(traces_agg["count_up"]),
                "PCT+": self.__config.round_pct(pct_dw),
                "PCT-": self.__config.round_pct(pct_up)
            },
            index=size_markers.index)

        result = size_markers.join(number_markers, how="outer").join(html_markers, how="outer").join(
            bw_up_markers, how="outer").join(bw_dw_markers, how="outer").join(additional, how="outer").fillna(0)

        nans = np.argwhere(np.isnan(result.values))
        if nans.size > 0:
            print("Detected NaN for")

            for index in nans:
                print("Site ID/Trace ID:", result.index[index[0]], "Column:", result.columns[index[1]])
            print()

        return result

    def reset(self):
        self.__attributes = {"UNQ+", "UNQ-", "PCT+", "PCT-", "N+", "N-"}
