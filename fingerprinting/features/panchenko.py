from logging import getLogger
from typing import Set, List, Optional

import numpy as np
import pandas as pd

from .util import get_bursts, round_to_increment, fill_missing, markers, empty_markers
from ..api import FeatureSet
from ..api.typing import TrainTestSplit, TracesStream, Traces
from ..util.pipeline import process_fenced

idx = pd.IndexSlice

_logger = getLogger(__name__)


class MarkerConfig:
    def __init__(self,
                 burst_size: bool = True,
                 burst_size_inc: int = 600,
                 burst_length: bool = True,
                 burst_length_steps: Optional[List[int]] = None,
                 html_size: bool = True,
                 html_size_inc: int = 600,
                 download_pct: bool = True,
                 download_pct_inc: int = 5,
                 packet_size_count: bool = True,
                 packet_size_count_inc: int = 2,
                 bandwidth: bool = True,
                 bandwidth_inc: int = 10_000,
                 total_count: bool = True,
                 total_count_inc: int = 15):

        if burst_length_steps is None:
            burst_length_steps = [1, 2, 3, 6, 9, 14]

        if any(it < 1 for it in [burst_size_inc, html_size_inc, download_pct_inc, packet_size_count_inc, bandwidth_inc,
                                 total_count_inc]):
            raise ValueError('All increments must be positive')
        if len(burst_length_steps) <= 1 or any(step <= 0 for step in burst_length_steps):
            raise ValueError('There must only be positive and at least two burst length steps')

        self.__burst_size = burst_size
        self.__burst_size_inc = burst_size_inc

        self.__burst_length = burst_length
        self.__burst_length_steps: np.ndarray = np.append(
            np.array(sorted((int(s) for s in burst_length_steps), reverse=True)), 0)

        self.__html_size = html_size
        self.__html_size_inc = html_size_inc

        self.__download_pct = download_pct
        self.__download_pct_inc = download_pct_inc

        self.__packet_size_count = packet_size_count
        self.__packet_size_count_inc = packet_size_count_inc

        self.__bandwidth = bandwidth
        self.__bandwidth_inc = bandwidth_inc

        self.__total_count = total_count
        self.__total_count_inc = total_count_inc

    @property
    def dict(self):
        return {
            'burst_size': self.__burst_size, 'burst_size_inc': self.__burst_size_inc,
            'burst_length': self.__burst_length, 'burst_length_steps': self.__burst_length_steps[:-1],  # remove the 0
            'html_size': self.__html_size, 'html_size_inc': self.__html_size_inc,
            'download_pct': self.__download_pct, 'download_pct_inc': self.__download_pct_inc,
            'packet_size_count': self.__packet_size_count, 'packet_size_count_inc': self.__packet_size_count_inc,
            'bandwidth': self.__bandwidth, 'bandwidth_inc': self.__bandwidth_inc,
            'total_count': self.__total_count, 'total_count_inc': self.__total_count_inc
        }

    @property
    def burst_size_enabled(self) -> bool:
        return self.__burst_size

    def round_burst_size(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__burst_size_inc)

    @property
    def burst_length_enabled(self) -> bool:
        return self.__burst_length

    def round_burst_lengths(self, lens: np.ndarray) -> int:
        cleaned = np.where(lens > self.__burst_length_steps[0], self.__burst_length_steps[0], lens)
        digitized = self.__burst_length_steps[np.digitize(cleaned, self.__burst_length_steps)]

        return np.where(lens > self.__burst_length_steps[0], lens, digitized)

    @property
    def html_size_enabled(self) -> bool:
        return self.__html_size

    def round_html_size(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__html_size_inc)

    @property
    def download_pct_enabled(self) -> bool:
        return self.__download_pct

    def round_pct(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__download_pct_inc)

    @property
    def packet_size_count_enabled(self) -> bool:
        return self.__packet_size_count

    def round_packet_size_count(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__packet_size_count_inc)

    @property
    def total_count_enabled(self) -> bool:
        return self.__total_count

    def round_packet_count(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__total_count_inc)

    @property
    def bandwidth_enabled(self) -> bool:
        return self.__bandwidth

    def round_bandwidth(self, s: np.ndarray) -> int:
        return round_to_increment(s, self.__bandwidth_inc)


class Markers(FeatureSet):
    def __init__(self, **params):
        self.__config = MarkerConfig(**params)
        self.__attributes: Set[str] = {'P_UNQ+', 'P_UNQ-', 'P_PCT-', 'P_PCT-', 'P_N+', 'P_N-'}

    def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = process_fenced(self.__extract_markers, train_traces, test_traces)
        return fill_missing(train, self.__attributes), fill_missing(test, self.__attributes)

    def __extract_markers(self, traces: Traces) -> Traces:
        bursts = traces.groupby(['site_id', 'trace_id']).apply(get_bursts)

        direction = np.clip(bursts['burst_size'], -1, 1)

        if self.__config.burst_size_enabled:
            bursts['burst_size_rd'] = direction * self.__config.round_burst_size(np.abs(bursts['burst_size']))
            size_markers = markers(bursts, 'burst_size_rd', 'P_S', self.__attributes)
        else:
            size_markers = empty_markers(traces)

        if self.__config.burst_length_enabled:
            bursts['burst_length'] = direction * self.__config.round_burst_lengths(bursts['burst_length'])
            number_markers = markers(bursts, 'burst_length', 'P_N', self.__attributes)
        else:
            number_markers = empty_markers(traces)

        if self.__config.html_size_enabled:
            html = bursts.loc[idx[:, :, 2], ['burst_size']].rename({'burst_size': 'html_size'}, axis=1)
            html['html_size'] = self.__config.round_html_size(html['html_size'])
            html_markers = markers(html, 'html_size', 'P_H', self.__attributes)
        else:
            html_markers = empty_markers(traces)

        traces['sign'] = np.clip(traces['size'], -1, 1)
        traces_agg = pd.pivot_table(traces,
                                    columns='sign',
                                    index=['site_id', 'trace_id'],
                                    values='size',
                                    aggfunc=['nunique', 'sum', 'count'],
                                    fill_value=0)
        traces_agg.columns = ['/'.join(str(c) for c in col).strip() for col in traces_agg.columns.values]
        traces_agg = traces_agg.rename(
            {
                'nunique/-1': 'unique_up',
                'sum/-1': 'bw_up',
                'count/-1': 'count_up',
                'nunique/1': 'unique_dw',
                'sum/1': 'bw_dw',
                'count/1': 'count_dw'
            },
            axis=1)

        if 'sum/0' in traces_agg:
            del traces_agg['sum/0']
            del traces_agg['nunique/0']
            del traces_agg['count/0']

        for col in {'unique_up', 'bw_up', 'count_up', 'unique_dw', 'bw_dw', 'count_dw'}:
            if col not in traces_agg:
                traces_agg[col] = 0

        if self.__config.bandwidth_enabled:
            traces_agg['bw_up'] = -1 * self.__config.round_bandwidth(-1 * traces_agg['bw_up'])
            traces_agg['bw_dw'] = self.__config.round_bandwidth(traces_agg['bw_dw'])

            bw_up_markers = markers(traces_agg, 'bw_up', 'P_B', self.__attributes)
            bw_dw_markers = markers(traces_agg, 'bw_dw', 'P_B', self.__attributes)
        else:
            bw_up_markers = empty_markers(traces)
            bw_dw_markers = empty_markers(traces)

        additional = empty_markers(traces)

        if self.__config.packet_size_count_enabled:
            additional['P_UNQ+'] = self.__config.round_packet_size_count(traces_agg['unique_dw'])
            additional['P_UNQ-'] = self.__config.round_packet_size_count(traces_agg['unique_up'])
        if self.__config.total_count_enabled:
            additional['P_N+'] = self.__config.round_packet_count(traces_agg['count_dw'])
            additional['P_N-'] = self.__config.round_packet_count(traces_agg['count_up'])
        if self.__config.download_pct_enabled:
            total = traces_agg['count_up'] + traces_agg['count_dw']
            non_zero = total > 0

            non_zero_total = total[non_zero]

            pct_up = np.zeros(traces_agg.shape[0])
            pct_up[non_zero.values] = ((100. * traces_agg.loc[non_zero, 'count_up']) / non_zero_total).values

            pct_dw = np.zeros(traces_agg.shape[0])
            pct_dw[non_zero.values] = (100. * traces_agg.loc[non_zero, 'count_dw'] / non_zero_total).values

            additional['P_PCT+'] = self.__config.round_pct(pct_dw)
            additional['P_PCT-'] = self.__config.round_pct(pct_up)

        result = size_markers.join(number_markers, how='outer').join(html_markers, how='outer').join(
            bw_up_markers, how='outer').join(bw_dw_markers, how='outer').join(additional, how='outer').fillna(0)

        nans = np.argwhere(np.isnan(result.values))
        if nans.size > 0:
            print('Detected NaN for')

            for index in nans:
                print('Site ID/Trace ID:', result.index[index[0]], 'Column:', result.columns[index[1]])
            print()

        return result

    def reset(self):
        self.__attributes = set()

        if self.__config.packet_size_count_enabled:
            self.__attributes.update({'P_UNQ+', 'P_UNQ-'})

        if self.__config.download_pct_enabled:
            self.__attributes.update({'P_PCT+', 'P_PCT-'})

        if self.__config.total_count_enabled:
            self.__attributes.update({'P_N+', 'P_N-'})

    def get_params(self, deep):
        return self.__config.dict

    def set_params(self, **params):
        merged = self.__config.dict
        merged.update(params)

        self.__config = MarkerConfig(**merged)
