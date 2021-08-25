import numpy as np
import pandas as pd

from ..api import StatelessFeatureSet
from ..api.typing import LabelledExamples


class CUMULFeatures(StatelessFeatureSet):
    def __init__(self, cumul_features: int = 100):
        self.cumul_features = cumul_features

    def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        p_in = traces[traces["size"] >= 0].groupby(["site_id", "trace_id"]).agg(
            {'size': [np.count_nonzero, "sum"]}).rename({'count_nonzero': 'CUMUL_N_in', 'sum': 'CUMUL_S_in'}, axis=1)
        p_in.columns = p_in.columns.levels[1]

        p_out = traces[traces["size"] <= 0].groupby(["site_id", "trace_id"]).agg(
            {'size': [np.count_nonzero, "sum"]}).rename({'count_nonzero': 'CUMUL_N_out', 'sum': 'CUMUL_S_out'},
                                                        axis=1).abs()
        p_out.columns = p_out.columns.levels[1]

        p = p_in.join(p_out, how='outer')

        cumul = traces.groupby(["site_id", "trace_id"]).apply(self.__trace_cumul)

        f: pd.DataFrame = p.join(cumul, how='outer').fillna(0).reset_index()
        del f["trace_id"]

        labels = f.pop('site_id').values

        return f.values, labels

    def __trace_cumul(self, trace: pd.DataFrame) -> pd.Series:
        repr = trace['size'].cumsum()
        time = trace['time']

        max_time = time.max()
        interval = max_time / self.__count
        points = np.append(np.arange(1, self.__count) * interval, max_time)

        return pd.Series(np.interp(points, time, repr)).rename(lambda p: f'CUMUL_SMP_{p}')

    @property
    def cumul_features(self):
        return self.__count

    @cumul_features.setter
    def cumul_features(self, cumul_features: int):
        if cumul_features < 0 or not isinstance(cumul_features, int):
            raise ValueError("The number of CUMUL features must be a non-negative integer")

        self.__count = cumul_features

    def get_params(self, deep):
        return {'cumul_features': self.cumul_features}

    def set_params(self, **params):
        if 'cumul_features' in params:
            self.cumul_features = params['cumul_features']

