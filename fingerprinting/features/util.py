from typing import Set, AsyncGenerator

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..api.typing import TracesStream, LabelledExamples


def get_bursts(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.shape[0] == 1 and trace.loc[trace.index[0], "size"] == 0:
        return pd.DataFrame(np.zeros((2, 2)), columns=["burst_length", "burst_size"], index=pd.RangeIndex(1, 3))

    signs = np.clip(trace["size"], -1, 1)
    burst_id = (signs != signs.shift()).cumsum()

    bursts = trace["size"].groupby(burst_id).aggregate(["sum",
                                                        "count"]).rename({
                                                            "sum": "burst_size",
                                                            "count": "burst_length"
                                                        },
                                                                         axis=1)
    bursts["burst_id"] = burst_id.unique()

    return bursts[["burst_id", "burst_length", "burst_size"]].set_index("burst_id")


def round_to_increment(values: np.ndarray, inc: int) -> int:
    rounded = (np.rint(values / inc) * inc).astype(np.uint32)
    return np.where((rounded == 0) & (values != 0), 1, rounded)


async def fill_missing(traces_stream: TracesStream, attributes: Set[str]) -> AsyncGenerator[LabelledExamples, None]:
    async for traces in traces_stream:
        additional_att = [att for att in attributes if att not in traces]

        if len(additional_att) > 0:
            additional = pd.DataFrame.sparse.from_spmatrix(sp.csr_matrix((traces.shape[0], len(additional_att)),
                                                                         dtype=np.uint8),
                                                           columns=additional_att,
                                                           index=traces.index)
            traces = pd.concat([traces, additional], axis=1)

        traces = traces.reset_index()
        del traces["trace_id"]

        labels = traces.pop("site_id").values
        traces = traces[traces.columns.sort_values()]

        yield sp.csr_matrix(traces.values), labels


def markers(bursts: pd.DataFrame, column: str, marker_id: str, attributes: Set[str]) -> pd.DataFrame:
    if "unit" not in bursts:
        bursts["unit"] = 1

    markers = pd.pivot_table(bursts,
                             values="unit",
                             index=["site_id", "trace_id"],
                             columns=column,
                             fill_value=0,
                             aggfunc="sum")

    if 0 in markers:
        del markers[0]

    markers = markers.rename(lambda size: f"{marker_id}{size}", axis=1)
    markers.columns.name = None

    for c in markers:
        attributes.add(c)

    return markers
