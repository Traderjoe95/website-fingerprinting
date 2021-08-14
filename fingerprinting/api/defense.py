from abc import ABCMeta, abstractmethod
from typing import Iterable

import pandas as pd

from .pipeline import Defense
from .typing import TracesStream


class StreamableDefense(Defense, metaclass=ABCMeta):
    """
    The base class for a defense that allows fitting in batches (i.e. stream processing of data).
    """

    # noinspection PyTypeChecker,Mypy
    def fit(self, stream: TracesStream) -> 'StreamableDefense':
        return super(StreamableDefense, self).fit(stream)

    @abstractmethod
    def partial_fit(self, traces: pd.DataFrame) -> 'StreamableDefense':
        """
        Fits the defense to the given input traces but does not assume that the data is complete. This method
        should be used in a streaming setup, where calling `fit(traces)` can overwrite the previous parameters with
        every batch.

        :param traces: The data frame containing packet traces.

            This data frame is expected to be structured as explained on the Dataset.load method.
        :return: this instance for chaining
        """
        ...

    def _do_fit(self, traces_stream: TracesStream) -> 'StreamableDefense':
        for traces in traces_stream:
            self.partial_fit(traces)

        return self


class StatelessDefense(StreamableDefense, metaclass=ABCMeta):
    """
    The base class for a defense that does not require fitting.
    """

    # noinspection PyTypeChecker,Mypy
    def fit(self, stream: TracesStream) -> 'StatelessDefense':
        return super(StatelessDefense, self).fit(stream)

    def _do_fit(self, traces_stream: TracesStream) -> 'StatelessDefense':
        return self

    def partial_fit(self, traces: pd.DataFrame) -> 'StatelessDefense':
        return self

    def fit_defend(self, traces_stream: TracesStream) -> Iterable[pd.DataFrame]:
        for defended in self.defend_all(traces_stream):
            yield defended

    def reset(self):
        pass
