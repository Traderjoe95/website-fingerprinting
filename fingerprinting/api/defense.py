from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator

import pandas as pd

from .pipeline import Defense
from .typing import TracesStream


class StreamableDefense(Defense, metaclass=ABCMeta):
    """
    The base class for a defense that allows fitting in batches (i.e. stream processing of data).
    """

    # noinspection PyTypeChecker,Mypy
    async def fit(self, stream: TracesStream) -> 'StreamableDefense':
        return await super(StreamableDefense, self).fit(stream)

    @abstractmethod
    async def partial_fit(self, traces: pd.DataFrame) -> 'StreamableDefense':
        """
        Fits the defense to the given input traces but does not assume that the data is complete. This method
        should be used in a streaming setup, where calling `fit(traces)` can overwrite the previous parameters with
        every batch.

        :param traces: The data frame containing packet traces.

            This data frame is expected to be structured as explained on the Dataset.load method.
        :return: this instance for chaining
        """
        ...

    async def _do_fit(self, traces_stream: TracesStream) -> 'StreamableDefense':
        async for traces in traces_stream:
            await self.partial_fit(traces)

        return self


class StatelessDefense(StreamableDefense, metaclass=ABCMeta):
    """
    The base class for a defense that does not require fitting.
    """

    # noinspection PyTypeChecker,Mypy
    async def fit(self, stream: TracesStream) -> 'StatelessDefense':
        return await super(StatelessDefense, self).fit(stream)

    async def _do_fit(self, traces_stream: TracesStream) -> 'StatelessDefense':
        return self

    async def partial_fit(self, traces: pd.DataFrame) -> 'StatelessDefense':
        return self

    async def fit_defend(self, traces_stream: TracesStream) -> AsyncGenerator[pd.DataFrame, None]:
        async for defended in self.defend_all(traces_stream):
            yield defended

    def reset(self):
        pass
