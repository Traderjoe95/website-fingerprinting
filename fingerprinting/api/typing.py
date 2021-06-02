from datetime import timedelta
from typing import Tuple, Union, Iterable, AsyncIterable, Callable, Awaitable

import numpy as np
import pandas as pd
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasParams(Protocol):
    def get_params(self):
        ...

    def set_params(self, **params):
        ...


# noinspection PyPep8Naming
@runtime_checkable
class Transformer(HasParams, Protocol):
    def fit(self, X, y=None, **fit_params):
        ...

    def transform(self, X):
        ...

    def fit_transform(self, X, y=None, **fit_params):
        ...


# noinspection PyPep8Naming
@runtime_checkable
class StreamableTransformer(Transformer, Protocol):
    def partial_fit(self, X, y=None, **fit_params):
        ...


# noinspection PyPep8Naming
@runtime_checkable
class Classifier(HasParams, Protocol):
    def fit(self, X, y, **fit_params):
        ...

    def predict(self, X):
        ...


# noinspection PyPep8Naming
@runtime_checkable
class StreamableClassifier(Classifier, Protocol):
    def partial_fit(self, X, y, **fit_params):
        ...


OffsetOrDelta = Union[int, timedelta]
SiteSelection = Union[None, int, range, Iterable[int]]

Traces = pd.DataFrame
TracesStream = AsyncIterable[pd.DataFrame]
TraceProcessor = Union[Callable[[pd.DataFrame], pd.DataFrame], Callable[[pd.DataFrame], Awaitable[pd.DataFrame]]]

Examples = np.ndarray
Labels = np.ndarray
LabelledExamples = Tuple[Examples, Labels]
ExampleStream = AsyncIterable[Examples]
LabelledExampleStream = AsyncIterable[LabelledExamples]
TrainTestSplit = Tuple[LabelledExampleStream, LabelledExampleStream]

Metric = Callable[[np.ndarray, np.ndarray], float]
MetricOrName = Union[str, Metric]
