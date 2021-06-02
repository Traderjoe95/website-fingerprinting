from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, AsyncIterable, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from asyncstdlib.itertools import tee, zip

from .pipeline import FeatureSet
from .typing import Transformer, StreamableTransformer, LabelledExamples, TrainTestSplit, TracesStream
from ..util.pipeline import collect, is_sparse_matrix


class StatelessFeatureSet(FeatureSet, metaclass=ABCMeta):
    """
    The base class for a feature set that does not require fitting.
    """
    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        return self._extract_all(train_traces), self._extract_all(test_traces)

    async def _extract_all(self, traces_stream: TracesStream) -> AsyncGenerator[LabelledExamples, None]:
        async for traces in traces_stream:
            yield await self._extract(traces)

    @abstractmethod
    async def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        ...

    def reset(self):
        pass


class CombinedFeatureSet(FeatureSet):
    def __init__(self, left: FeatureSet, right: FeatureSet):
        self.__left = left
        self.__right = right

    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        _left_train, _right_train = tee(train_traces, n=2)
        _left_test, _right_test = tee(test_traces, n=2)

        left_train, left_test = await self.__left.extract_features(_left_train, _left_test)
        right_train, right_test = await self.__right.extract_features(_right_train, _right_test)

        return _combine_all_features(left_train, right_train), _combine_all_features(left_test, right_test)

    def reset(self):
        pass


class StatelessCombinedFeatureSet(StatelessFeatureSet):
    def __init__(self, left: StatelessFeatureSet, right: StatelessFeatureSet):
        self.__left = left
        self.__right = right

    async def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        return _combine_features(await self.__left._extract(traces), await self.__right._extract(traces))


class TransformedFeatureSet(FeatureSet):
    def __init__(self, features: FeatureSet, transformer: Transformer):
        self.__features = features
        self.__transformer = transformer

    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = await self.__features.extract_features(train_traces, test_traces)
        train_trans, train = tee(train, n=2)

        if isinstance(self.__transformer, StreamableTransformer):
            async for examples, labels in train_trans:
                self.__transformer.partial_fit(examples, labels)
        else:
            examples, labels = await collect(train_trans)
            self.__transformer.fit(examples, labels)

        return self.__extract_all(train), self.__extract_all(test)

    async def __extract_all(self, examples_stream: AsyncIterable[LabelledExamples]) -> AsyncIterable[LabelledExamples]:
        async for examples, labels in examples_stream:
            yield self.__transformer.transform(examples), labels


async def _combine_all_features(
        left_stream: AsyncIterable[LabelledExamples],
        right_stream: AsyncIterable[LabelledExamples]) -> AsyncGenerator[LabelledExamples, None]:
    async for left, right in zip(left_stream, right_stream, strict=True):
        yield _combine_features(left, right)


def _combine_features(left: LabelledExamples, right: LabelledExamples) -> LabelledExamples:
    left_examples, left_labels = left
    right_examples, right_labels = right

    if left_examples.shape[0] != right_examples.shape[0]:
        raise ValueError("Found inconsistent example counts when combining features: "
                         f"{left_examples.shape[0]} != {right_examples.shape[0]}")

    if (left_labels != right_labels).any():
        raise ValueError("Found inconsistent class labels when combining features")

    if is_sparse_matrix(left_examples) or is_sparse_matrix(right_examples):
        examples = sp.hstack([left_examples, right_examples], format="csr")
    else:
        examples = np.concatenate([left_examples, right_examples], axis=1)

    return examples, left_labels
