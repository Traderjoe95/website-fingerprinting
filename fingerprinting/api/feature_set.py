from abc import ABCMeta, abstractmethod
from itertools import tee
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .pipeline import FeatureSet
from .typing import Transformer, StreamableTransformer, LabelledExamples, LabelledExampleStream, TrainTestSplit, \
    TracesStream, HasParams
from ..util.pipeline import collect, is_sparse_matrix


class StatelessFeatureSet(FeatureSet, metaclass=ABCMeta):
    """
    The base class for a feature set that does not require fitting.
    """

    def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        return self._extract_all(train_traces), self._extract_all(test_traces)

    def _extract_all(self, traces_stream: TracesStream) -> LabelledExampleStream:
        for traces in traces_stream:
            yield self._extract(traces)

    @abstractmethod
    def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        ...

    def reset(self):
        pass


class CombinedFeatureSet(FeatureSet):
    def __init__(self, left: FeatureSet, right: FeatureSet):
        self.__left = left
        self.__right = right

    def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        _left_train, _right_train = tee(train_traces, 2)
        _left_test, _right_test = tee(test_traces,  2)

        left_train, left_test = self.__left.extract_features(_left_train, _left_test)
        right_train, right_test = self.__right.extract_features(_right_train, _right_test)

        return _combine_all_features(left_train, right_train), _combine_all_features(left_test, right_test)

    def get_params(self):
        return _get_combined_params(self.__left, self.__right)

    def set_params(self, **params):
        _set_combined_params(self.__left, self.__right, params)

    def reset(self):
        self.__left.reset()
        self.__right.reset()


class StatelessCombinedFeatureSet(StatelessFeatureSet):
    def __init__(self, left: StatelessFeatureSet, right: StatelessFeatureSet):
        self.__left = left
        self.__right = right

    def _extract(self, traces: pd.DataFrame) -> LabelledExamples:
        return _combine_features(self.__left._extract(traces), self.__right._extract(traces))

    def get_params(self):
        return _get_combined_params(self.__left, self.__right)

    def set_params(self, **params):
        _set_combined_params(self.__left, self.__right, params)


def _get_combined_params(left: HasParams, right: HasParams,
                         left_tag: str = 'left', right_tag: str = 'right'):
    params = {f'{left_tag}__{name}': value for name, value in left.get_params().items()}
    params.update({f'{right_tag}__{name}': value for name, value in right.get_params().items()})

    return params


def _filter_params(params: Dict[str, Any], tag: str) -> Dict[str, Any]:
    return {name[len(tag) + 2:]: value for name, value in params.items() if name.startswith(f'{tag}__')}


def _set_combined_params(left: HasParams, right: HasParams, params: Dict[str, Any],
                         left_tag: str = 'left', right_tag: str = 'right'):
    left_params = _filter_params(params, left_tag)
    right_params = _filter_params(params, right_tag)

    left.set_params(**left_params)
    right.set_params(**right_params)


class TransformedFeatureSet(FeatureSet):
    def __init__(self, features: FeatureSet, transformer: Transformer):
        self.__features = features
        self.__transformer = transformer

    def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        train, test = self.__features.extract_features(train_traces, test_traces)
        train_trans, train = tee(train, 2)

        if isinstance(self.__transformer, StreamableTransformer):
            for examples, labels in train_trans:
                self.__transformer.partial_fit(examples, labels)
        else:
            examples, labels = collect(train_trans)
            self.__transformer.fit(examples, labels)

        return self.__transform_all(train), self.__transform_all(test)

    def __transform_all(self, features: LabelledExampleStream) -> LabelledExampleStream:
        for examples, labels in features:
            yield self.__transformer.transform(examples), labels

    def get_params(self):
        return _get_combined_params(self.__features, self.__transformer, left_tag='features', right_tag='transformer')

    def set_params(self, **params):
        _set_combined_params(self.__features, self.__transformer, params, left_tag='features', right_tag='transformer')

    def reset(self):
        self.__features.reset()


def _combine_all_features(
    left_stream: LabelledExampleStream, right_stream: LabelledExampleStream
) -> LabelledExampleStream:
    for left, right in zip(left_stream, right_stream):
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
