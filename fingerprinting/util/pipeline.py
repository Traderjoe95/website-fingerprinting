from typing import Tuple, Generator, TypeVar, Iterator, AsyncIterable

import curio
import numpy as np
import scipy.sparse as sp
from asyncstdlib import anext

from ..api.typing import LabelledExamples, TracesStream, TraceProcessor, ExampleStream, Examples, \
    LabelledExampleStream


def collect(labelled_examples: LabelledExampleStream) -> LabelledExamples:
    _examples = []
    _labels = []

    for examples, labels in labelled_examples:
        _examples.append(examples)
        _labels.append(labels)

    examples = concat_sparse_aware(_examples)
    labels = concat_sparse_aware(_labels)

    return examples, labels


def concat_sparse_aware(_arrays):
    if any(is_sparse_matrix(item) for item in _arrays):
        return sp.vstack(_arrays)
    else:
        return np.concatenate(_arrays, axis=0)


def collect_examples(example_stream: ExampleStream) -> Examples:
    _examples = [ex for ex in example_stream]

    return concat_sparse_aware(_examples)


def drop_labels(labelled_examples: LabelledExampleStream) -> Generator[np.ndarray, None, None]:
    for examples, _ in labelled_examples:
        yield examples


def process_fenced(extract: TraceProcessor, train_traces: TracesStream,
                   test_traces: TracesStream) -> Tuple[TracesStream, TracesStream]:
    train_extracted = [extract(train) for train in train_traces]
    test_extracted = [extract(test) for test in test_traces]

    return iter(train_extracted), iter(test_extracted)


def is_sparse_matrix(arr) -> bool:
    return type(arr).__module__.startswith("scipy.sparse")


def dense(arr) -> np.ndarray:
    if is_sparse_matrix(arr):
        return arr.todense()
    return arr


T = TypeVar("T")


def synchronize(_aiter: AsyncIterable[T]) -> Iterator[T]:
    try:
        while True:
            yield curio.run(anext, _aiter)
    except StopAsyncIteration:
        pass
