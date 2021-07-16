from typing import AsyncIterable, AsyncGenerator, Tuple, Union, Awaitable

import numpy as np
import scipy.sparse as sp
from asyncstdlib.itertools import aiter

from ..api.typing import LabelledExamples, TracesStream, TraceProcessor, Traces, ExampleStream, Examples, \
    LabelledExampleStream


async def collect(labelled_examples: LabelledExampleStream) -> LabelledExamples:
    _examples = []
    _labels = []

    async for examples, labels in labelled_examples:
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


async def collect_examples(example_stream: ExampleStream) -> Examples:
    _examples = [ex async for ex in example_stream]

    return concat_sparse_aware(_examples)


async def drop_labels(labelled_examples: AsyncIterable[LabelledExamples]) -> AsyncGenerator[np.ndarray, None]:
    async for examples, _ in labelled_examples:
        yield examples


async def process_fenced(extract: TraceProcessor, train_traces: TracesStream,
                         test_traces: TracesStream) -> Tuple[TracesStream, TracesStream]:
    train_extracted = [await __await(extract(train)) async for train in train_traces]
    test_extracted = [await __await(extract(test)) async for test in test_traces]

    return aiter(train_extracted), aiter(test_extracted)


def is_sparse_matrix(arr) -> bool:
    return type(arr).__module__.startswith("scipy.sparse")


async def __await(traces: Union[Traces, Awaitable[Traces]]) -> Traces:
    if isinstance(traces, Traces):
        return traces
    return await traces
