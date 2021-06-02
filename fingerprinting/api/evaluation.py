import re
from datetime import timedelta
from typing import Optional, Dict, Any, Union, List, Iterator

import pendulum
from pendulum import duration
from sklearn.metrics import accuracy_score

from ..algorithms.metrics import resolve_metric
from .typing import MetricOrName, OffsetOrDelta, Metric

IntOrRange = Union[int, str, range, List[int]]
OffsetOrRange = Union[OffsetOrDelta, str, range, List[Union[str, OffsetOrDelta]]]


class EvaluationParams:
    def __init__(self, websites: int, runs: int, metric: Metric, train_offset: OffsetOrDelta,
                 train_test_delta: OffsetOrDelta, train_examples: int, test_examples: int,
                 feature_set: Optional[Dict[str, Dict[str, Any]]], classifier: Optional[Dict[str, Dict[str, Any]]]):
        self.__websites = websites
        self.__runs = runs
        self.__metric = metric
        self.__train_offset = train_offset
        self.__train_test_delta = train_test_delta
        self.__train_examples = train_examples
        self.__test_examples = test_examples
        self.__feature_set = feature_set
        self.__classifier = classifier

    @property
    def websites(self) -> int:
        return self.__websites

    @property
    def runs(self) -> int:
        return self.__runs

    @property
    def metric(self) -> Metric:
        return self.__metric

    @property
    def train_offset(self) -> OffsetOrDelta:
        return self.__train_offset

    @property
    def train_test_delta(self) -> OffsetOrDelta:
        return self.__train_test_delta

    @property
    def train_examples(self) -> int:
        return self.__train_examples

    @property
    def test_examples(self) -> int:
        return self.__test_examples

    def feature_set_params(self, attack_name: str) -> Optional[Dict[str, Any]]:
        if self.__feature_set is None:
            return None

        return self.__feature_set[attack_name] if attack_name in self.__feature_set else None

    def classifier_params(self, attack_name: str) -> Optional[Dict[str, Any]]:
        if self.__classifier is None:
            return None

        return self.__classifier[attack_name] if attack_name in self.__classifier else None


class EvaluationConfig:
    def __init__(self,
                 websites: IntOrRange = 2,
                 runs: int = 10,
                 metric: MetricOrName = accuracy_score,
                 train_offset: OffsetOrRange = 0,
                 train_test_delta: OffsetOrRange = 0,
                 train_examples: IntOrRange = 10,
                 test_examples: IntOrRange = 10,
                 feature_set: Optional[Dict[str, Dict[str, Any]]] = None,
                 classifier: Optional[Dict[str, Dict[str, Any]]] = None):
        self.__websites = EvaluationConfig.parse_int_or_range(websites, 'websites', 2)
        self.__runs = runs

        self.__metric = resolve_metric(metric)

        self.__train_offset = EvaluationConfig.parse_offset_or_range(train_offset, "train_offset")
        self.__train_test_delta = EvaluationConfig.parse_offset_or_range(train_test_delta, "train_test_delta")

        self.__train_examples = EvaluationConfig.parse_int_or_range(train_examples, "train_examples", 1)
        self.__test_examples = EvaluationConfig.parse_int_or_range(test_examples, "test_examples", 1)

        self.__feature_set = feature_set
        self.__classifier = classifier

    def __len__(self) -> int:
        return self.__runs * len(self.__websites) * len(self.__train_offset) * len(self.__train_test_delta) * len(
            self.__train_examples) * len(self.__test_examples)

    def __iter__(self) -> Iterator[EvaluationParams]:
        for w in self.__websites:
            for tr_off in self.__train_offset:
                for delta in self.__train_test_delta:
                    for tr_ex in self.__train_examples:
                        for te_ex in self.__test_examples:
                            yield EvaluationParams(w, self.__runs, self.__metric, tr_off, delta, tr_ex, te_ex,
                                                   self.__feature_set, self.__classifier)

    @staticmethod
    def parse_int_or_range(value: IntOrRange, name: str, min_value: int = 0) -> Union[range, List[int]]:
        if isinstance(value, list) or isinstance(value, range):
            if len(value) == 0:
                raise ValueError(f"At least one value is required for {name}")
            if any(x < min_value for x in value):
                raise ValueError(f"Only integer values greater than or equal to {min_value} are allowed for {name}")

            return sorted(set(value)) if isinstance(value, list) else value
        elif isinstance(value, int):
            return EvaluationConfig.parse_int_or_range([value], name, min_value)
        else:
            value = re.sub(r'\s+', '', value)
            if ".." in value:
                # range syntax
                pattern = re.compile(r'^(?P<start>0|[1-9][0-9]*)?\.\.(?P<end>[1-9][0-9]*)(%(?P<step>[1-9][0-9]*))?$')
                match = pattern.fullmatch(value)

                if not match:
                    raise ValueError(f"Invalid value '{value}' for '{name}': Must be 'start..end' or 'start..end%step'")

                start = int(match.group("start") or min_value)
                end = int(match.group("end"))
                step = int(match.group("step") or "1")

                if step <= 0:
                    raise ValueError(f"The range step size must be positive for {name}")

                return EvaluationConfig.parse_int_or_range(range(start, end, step), name, min_value)
            elif "," in value:
                # comma-separated list syntax
                return EvaluationConfig.parse_int_or_range([int(x.strip()) for x in value.split(",")], name, min_value)
            else:
                # int as str
                return EvaluationConfig.parse_int_or_range([int(value)], name, min_value)

    @staticmethod
    def parse_offset_or_range(value: OffsetOrRange, name: str) -> Union[range, List[OffsetOrDelta]]:
        if isinstance(value, range):
            if len(value) == 0:
                raise ValueError(f"At least one value is required for {name}")
            if min(value) < 0:
                raise ValueError(f"Only non-negative integers are allowed for {name}")

            return value
        elif isinstance(value, list):
            result: List[OffsetOrDelta] = []

            for elem in value:
                if isinstance(elem, str):
                    if elem.startswith("P"):
                        elem = pendulum.parse(elem)
                    else:
                        result += list(EvaluationConfig.parse_int_or_range(elem, name))
                        continue

                if isinstance(elem, int):
                    if elem < 0:
                        raise ValueError(f"Only non-negative integers are allowed for {name}")
                    result.append(elem)
                elif isinstance(elem, timedelta):
                    if elem < duration():
                        raise ValueError(f"Only non-negative deltas are allowed for {name}")
                    result.append(elem)

            return result
        elif isinstance(value, int):
            # noinspection Mypy
            return EvaluationConfig.parse_int_or_range(value, name)
        elif isinstance(value, timedelta):
            if value < duration():
                raise ValueError(f"Only non-negative deltas are allowed for {name}")
            return [value]
        else:
            if value.startswith("P"):
                return EvaluationConfig.parse_offset_or_range(pendulum.parse(value), name)
            else:
                # noinspection Mypy
                return EvaluationConfig.parse_int_or_range(value, name)
