import random
from abc import ABCMeta, abstractmethod
from datetime import timedelta
from functools import partial
from logging import getLogger
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import AsyncIterable, AsyncGenerator, Tuple, Optional, Iterable, Union, Set, TypeVar, Generic, Dict, Any, \
    List, Type

import curio
import numpy as np
import pandas as pd
from asyncstdlib import tee
from sklearn.metrics import accuracy_score  #

from .dataset import DatasetConfig
from .evaluation import EvaluationConfig
from .typing import OffsetOrDelta, SiteSelection, LabelledExamples, Transformer, TrainTestSplit, TracesStream, \
    HasParams, Metric, Classifier, StreamableClassifier, ExampleStream, LabelledExampleStream, Traces
from ..util.logging import configure_worker, get_queue
from ..util.multiprocessing import WorkerAdapter, Manager
from ..util.pipeline import drop_labels, collect, collect_examples, concat_sparse_aware
from ..util.range import intersect as range_intersect

T = TypeVar('T')

_logger = getLogger(__name__)


class Fitted(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    async def _do_fit(self, stream: AsyncIterable[T]) -> 'Fitted[T]':
        ...

    async def fit(self, stream: AsyncIterable[T]) -> 'Fitted[T]':
        self.reset()
        return await self._do_fit(stream)

    @abstractmethod
    def reset(self):
        ...


class Named(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class StaticNamed(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def name() -> str:
        ...


# noinspection PyUnusedLocal
class Dataset(StaticNamed, metaclass=ABCMeta):
    def __init__(self, *, sites: int, traces_per_site: int, config: DatasetConfig):
        if sites < 1:
            raise ValueError("There must be at least one site in any dataset")

        if traces_per_site < 1:
            raise ValueError("Each site must at least have one trace")

        self.__sites = sites
        self.__traces_per_site = traces_per_site
        self.__config = config

    async def load_all(self, *, sites: SiteSelection = None) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Loads all traces from the dataset.
        :param sites: The IDs of the websites to load traces for. If omitted, traces will be loaded for all websites.
        :return: An async iterable, see Dataset.load for details
        """
        async for df in self.load(sites=sites):
            yield df

    def load_train_test(
            self,
            *,
            train_examples_per_site: int = 1,
            train_offset: OffsetOrDelta = 0,
            test_examples_per_site: int = 1,
            train_test_delta: OffsetOrDelta = 0,
            sites: SiteSelection = None
    ) -> Tuple[AsyncGenerator[pd.DataFrame, None], AsyncGenerator[pd.DataFrame, None]]:
        """
        Loads disjoint training and test data from the data set.

        :param train_examples_per_site: the number of training examples to return per site
        :param train_offset: the offset of the training exeamples into the dataset.
            Can be a non-negative integer of a timedelta, taken from the time of the first trace collection.

            If a timedelta is used, the first traces will have been collected after or at that time.
        :param test_examples_per_site: the number of test examples to return per site (default: 1)
        :param train_test_delta: the offset between the end of training and the start of test examples.
            Can be a non-negative integer of a timedelta, taken from the last training example.

            If a timedelta is used, the first test example will have been collected at least that long after the
            last training example.
        :param sites: The IDs of the websites to load traces for. If omitted, traces will be loaded for all websites.
        :return: A tuple of async iterables (train, test). See Dataset.load for details
        """
        test_offset = train_examples_per_site

        if isinstance(train_offset, timedelta):
            test_offset += self._delta_to_offset(train_offset)
        else:
            test_offset += train_offset

        if isinstance(train_test_delta, timedelta):
            test_offset += self._delta_to_offset(train_test_delta, base=train_examples_per_site)
        else:
            test_offset += train_test_delta

        train = self.load(offset=train_offset, examples_per_site=train_examples_per_site,sites=sites)
        test = self.load(offset=test_offset, examples_per_site=test_examples_per_site, sites=sites)

        return train, test

    @abstractmethod
    async def load(self,
                   *,
                   offset: OffsetOrDelta = 0,
                   examples_per_site: Optional[int] = None,
                   sites: SiteSelection = None) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Loads data from the dataset.

        This method loads data beginning at or after the specified offset and produces exactly the requested number
        of examples per site.

        This method is expected to throw if the offset is greater than the maximum offset in the dataset or if there
        aren't enough examples per website (after the offset) to satisfy the requested count.

        :param offset: the offset into the dataset.
            Can be a non-negative integer of a timedelta, taken from the time of the first trace collection.

            If a timedelta is used, the first traces will have been collected after or at that time.
        :param examples_per_site: the number of examples to return per site
        :param sites: The IDs of the websites to load traces for. If omitted, traces will be loaded for all websites.

        :return: An async iterable of data frames. There is no assumption made about how the chunks of data are
            separated, i.e. there can be a data frame for every trace, a data frame for every site or any other
            subdivision of data.

            Each data frame contains packet observations within the rows and MUST contain the following columns:
             - site_id: The ID of the website the trace belongs to. Must be a zero-based integer.
             - trace_id: The trace ID, each trace within the same site must get its unique ID
             - collection_time: The time at which the trace collection started, must be the same for all packets
                                within the same trace
             - time: The timing of the packet within the trace. The first packet MUST have this set to 0
             - size: The size of the packet, including the direction. negative means "upstream", positive means
                     "downstream"
        """
        yield

    @abstractmethod
    def _delta_to_offset(self, delta: timedelta, base: int = 0) -> int:
        ...

    @property
    def sites(self):
        return self.__sites

    @property
    def traces_per_site(self) -> int:
        return self.__traces_per_site

    def __rshift__(self, attack: Union[Type['AttackDefinition'], 'AttackDefinition']) -> 'EvaluationPipeline':
        return _evaluation_pipeline(attack, self)

    def __gt__(self, attack: Union[Type['AttackDefinition'], 'AttackDefinition']) -> 'EvaluationPipeline':
        return self >> attack

    def __or__(self, attack: Union[Type['AttackDefinition'], 'AttackDefinition']) -> 'EvaluationPipeline':
        return self >> attack

    def labels(self, sites: SiteSelection) -> pd.Series:
        _sites = self._check_sites(sites)

        if _sites is None:
            return pd.Series(range(self.__sites))
        elif isinstance(_sites, int):
            return pd.Series(range(_sites))
        elif isinstance(_sites, range):
            return pd.Series(_sites)
        elif isinstance(_sites, set):
            return pd.Series(sorted(_sites))

    def _check_offset_and_count(self, offset: OffsetOrDelta, count: Optional[int]) -> Tuple[int, int]:
        offset = offset if isinstance(offset, int) else self._delta_to_offset(offset)

        if offset < 0:
            raise ValueError(f"The offset inside the dataset must not be negative, but was {offset}")
        if count is not None and count < 1:
            raise ValueError(f"The trace count must be positive, but was {count}")

        traces = self.__traces_per_site
        if offset >= traces or (count is not None and offset + count > traces):
            raise IndexError("There are only 50 traces for each website. offset must be in "
                             "[0, 49) and count must be in [1, 50 - offset]")

        if count is None:
            count = traces - offset

        return offset, count

    def _check_sites(self, sites: SiteSelection) -> Union[None, int, range, Set[int]]:
        sites = sites or self.__config.sites

        zero = False
        out_of_bounds = None

        if isinstance(sites, int):
            if sites == 0:
                zero = True
            if sites > self.__sites or sites < 0:
                out_of_bounds = sites
        elif isinstance(sites, range):
            if sites.step < 0:
                raise ValueError("Negative step sizes aren't allowed")

            if len(sites) == 0:
                zero = True

            if sites.start < 0:
                out_of_bounds = sites.start
            elif sites.stop > self.__sites:
                out_of_bounds = sites.stop
        elif isinstance(sites, Iterable):
            sites = set(sites)

            if len(sites) == 0:
                zero = True

            if any(map(lambda s: s < 0 or s >= self.__sites, sites)):
                mx, mn = max(self.__sites - 1, *sites), min(0, *sites)

                if mx >= self.__sites:
                    out_of_bounds = mx
                elif mn < 0:
                    out_of_bounds = mn

        if zero:
            raise ValueError("Can't load zero sites")
        if out_of_bounds is not None:
            raise IndexError(f"Site index {out_of_bounds} is out of bounds for [0, {self.__sites})")

        return sites

    @staticmethod
    def _intersect(sites: SiteSelection, store_range: range) -> Union[bool, range, Set[int]]:
        if sites is None:
            return True
        elif isinstance(sites, int):
            if sites > store_range.start:
                return range(store_range.start, min(sites, store_range.stop))
        elif isinstance(sites, range):
            if store_range.start < sites.stop and store_range.stop > sites.start:
                if store_range.step == sites.step == 1:
                    return range(max(sites.start, store_range.start), min(sites.stop, store_range.stop))
                else:
                    r_intersection = range_intersect(sites, store_range)

                    if r_intersection is not None:
                        return r_intersection
        else:
            intersection = set(sites).intersection(store_range)

            if len(intersection) > 0:
                return intersection

        return False


class Defense(StaticNamed, Fitted[pd.DataFrame], HasParams, metaclass=ABCMeta):
    """
    The base class of any defense.

    It takes data loaded from a `fingerprinting.api.Dataset` and alters the traces to thwart identification of the
    websites.
    """

    # noinspection PyTypeChecker,Mypy
    async def fit(self, stream: AsyncIterable[T]) -> 'Defense':
        return await super().fit(stream)

    @abstractmethod
    async def _do_fit(self, traces_stream: TracesStream) -> 'Defense':
        """
        Fits the defense to the given input traces.

        :param traces_stream: The stream of data frames containing packet traces.

            These data frames are expected to be structured as explained on the Dataset.load method.
        :return: this instance for chaining
        """
        ...

    @abstractmethod
    async def defend(self, traces: pd.DataFrame) -> pd.DataFrame:
        """
        Defends traces in the input data frame and returns them as another data frame.

        :param traces: The data frame containing packet traces.

            This data frame is expected to be structured as explained on the Dataset.load method.
        :return: the data frame containing the defended traces

            This data frame is expected to be structured as explained on the Dataset.load method.
         """
        ...

    async def defend_all(self, traces_stream: TracesStream) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Defends traces in the input data frames and returns them as other data frames

        :param traces_stream: An async iterable of the data frame containing packet traces.

            These data frames are expected to be structured as explained on the Dataset.load method.
        :return: An async iterable of the data frames containing the defended traces.

            These data frames are expected to be structured as explained on the Dataset.load method.
        """
        async for traces in traces_stream:
            yield await self.defend(traces)

    async def fit_defend(self, traces_stream: TracesStream) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Fits the defense to the input traces and defends them in one step.

        :param traces_stream: The stream of data frames containing packet traces.

            These data frames are expected to be structured as explained on the Dataset.load method.
        :return: An async iterable of the data frames containing the defended traces.

            These data frames are expected to be structured as explained on the Dataset.load method.
        """
        (left, right) = tee(traces_stream, n=2)

        async for defended in (await self.fit(left)).defend_all(right):
            yield defended


class FeatureSet(HasParams, metaclass=ABCMeta):
    """
    The base class of any feature set.

    It takes data loaded from a `fingerprinting.api.Dataset` and extracts features for a subsequent ML algorithm.
    """
    async def extract_features(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        self.reset()
        return await self._do_extract(train_traces, test_traces)

    @abstractmethod
    async def _do_extract(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        ...

    def __add__(self, other: 'FeatureSet') -> 'FeatureSet':
        return _combine_feature_sets(self, other)

    def __truediv__(self, transformer: Transformer) -> 'FeatureSet':
        return _transform_feature_set(self, transformer)

    @abstractmethod
    def reset(self):
        ...


class AttackInstance:
    def __init__(self, features: FeatureSet, classifier: Classifier, classes: Optional[np.ndarray] = None):
        if classes is not None and (len(classes.shape) != 1 or np.unique(classes).size < 2):
            raise ValueError("The classes must either be None or a one-dimensional array with at "
                             "least two unique entries")

        self.__fts = features
        self.__clf = classifier
        self.__classes = classes

        self.__first = True
        self.__fitted = False

    async def extract_features(self, train_traces: TracesStream, test_traces: TracesStream) -> TrainTestSplit:
        return await self.__fts.extract_features(train_traces, test_traces)

    # noinspection PyPep8Naming
    async def fit(self, train: AsyncIterable[LabelledExamples]):
        if self.__classes is not None and isinstance(self.__clf, StreamableClassifier):
            async for X, y in train:
                if self.__first:
                    self.__clf.partial_fit(X, y, classes=self.__classes)
                    self.__first = False
                else:
                    self.__clf.partial_fit(X, y)
        else:
            X, y = await collect(train)
            self.__clf.fit(X, y)

        self.__fitted = True

    # noinspection PyPep8Naming
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self.__clf.predict(X)

    # noinspection PyPep8Naming
    async def predict_stream(self, X_stream: ExampleStream, *, collect_before_predict: bool = False) -> np.ndarray:
        self._ensure_fitted()

        if collect_before_predict:
            X = await collect_examples(X_stream)
            return self.predict(X)
        else:
            y_ = [self.predict(X) async for X in X_stream]
            return concat_sparse_aware(y_)

    # noinspection PyPep8Naming
    def score(self, X: np.ndarray, y: np.ndarray, *, metric: Metric = accuracy_score) -> float:
        self._ensure_fitted()
        return metric(y, self.__clf.predict(X))

    # noinspection PyPep8Naming
    async def score_stream(self,
                           stream: LabelledExampleStream,
                           *,
                           collect_before_predict: bool = False,
                           metric: Metric = accuracy_score) -> float:
        self._ensure_fitted()

        if collect_before_predict:
            X, y = await collect(stream)
            return self.score(X, y, metric=metric)
        else:
            example_count = 0
            score_ = 0.

            async for X, y in stream:
                chunk_score = self.score(X, y, metric=metric)
                score_ += chunk_score * X.shape[0]
                example_count += X.shape[0]

            return score_ / example_count

    # noinspection PyPep8Naming
    async def fit_predict(self,
                          train_traces: TracesStream,
                          test_traces: TracesStream,
                          *,
                          collect_before_predict: bool = False) -> np.ndarray:
        train, test = await self.extract_features(train_traces, test_traces)
        await self.fit(train)

        return await self.predict_stream(drop_labels(test), collect_before_predict=collect_before_predict)

    # noinspection PyPep8Naming
    async def fit_score(self,
                        train_traces: TracesStream,
                        test_traces: TracesStream,
                        *,
                        collect_before_predict: bool = False,
                        metric: Metric = accuracy_score) -> float:
        train, test = await self.extract_features(train_traces, test_traces)
        await self.fit(train)

        return await self.score_stream(test, collect_before_predict=collect_before_predict, metric=metric)

    def _ensure_fitted(self):
        if not self.__fitted:
            raise AssertionError("The attack must be fitted first")


class AttackDefinition(StaticNamed, metaclass=ABCMeta):
    def instantiate(self,
                    *,
                    classes: Optional[np.ndarray] = None,
                    featureset_params: Optional[Dict[str, Any]] = None,
                    classifier_params: Optional[Dict[str, Any]] = None) -> 'AttackInstance':
        fts = self._create_featureset()
        clf = self._create_classifier()

        fts.set_params(**(featureset_params or {}))
        clf.set_params(**(classifier_params or {}))

        return AttackInstance(fts, clf, classes)

    @abstractmethod
    def _create_featureset(self) -> FeatureSet:
        """
        Creates a default instance of the attack's feature set.

        :return: a `fingerprinting.api.FeatureSet` instance with default parameters
        """
        ...

    @abstractmethod
    def _create_classifier(self) -> Classifier:
        """
        Creates a default instance of the attack's classifier.

        :return: a classifier instance with default parameters
        """
        ...


Dataset_ = Union[Type[Dataset], Dataset]
EvaluationDataset = Union[Dataset_, Tuple[str, Dataset_]]
Datasets = Union[EvaluationDataset, List[EvaluationDataset], Dict[str, Dataset_]]

AttackDefinition_ = Union[Type[AttackDefinition], AttackDefinition]
EvaluationAttack = Union[AttackDefinition_, Tuple[str, AttackDefinition_]]
Attacks = Union[EvaluationAttack, List[EvaluationAttack], Dict[str, AttackDefinition_]]

Defense_ = Optional[Union[Defense, Type[Defense]]]
EvaluationDefense = Union[Defense_, Tuple[str, Defense_]]
Defenses = Union[None, EvaluationDefense, List[EvaluationDefense], Dict[str, Defense_]]


class EvaluationPipeline:
    def __init__(self, datasets: Datasets, attacks: Attacks, *, defenses: Defenses = None):
        if (isinstance(datasets, list) or isinstance(datasets, dict)) and len(datasets) == 0:
            raise ValueError("At least one dataset is required for the evaluation pipeline")
        if (isinstance(attacks, list) or isinstance(attacks, dict)) and len(attacks) == 0:
            raise ValueError("At least one attack is required for the evaluation pipeline")

        self.__datasets: Dict[str, Dataset] = {}
        if isinstance(datasets, Dataset) or isinstance(datasets, type) or isinstance(datasets, tuple):
            self.add_dataset(datasets)
        elif isinstance(datasets, list):
            for ds in datasets:
                self.add_dataset(ds)
        else:
            for name in datasets:
                ds = datasets[name]
                self.add_dataset((name, ds))

        self.__attacks: Dict[str, AttackDefinition] = {}
        if isinstance(attacks, AttackDefinition) or isinstance(attacks, type) or isinstance(attacks, tuple):
            self.add_attack(attacks)
        elif isinstance(attacks, list):
            for att in attacks:
                self.add_attack(att)
        else:
            for name in attacks:
                att = attacks[name]
                self.add_attack((name, att))

        self.__defenses: Dict[str, Optional[Defense]] = {}
        if isinstance(defenses, Defense) or isinstance(defenses, type) or isinstance(defenses, tuple):
            self.add_defense(defenses)
        elif isinstance(defenses, list):
            for defense in defenses:
                self.add_defense(defense)

            if len(defenses) == 0:
                self.__defenses["none"] = None
        elif defenses is not None:
            for name in defenses:
                defense = defenses[name]
                self.add_defense((name, defense))
        else:
            self.__defenses["none"] = None

    def add_dataset(self, dataset: EvaluationDataset) -> 'EvaluationPipeline':
        if isinstance(dataset, Dataset):
            self.__datasets[dataset.name()] = dataset
        elif isinstance(dataset, type):
            return self.add_dataset(dataset())
        else:
            name, dataset = dataset

            if isinstance(dataset, Dataset):
                self.__datasets[name] = dataset
            else:
                return self.add_dataset((name, dataset()))

        return self

    def add_attack(self, attack: EvaluationAttack) -> 'EvaluationPipeline':
        if isinstance(attack, AttackDefinition):
            self.__attacks[attack.name()] = attack
        elif isinstance(attack, type):
            return self.add_attack(attack())
        else:
            name, attack = attack

            if isinstance(attack, AttackDefinition):
                self.__attacks[name] = attack
            else:
                return self.add_attack((name, attack()))

        return self

    def add_defense(self, defense: EvaluationDefense) -> 'EvaluationPipeline':
        if isinstance(defense, Defense):
            self.__defenses[defense.name()] = defense
        elif isinstance(defense, type):
            return self.add_defense(defense())
        elif defense is not None:
            name, defense = defense

            if defense is not None and isinstance(defense, Defense):
                self.__defenses[name] = defense
            elif defense is not None:
                return self.add_defense((name, defense()))
            else:
                self.__defenses[name] = None
        else:
            self.__defenses["none"] = None

        return self

    @staticmethod
    def _score_worker(attack_def: AttackDefinition,
                      classes: Optional[np.ndarray],
                      featureset_params: Optional[Dict[str, Any]],
                      classifier_params: Optional[Dict[str, Any]],
                      dataset_conns: Tuple[Connection, Connection],
                      queue: Queue = None,
                      log_level: Optional[int] = None,
                      *,
                      concat: bool = False,
                      metric: Metric = accuracy_score) -> float:
        # noinspection Mypy
        configure_worker(queue, log_level)
        train_conn, test_conn = dataset_conns

        train: WorkerAdapter[Traces] = WorkerAdapter(train_conn)
        test: WorkerAdapter[Traces] = WorkerAdapter(test_conn)

        attack = attack_def.instantiate(classes=classes,
                                        featureset_params=featureset_params,
                                        classifier_params=classifier_params)
        fit_score = partial(attack.fit_score, collect_before_predict=concat, metric=metric)

        return curio.run(fit_score, train, test)

    async def __run_evaluation(self, config: EvaluationConfig, site_ids: Optional[Set[int]]) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        async with curio.TaskGroup() as g:
            for dataset_name, dataset in self.__datasets.items():
                for params in config:
                    runner = partial(self._score_worker, metric=params.metric)

                    for run in range(params.runs):
                        sites = random.sample(site_ids or range(dataset.sites), k=params.websites)
                        classes = dataset.labels(sites).values

                        train, test = dataset.load_train_test(train_examples_per_site=params.train_examples,
                                                              test_examples_per_site=params.test_examples,
                                                              train_offset=params.train_offset,
                                                              train_test_delta=params.train_test_delta,
                                                              sites=sites)
                        trains = tee(train, n=len(self.__defenses))
                        tests = tee(test, n=len(self.__defenses))

                        tasks: Dict[str, curio.Task] = {}

                        for defense_idx, (defense_name, defense) in enumerate(self.__defenses.items()):
                            train = trains[defense_idx]
                            test = tests[defense_idx]

                            if defense is not None:
                                defense_params = params.defense_params(defense_name)
                                if defense_params is not None:
                                    defense.set_params(**defense_params)

                                train = defense.fit_defend(train)
                                test = defense.defend_all(test)

                            train_manager: Manager[Traces] = Manager(train, len(self.__attacks))
                            test_manager: Manager[Traces] = Manager(test, len(self.__attacks))

                            await g.spawn(train_manager.run)
                            await g.spawn(test_manager.run)

                            for attack_idx, (attack_name, attack) in enumerate(self.__attacks.items()):
                                tasks[f'{dataset_name}/{defense_name}/{attack_name}/{params.websites}'] = await g.spawn(
                                    curio.run_in_process, runner, attack, classes,
                                    params.feature_set_params(attack_name), params.classifier_params(attack_name),
                                    (train_manager.worker_conns[attack_idx], test_manager.worker_conns[attack_idx]),
                                    get_queue())

                        for key in tasks:
                            score = await tasks[key].join()

                            [ds_name, def_name, att_name, s] = key.split("/")

                            results.append({
                                "dataset": ds_name,
                                "defense": def_name,
                                "attack": att_name,
                                "sites": int(s),
                                "train_offset": params.train_offset,
                                "train_test_delta": params.train_test_delta,
                                "train_examples": params.train_examples,
                                "test_examples": params.test_examples,
                                "score": score
                            })
                            _logger.info(f"Finished evaluation '{key}' with a score of {score}")

        result_df = pd.DataFrame(results)
        result_df['dataset'] = pd.Series(
            pd.Categorical(result_df['dataset'], categories=np.unique(result_df['dataset']), ordered=False))
        result_df['defense'] = pd.Series(
            pd.Categorical(result_df['defense'], categories=np.unique(result_df['defense']), ordered=False))
        result_df['attack'] = pd.Series(
            pd.Categorical(result_df['attack'], categories=np.unique(result_df['attack']), ordered=False))

        return result_df

    def run_evaluation(self, config: EvaluationConfig, site_ids: Optional[Set[int]] = None) -> pd.DataFrame:
        return curio.run(self.__run_evaluation, config, site_ids)


def _evaluation_pipeline(attack: Union[Type[AttackDefinition], AttackDefinition], data: Dataset) -> EvaluationPipeline:
    if isinstance(attack, type):
        return EvaluationPipeline(data, attack())
    else:
        return EvaluationPipeline(data, attack)


def _combine_feature_sets(left: FeatureSet, right: FeatureSet) -> FeatureSet:
    from .feature_set import StatelessFeatureSet, StatelessCombinedFeatureSet, CombinedFeatureSet

    if isinstance(left, StatelessFeatureSet) and isinstance(right, StatelessFeatureSet):
        return StatelessCombinedFeatureSet(left, right)
    else:
        return CombinedFeatureSet(left, right)


def _transform_feature_set(features: FeatureSet, transformer: Transformer):
    from .feature_set import TransformedFeatureSet

    return TransformedFeatureSet(features, transformer)
