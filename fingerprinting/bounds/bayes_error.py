import random
from datetime import timedelta
from functools import partial
from logging import getLogger
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Tuple, Optional, Dict, Any, Union, Set, List

import curio
import numpy as np
import numpy.random
import pandas as pd
import pendulum
from asyncstdlib.itertools import tee
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from ..api import Dataset, AttackDefinition, Defense
from ..api.pipeline import Datasets, Attacks, Defenses, EvaluationDataset, EvaluationAttack, EvaluationDefense
from ..api.typing import LabelledExampleStream, HasParams, OffsetOrDelta, Traces
from ..util.logging import configure_worker, get_queue
from ..util.multiprocessing import WorkerAdapter, Manager, WorkerPool
from ..util.pipeline import collect, dense

_RNG = np.random.default_rng()

_logger = getLogger(__name__)


class BayesErrorEstimator(HasParams):
    def __init__(self, closed_world: bool = True):
        self.closed_world = closed_world

    def estimate(self, examples: LabelledExampleStream) -> Tuple[float, float]:
        _examples, _labels = collect(examples)

        if self.closed_world:
            classes = np.unique(_labels).shape[0]
            bayes_error_bound = BayesErrorEstimator.__estimate_error(_examples, _labels)
        else:
            site_error = []

            for site_id in np.unique(_labels):
                example_count = np.count_nonzero(_labels == site_id)
                non_monitored_indices = np.squeeze(np.argwhere(_labels != site_id))

                non_monitored = _RNG.choice(non_monitored_indices, size=example_count, replace=False)

                _site_examples = np.concatenate([dense(_examples[_labels == site_id]),
                                                 dense(_examples[non_monitored])])
                _site_labels = np.concatenate([dense(_labels[_labels == site_id]),
                                               np.broadcast_to(-1, non_monitored.shape)])

                site_error.append(BayesErrorEstimator.__estimate_error(_site_examples, _site_labels))

            classes = 2
            bayes_error_bound = np.average(np.array(site_error))

        # Divide the Bayes error by the random guessing error to get epsilon
        return bayes_error_bound, bayes_error_bound * classes / (classes - 1)

    @staticmethod
    def __estimate_error(examples: np.ndarray, labels: np.ndarray) -> float:
        r_nn = BayesErrorEstimator.__estimate_nn_error(examples, labels)
        classes = np.unique(labels).shape[0]

        scaled_r_nn = np.minimum(classes / (classes - 1) * r_nn, 1)

        return (classes - 1) / classes * (1 - np.sqrt(1 - scaled_r_nn))

    @staticmethod
    def __estimate_nn_error(examples: np.ndarray, labels: np.ndarray) -> float:
        return 1. - np.average(cross_val_score(KNeighborsClassifier(n_neighbors=1), examples, labels,
                                               scoring='accuracy', cv=5))

    @property
    def closed_world(self) -> bool:
        return self.__closed_world

    @closed_world.setter
    def closed_world(self, closed_world: bool):
        if not isinstance(closed_world, bool):
            raise TypeError("The closed_world parameter must be boolean")

        self.__closed_world = closed_world

    def get_params(self):
        return {'closed_world': self.closed_world}

    def set_params(self, **params):
        if 'closed_world' in params:
            self.closed_world = params['closed_world']


class ErrorBoundsPipeline:
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

    def add_dataset(self, dataset: EvaluationDataset) -> 'ErrorBoundsPipeline':
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

    def add_attack(self, attack: EvaluationAttack) -> 'ErrorBoundsPipeline':
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

    def add_defense(self, defense: EvaluationDefense) -> 'ErrorBoundsPipeline':
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
    def _estimate_worker(closed_world: bool,
                         attack_def: AttackDefinition,
                         featureset_params: Optional[Dict[str, Any]],
                         dataset_conn: Connection,
                         queue: Queue = None,
                         log_level: Optional[int] = None) -> Tuple[float, float]:
        # noinspection Mypy
        configure_worker(queue, log_level)

        data: WorkerAdapter[Traces] = WorkerAdapter(dataset_conn)

        feature_set = attack_def.create_feature_set()

        if featureset_params is not None:
            feature_set.set_params(**featureset_params)

        estimator = BayesErrorEstimator(closed_world=closed_world)

        return estimator.estimate(feature_set.extract_features_single(data))

    async def __run_evaluation(self, config: 'ErrorBoundsConfig', site_ids: Optional[Set[int]]) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        with WorkerPool() as pool:
            async with curio.TaskGroup() as g:
                for dataset_name, dataset in self.__datasets.items():
                    for run in range(config.runs):
                        sites = random.sample(site_ids or dataset.sites, k=config.websites)

                        data = dataset.load(offset=config.offset, examples_per_site=config.examples, sites=sites)
                        datas = tee(data, n=len(self.__defenses))

                        tasks: Dict[str, curio.Task] = {}

                        for defense_idx, (defense_name, defense) in enumerate(self.__defenses.items()):
                            data = datas[defense_idx]

                            if defense is not None:
                                defense_params = config.defense_params(defense_name)
                                if defense_params is not None:
                                    defense.set_params(**defense_params)

                                data = defense.fit_defend(data)

                            data_manager: Manager[Traces] = Manager(data, len(self.__attacks))

                            await g.spawn(data_manager.run)

                            for attack_idx, (attack_name, attack) in enumerate(self.__attacks.items()):
                                tasks[f'{dataset_name}/{defense_name}/{attack_name}'] = await g.spawn(
                                    pool.submit, partial(ErrorBoundsPipeline._estimate_worker, config.closed_world,
                                                         attack,
                                                         config.feature_set_params(attack_name),
                                                         data_manager.worker_conns[attack_idx],
                                                         get_queue()))

                        for key in tasks:
                            error_bound, epsilon = await tasks[key].join()

                            [ds_name, def_name, att_name] = key.split("/")

                            results.append({
                                "dataset": ds_name,
                                "defense": def_name,
                                "feature_set": att_name,
                                "error_bound": error_bound,
                                "epsilon": epsilon
                            })

                            _logger.info(f"Finished error bound estimation '{key}' with a result of {error_bound} "
                                         f"(epsilon = {epsilon})")

        result_df = pd.DataFrame(results)
        result_df['dataset'] = pd.Series(
            pd.Categorical(result_df['dataset'], categories=np.unique(result_df['dataset']), ordered=False))
        result_df['defense'] = pd.Series(
            pd.Categorical(result_df['defense'], categories=np.unique(result_df['defense']), ordered=False))
        result_df['feature_set'] = pd.Series(
            pd.Categorical(result_df['feature_set'], categories=np.unique(result_df['feature_set']), ordered=False))

        return result_df

    def run_evaluation(self, config: 'ErrorBoundsConfig', site_ids: Optional[Set[int]] = None) -> pd.DataFrame:
        return curio.run(self.__run_evaluation, config, site_ids)


OffsetOrDuration = Union[OffsetOrDelta, str]


class ErrorBoundsConfig:
    def __init__(self,
                 websites: int = 2,
                 examples: int = 205,
                 runs: int = 10,
                 offset: OffsetOrDuration = 0,
                 closed_world: bool = True,
                 defense: Optional[Dict[str, Dict[str, Any]]] = None,
                 feature_set: Optional[Dict[str, Dict[str, Any]]] = None):
        self.__websites = websites
        self.__examples = examples
        self.__runs = runs
        self.__offset = parse_offset_or_duration(offset, "offset")
        self.__closed_world = closed_world

        self.__defense = defense
        self.__feature_set = feature_set

    def __len__(self) -> int:
        return self.__runs

    @property
    def websites(self) -> int:
        return self.__websites

    @property
    def examples(self) -> int:
        return self.__examples

    @property
    def runs(self) -> int:
        return self.__runs

    @property
    def offset(self) -> OffsetOrDelta:
        return self.__offset

    @property
    def closed_world(self) -> bool:
        return self.__closed_world

    def defense_params(self, defense_name: str) -> Optional[Dict[str, Any]]:
        if self.__defense is None:
            return None

        return self.__defense[defense_name] if defense_name in self.__defense else None

    def feature_set_params(self, attack_name: str) -> Optional[Dict[str, Any]]:
        if self.__feature_set is None:
            return None

        return self.__feature_set[attack_name] if attack_name in self.__feature_set else None


def parse_offset_or_duration(value: OffsetOrDuration, name: str) -> OffsetOrDelta:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"'{name}' must not be negative")
        # noinspection Mypy
        return value
    elif isinstance(value, timedelta):
        if value < pendulum.duration():
            raise ValueError(f"Only non-negative deltas are allowed for '{name}'")
        return value
    elif isinstance(value, str):
        if value.startswith("P"):
            return parse_offset_or_duration(pendulum.parse(value), name)
    else:
        raise TypeError(f"Unsupported type for '{name}': {type(name)}")
