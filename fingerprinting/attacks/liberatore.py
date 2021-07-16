from typing import List, Type

from ..algorithms.jaccard import JaccardClassifier
from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketSet, PacketHistogram


def plugin_path() -> str:
    return 'core.attacks.liberatore'


def attacks() -> List[Type[AttackDefinition]]:
    return [Jaccard, LiberatoreNB]


class Jaccard(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketSet()

    def _create_classifier(self) -> Classifier:
        return JaccardClassifier(alpha=0.1)

    @staticmethod
    def name() -> str:
        return "jaccard"


class LiberatoreNB(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketHistogram()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde='1d', bandwidth='weka')

    @staticmethod
    def name() -> str:
        return 'liberatore'
