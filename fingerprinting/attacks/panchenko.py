from typing import List, Type

from sklearn.svm import SVC

from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketHistogram, PanchenkoMarkers


def plugin_path() -> str:
    return 'core.attacks.panchenko'


def attacks() -> List[Type[AttackDefinition]]:
    return [Panchenko, PanchenkoNB]


class Panchenko(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return PacketHistogram() + PanchenkoMarkers()

    def create_classifier(self) -> Classifier:
        return SVC(C=131072, kernel="rbf", gamma=1.9073486328125e-06)

    @staticmethod
    def name() -> str:
        return "panchenko"


class PanchenkoNB(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return PacketHistogram() + PanchenkoMarkers()

    def create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth='weka')

    @staticmethod
    def name() -> str:
        return "panchenko-nb"
