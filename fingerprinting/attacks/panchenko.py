from sklearn.svm import SVC

from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketHistogram, PanchenkoMarkers


class Panchenko(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketHistogram() + PanchenkoMarkers()

    def _create_classifier(self) -> Classifier:
        return SVC(C=131072, kernel="rbf", gamma=1.9073486328125e-06)

    @property
    def name(self) -> str:
        return "panchenko"


class PanchenkoNB(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketHistogram() + PanchenkoMarkers()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth=0.05)

    @property
    def name(self) -> str:
        return "panchenko-nb"
