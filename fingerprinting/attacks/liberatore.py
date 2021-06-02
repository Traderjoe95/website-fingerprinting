from ..algorithms.jaccard import JaccardClassifier
from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketSet, PacketHistogram


class Jaccard(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketSet()

    def _create_classifier(self) -> Classifier:
        return JaccardClassifier(alpha=0.1)

    @property
    def name(self) -> str:
        return "jaccard"


class LiberatoreNB(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketHistogram()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde='1d', bandwidth='silverman')

    @property
    def name(self) -> str:
        return 'liberatore'
