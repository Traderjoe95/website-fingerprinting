from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import Time as TimeFeatures, Bandwidth as BandwidthFeatures, VariableNGram


class Time(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return TimeFeatures()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth=0.05)

    @property
    def name(self) -> str:
        return "time"


class Bandwidth(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return BandwidthFeatures()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth=0.05)

    @property
    def name(self) -> str:
        return "bandwidth"


class VNG(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return VariableNGram()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth=0.05)

    @property
    def name(self) -> str:
        return "vng"


class VNGPlusPlus(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return TimeFeatures() + BandwidthFeatures() + VariableNGram()

    def _create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth=0.05)

    @property
    def name(self) -> str:
        return "vng++"
