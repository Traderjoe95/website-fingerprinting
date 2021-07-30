from typing import List, Type

from ..algorithms.naive_bayes import KernelDensityNB
from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import Time as TimeFeatures, Bandwidth as BandwidthFeatures, VariableNGram


def plugin_path() -> str:
    return "core.attacks.dyer"


def attacks() -> List[Type[AttackDefinition]]:
    return [Time, Bandwidth, VNG, VNGPlusPlus]


class Time(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return TimeFeatures()

    def create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth='weka')

    @staticmethod
    def name() -> str:
        return "time"


class Bandwidth(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return BandwidthFeatures()

    def create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth='weka')

    @staticmethod
    def name() -> str:
        return "bandwidth"


class VNG(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return VariableNGram()

    def create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth='weka')

    @staticmethod
    def name() -> str:
        return "vng"


class VNGPlusPlus(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return TimeFeatures() + BandwidthFeatures() + VariableNGram()

    def create_classifier(self) -> Classifier:
        return KernelDensityNB(priors='uniform', kde="1d", bandwidth='weka')

    @staticmethod
    def name() -> str:
        return "vng++"
