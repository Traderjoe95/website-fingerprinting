from typing import List, Type

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketHistogram


def plugin_path() -> str:
    return 'core.attacks.herrmann'


def attacks() -> List[Type[AttackDefinition]]:
    return [Herrmann]


class Herrmann(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return PacketHistogram() / TfidfTransformer(use_idf=True, sublinear_tf=True, norm='l2')

    def create_classifier(self) -> Classifier:
        return MultinomialNB()

    @staticmethod
    def name() -> str:
        return "herrmann"
