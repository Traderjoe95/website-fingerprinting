import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import PacketHistogram


class Herrmann(AttackDefinition):
    def _create_featureset(self) -> FeatureSet:
        return PacketHistogram() / TfidfTransformer(use_idf=True, sublinear_tf=True, norm='l2')

    def _create_classifier(self) -> Classifier:
        return MultinomialNB()

    @property
    def name(self) -> str:
        return "herrmann"
