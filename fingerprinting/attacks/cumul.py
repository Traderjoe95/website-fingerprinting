from typing import List, Type

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from ..api import AttackDefinition, FeatureSet
from ..api.typing import Classifier
from ..features import CUMULFeatures


def plugin_path() -> str:
    return 'core.attacks.cumul'


def attacks() -> List[Type[AttackDefinition]]:
    return [CUMUL]


class CUMUL(AttackDefinition):
    def create_feature_set(self) -> FeatureSet:
        return CUMULFeatures() / MinMaxScaler(feature_range=(-1, 1))

    def create_classifier(self) -> Classifier:
        param_grid = {
            'C': [2**11, 2**13, 2**15, 2**17],
            'gamma': [2**-3, 2**-1, 2**1, 2**3],
            'kernel': ['rbf']
        }

        return GridSearchCV(SVC(), param_grid, cv=10, refit=True)

    @staticmethod
    def name() -> str:
        return "CUMUL"
