import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin

# noinspection PyPep8Naming
from ..api.typing import StreamableClassifier
from ..util.pipeline import is_sparse_matrix


# noinspection PyPep8Naming
class JaccardClassifier(StreamableClassifier, BaseEstimator, ClassifierMixin):
    def __init__(self, *, alpha: float = 0):
        self.alpha = alpha

        self.classes_: np.ndarray = np.array([])
        self.class_count_: np.ndarray = np.array([])

        self.model_: Optional[np.ndarray] = None

        self.__dirty = False
        self.__unit_model: Optional[np.ndarray] = None

    def fit(self, X, y, **fit_params) -> 'JaccardClassifier':
        self.model_ = None
        return self.partial_fit(X, y, np.sort(np.unique(y)))

    def partial_fit(self, X, y, classes=None) -> 'JaccardClassifier':
        if self.model_ is None and classes is None:
            raise ValueError("You need to specify all possible classes on the first call to partial_fit")
        if self.model_ is None:
            self.model_ = np.zeros((classes.shape[0], X.shape[1]))

            self.classes_ = classes
            self.class_count_ = np.zeros(classes.shape[0])

        if X.shape[1] != self.model_.shape[1]:
            raise ValueError(f"Wrong number of features, expected {self.model_.shape[1]}, but got {X.shape[1]}. "
                             f"Did you mean to call clf.fit(X, y)?")
        if np.setdiff1d(y, self.classes_).size > 0:
            unknown_classes = np.setdiff1d(y, self.classes_)
            warnings.warn(f"Encountered unknown classes {unknown_classes}", RuntimeWarning)

        classes = np.expand_dims(self.classes_, axis=0)
        X.data = np.clip(X.data, 0, 1)

        self.model_ += np.apply_along_axis(lambda c: np.sum(X[y == c], axis=0), 0, classes).T
        self.class_count_ += np.apply_along_axis(lambda c: np.count_nonzero(y == c), 0, classes)
        self.__dirty = True

        return self

    def calculate_jaccard(self, X):
        if self.__dirty:
            over_treshold = (self.model_ >= np.tile(self.class_count_, (self.model_.shape[1], 1)).T / 2)
            self.__unit_model = sp.csr_matrix(over_treshold.astype(np.uint8))
            self.__dirty = False

        X_sum = np.matrix(np.sum(X, axis=1))
        class_sum = np.matrix(np.sum(self.__unit_model, axis=1)).T

        X_counts = np.tile(X_sum, (1, self.classes_.shape[0]))
        class_counts = np.tile(class_sum, (X.shape[0], 1))

        intersection = (X * self.__unit_model.T).astype(np.float64)
        union = (X_counts + class_counts - intersection).astype(np.float64)

        if is_sparse_matrix(intersection):
            intersection = intersection.todense()

        if is_sparse_matrix(union):
            union = union.todense()

        jaccard = (intersection + self.alpha) / (union + self.alpha)

        return jaccard

    def predict_proba(self, X):
        jaccard = self.calculate_jaccard(X)
        return jaccard / np.tile(np.sum(jaccard, axis=1), (self.classes_.shape[0], 1)).T

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.calculate_jaccard(X), axis=1)]

    def get_params(self):
        return {'alpha': self.alpha}

    def set_params(self, **params):
        if 'alpha' in params:
            self.alpha = params['alpha']

    @property
    def alpha(self) -> float:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float):
        if alpha < 0:
            raise ValueError("alpha must not be negative")

        self.__alpha = alpha
