from typing import Optional, Union

import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator, DensityMixin

from .bandwidth import scott, silverman, norm, sheather_jones, weka, weka_precision
from ..api.typing import HasParams
from ..util.pipeline import is_sparse_matrix

VALID_KERNELS = frozenset(['gaussian', 'tophat', 'epanechnikov', 'quartic', 'triweight', 'tricube', 'exponential',
                           'logistic', 'sigmoid', 'linear', 'cosine'])
VALID_BW_ESTIMATORS = frozenset(['scott', 'silverman', 'norm', 'sj', 'weka'])
SQRT_2PI = np.sqrt(2 * np.pi)


# noinspection PyPep8Naming
class KernelDensity1D(HasParams, BaseEstimator, DensityMixin):
    def __init__(self, bandwidth: Union[str, float, np.ndarray] = 1., kernel: str = 'gaussian',
                 min_bandwidth: Optional[float] = None):
        self.__model: Optional[np.ndarray] = None
        self.__dirty = False

        self.bandwidth_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None

        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth
        self.kernel = kernel

    @property
    def bandwidth(self):
        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth: Union[str, float, np.ndarray]):
        if isinstance(bandwidth, str) and bandwidth not in VALID_BW_ESTIMATORS:
            raise ValueError(f'Invalid bandwidth estimator: {bandwidth}, must be one '
                             f'of {", ".join(VALID_BW_ESTIMATORS)}')
        elif isinstance(bandwidth, np.ndarray) and self.__model is not None:
            if bandwidth.shape != (self.__model.shape[1],):
                raise ValueError(f"Invalid bandwidth shape, expected {(self.__model.shape[1],)}")
        elif isinstance(bandwidth, float) and bandwidth <= 0.:
            raise ValueError("The bandwidth must be positive")

        self.__bandwidth = bandwidth

    @property
    def kernel(self):
        return self.__kernel

    @kernel.setter
    def kernel(self, kernel: str):
        kernel = kernel.lower()
        if kernel not in VALID_KERNELS:
            raise ValueError(f'Invalid kernel: {kernel}, must be one of {", ".join(VALID_KERNELS)}')

        self.__kernel = kernel
        if kernel == 'gaussian':
            self.__kernel_cdf = _gaussian_kernel_cdf
        elif kernel == 'tophat':
            self.__kernel_cdf = _tophat_kernel_cdf
        elif kernel == 'epanechnikov':
            self.__kernel_cdf = _epanechnikov_kernel_cdf
        elif kernel == 'quartic':
            self.__kernel_cdf = _quartic_kernel_cdf
        elif kernel == 'triweight':
            self.__kernel_cdf = _triweight_kernel_cdf
        elif kernel == 'tricube':
            self.__kernel_cdf = _tricube_kernel_cdf
        elif kernel == 'exponential':
            self.__kernel_cdf = _exponential_kernel_cdf
        elif kernel == 'logistic':
            self.__kernel_cdf = _logistic_kernel_cdf
        elif kernel == 'sigmoid':
            self.__kernel_cdf = _sigmoid_kernel_cdf
        elif kernel == 'linear':
            self.__kernel_cdf = _linear_kernel_cdf
        elif kernel == 'cosine':
            self.__kernel_cdf = _cosine_kernel_cdf

    @property
    def min_bandwidth(self):
        return self.__min_bandwidth

    @min_bandwidth.setter
    def min_bandwidth(self, min_bandwidth: Optional[float]):
        if min_bandwidth is not None and min_bandwidth <= 0:
            raise ValueError("The minimum bandwidth must be positive")

        self.__min_bandwidth = min_bandwidth

    # noinspection PyUnusedLocal
    def partial_fit(self, X, y=None) -> 'KernelDensity1D':
        if self.__model is None:
            self.__model = np.copy(X)
        else:
            if X.shape[1] != self.__model.shape[1]:
                raise ValueError(f"Invalid number of features, expected {self.__model.shape[1]}, but got {X.shape[1]}")
            self.__model = np.concatenate([self.__model, X], axis=0)
        self.__dirty = True

        return self

    def fit(self, X, y=None) -> 'KernelDensity1D':
        self.__model = None
        return self.partial_fit(X, y)

    def score(self, X, y=None) -> float:
        return np.sum(self.score_samples(X))

    def score_samples(self, X) -> np.ndarray:
        if self.__model is None:
            raise RuntimeError("The model was not yet fitted")

        if X.shape[1] != self.__model.shape[1]:
            raise ValueError(f"Invalid number of features, expected {self.__model.shape[1]}, but got {X.shape[1]}")

        if self.__dirty:
            self.__estimate_bandwidth()
            self.__dirty = False

        sample_count = X.shape[0]

        model_size = self.__model.shape[0]
        feature_count = self.__model.shape[1]

        if is_sparse_matrix(X):
            X = np.asarray(X.todense())

        model_3d = np.transpose(np.broadcast_to(np.atleast_3d(self.__model), (model_size, feature_count, sample_count)),
                                axes=(2, 0, 1)).astype(np.float32)
        X_3d = np.transpose(np.atleast_3d(X), axes=(0, 2, 1)).astype(np.float32)

        delta = X_3d - model_3d

        # calculating probabilities by taking difference of a small interval on CDF
        precision_half = self.precision_.reshape((1, 1, self.precision_.shape[0])) / 2.
        X_upper = delta + precision_half
        X_lower = delta - precision_half

        h = self.bandwidth_.reshape(1, 1, feature_count)
        prob = self.__kernel_cdf(X_upper / h) - self.__kernel_cdf(X_lower / h)
        kernel_estimate = np.sum(prob, axis=1) / model_size + 1.e-32

        return np.sum(np.log(kernel_estimate), axis=1)

    def __estimate_bandwidth(self):
        if isinstance(self.__bandwidth, str):
            if self.__bandwidth == 'scott':
                estimator = scott
            elif self.__bandwidth == 'silverman':
                estimator = silverman
            elif self.__bandwidth == 'norm':
                estimator = norm
            elif self.__bandwidth == 'weka':
                estimator = weka
            else:
                estimator = sheather_jones

            min_h = self.min_bandwidth or 0
            h = np.squeeze(estimator(self.__model))
            self.bandwidth_ = np.maximum(h, min_h)
        elif isinstance(self.__bandwidth, float):
            self.bandwidth_ = np.ones(self.__model.shape[1]) * np.maximum(self.__bandwidth, self.min_bandwidth or 0.)
        else:
            if self.__bandwidth.shape[0] != self.__model.shape[1]:
                raise ValueError(f"Wrong number of bandwidths, expected {self.__model.shape[1]} (one per feature), "
                                 f"but got {self.__bandwidth.shape[0]}")
            self.bandwidth_ = np.maximum(self.__bandwidth, self.min_bandwidth or 0.)

        self.precision_ = weka_precision(np.sort(self.__model, axis=0))


# noinspection PyPep8Naming
def _gaussian_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return st.norm.cdf(X)


# noinspection PyPep8Naming
def _tophat_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return st.uniform.cdf(X, loc=-1, scale=2)


# noinspection PyPep8Naming
def _epanechnikov_kernel_cdf(X: np.ndarray) -> np.ndarray:
    _X = np.clip(X, -1, 1)
    return 3 * _X / 4 - _X ** 3 / 4


# noinspection PyPep8Naming
def _quartic_kernel_cdf(X: np.ndarray) -> np.ndarray:
    _X = np.clip(X, -1, 1)
    return 15 * _X / 16 - 5 * _X ** 3 / 8 + 3 * _X ** 5 / 16


# noinspection PyPep8Naming
def _triweight_kernel_cdf(X: np.ndarray) -> np.ndarray:
    _X = np.clip(X, -1, 1)
    return -35 / 32 * (-_X + _X ** 3 - (3 * _X ** 5) / 5 + _X ** 7 / 7)


# noinspection PyPep8Naming
def _tricube_kernel_cdf(X: np.ndarray) -> np.ndarray:
    _X = np.clip(X, -1, 1)
    return np.where(_X >= 0,
                    1 / 2 - 70 / 81 * (_X ** 10 / 10 - 3 * _X ** 7 / 7 + 3 * _X ** 4 / 4 - _X),
                    1 / 2 + 70 / 81 * (_X ** 10 / 10 + 3 * _X ** 7 / 7 + 3 * _X ** 4 / 4 + _X))


# noinspection PyPep8Naming
def _exponential_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return np.where(X >= 0, 1 - np.exp(-X) / 2, np.exp(X) / 2)


# noinspection PyPep8Naming
def _logistic_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return -1 / (np.exp(X) + 1)


# noinspection PyPep8Naming
def _sigmoid_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return 2 / np.pi * np.arctan(np.exp(X))


# noinspection PyPep8Naming
def _linear_kernel_cdf(X: np.ndarray) -> np.ndarray:
    return st.triang.cdf(X, 0.5, loc=-1, scale=2)


# noinspection PyPep8Naming
def _cosine_kernel_cdf(X: np.ndarray) -> np.ndarray:
    _X = np.clip(X, -1, 1)
    return np.sin(np.pi * _X / 2) / 2
