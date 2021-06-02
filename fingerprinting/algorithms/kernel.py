from typing import Optional, Union

import numpy as np

from ..util.pipeline import is_sparse_matrix
from .bandwidth import scott, silverman, norm, sheather_jones

VALID_KERNELS = frozenset(['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'])
VALID_BW_ESTIMATORS = frozenset(['scott', 'silverman', 'norm', 'sj'])
SQRT_2PI = np.sqrt(2 * np.pi)


# noinspection PyPep8Naming
def _gaussian_kernel(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.exp(-(X**2) / (2 * h**2)) / (h * SQRT_2PI)


# noinspection PyPep8Naming
def _tophat_kernel(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.where(np.abs(X) > h, 0, 1 / (2 * h))


# noinspection PyPep8Naming
def _epanechnikov_kernel(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.maximum((-(X**2) / h**2 + 1), 0) * 3 / (4 * h)


# noinspection PyPep8Naming
def _exponential_kernel(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(X) / h) / (2 * h)


# noinspection PyPep8Naming
def _linear_kernel(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.maximum(-np.abs(X) / h + 1, 0) / h


# noinspection PyPep8Naming
def _cosine_kernel(X: np.ndarray, h: np.ndarray):
    return np.where(np.abs(X) > h, 0, np.cos(np.pi * X / (2 * h))) * np.pi / (4 * h)


# noinspection PyPep8Naming
class KernelDensity1D:
    def __init__(self, bandwidth: Union[str, float, np.ndarray] = 1., kernel: str = 'gaussian'):
        self.__model: Optional[np.ndarray] = None
        self.__dirty = False

        self.bandwidth_: Optional[np.ndarray] = None

        self.bandwidth = bandwidth
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
            self.__kernel_func = _gaussian_kernel
        elif kernel == 'tophat':
            self.__kernel_func = _tophat_kernel
        elif kernel == 'epanechnikov':
            self.__kernel_func = _epanechnikov_kernel
        elif kernel == 'exponential':
            self.__kernel_func = _exponential_kernel
        elif kernel == 'linear':
            self.__kernel_func = _linear_kernel
        elif kernel == 'cosine':
            self.__kernel_func = _cosine_kernel

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

    def score(self, X) -> float:
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

        kernel_estimate = np.sum(self.__kernel_func(model_3d - X_3d, self.bandwidth_.reshape(1, 1, feature_count)),
                                 axis=1) / (model_size * self.bandwidth_.reshape(1, feature_count)) + 1.e-32

        return np.sum(np.log(kernel_estimate), axis=1)

    def __estimate_bandwidth(self):
        if isinstance(self.__bandwidth, str):
            if self.__bandwidth == 'scott':
                estimator = scott
            elif self.__bandwidth == 'silverman':
                estimator = silverman
            elif self.__bandwidth == 'norm':
                estimator = norm
            else:
                estimator = sheather_jones

            self.bandwidth_ = np.squeeze(estimator(self.__model))

            if np.count_nonzero(self.bandwidth_) < self.bandwidth_.size:
                self.bandwidth_ = self.bandwidth_ + 1.e-32
        elif isinstance(self.__bandwidth, float):
            self.bandwidth_ = np.ones(self.__model.shape[1]) * self.__bandwidth
        else:
            if self.__bandwidth.shape[0] != self.__model.shape[1]:
                raise ValueError(f"Wrong number of bandwidths, expected {self.__model.shape[1]} (one per feature), "
                                 f"but got {self.__bandwidth.shape[0]}")
            self.bandwidth_ = self.__bandwidth
