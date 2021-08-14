from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

# noinspection PyPep8Naming
from .bandwidth import weka_precision
from .kernel import KernelDensity1D
from ..api.typing import Classifier
from ..util.pipeline import is_sparse_matrix
from logging import getLogger

_LOGGER = getLogger(__name__)


class KernelDensityNB(Classifier, BaseEstimator, ClassifierMixin):
    def __init__(self, priors=None, kernel='gaussian', bandwidth=1.0, kde="nd", min_bandwidth: Optional[float] = None):
        self.priors = priors
        self.kde = kde

        self.__kernel = 'gaussian'
        self.__bandwidth = 1.0
        self.__min_bandwidth = None

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth

        self.classes_ = np.array([])
        self.class_count_ = np.array([])
        self.class_prior_ = np.array([])
        self.log_prior_ = np.array([])
        self.models_ = np.array([])

    # noinspection PyPep8Naming
    def fit(self, X, y, **fit_params) -> 'KernelDensityNB':
        self.classes_ = np.sort(np.unique(y))

        if is_sparse_matrix(X):
            X = np.asarray(X.todense())

        training = [X[y == yi] for yi in self.classes_]
        self.class_count_ = np.array([Xi.shape[0] for Xi in training])

        if self.priors is None:
            self.class_prior_ = self.class_count_ / y.shape[0]
        elif self.priors == 'uniform':
            self.class_prior_ = np.ones_like(self.classes_) / self.classes_.shape[0]
        else:
            if self.priors.shape[0] != self.classes_.shape[0]:
                raise ValueError("Predefined class priors do not have shape (n_classes,) "
                                 f"({self.priors.shape} != {self.classes_.shape})")
            self.class_prior_ = self.priors

        self.log_prior_ = np.log(self.class_prior_)
        if self.kde == "nd":
            # Use n-dimensional KDE (one model per class)
            self.models_ = [KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(Xi) for Xi in training]
        else:
            # Calculate precision per attribute first
            precision = np.squeeze(weka_precision(np.sort(X, axis=0)))
            # Use 1-dimensional KDE (one model per class)
            self.models_ = [KernelDensity1D(kernel=self.kernel, bandwidth=self.bandwidth, precision=precision,
                                            min_bandwidth=self.min_bandwidth).fit(Xi) for Xi in training]

        return self

    # noinspection PyPep8Naming
    def predict(self, X) -> np.ndarray:
        log_prob = np.array([model.score_samples(X) for model in self.models_]).T
        result = log_prob + self.log_prior_

        return self.classes_[np.argmax(result, axis=1)]

    def get_params(self, **kwargs):
        return {
            'priors': self.priors,
            'kernel': self.kernel,
            'kde': self.kde,
            'bandwidth': self.bandwidth,
            'min_bandwidth': self.min_bandwidth
        }

    def set_params(self, **params):
        if 'priors' in params:
            self.priors = params['priors']
        if 'kde' in params:
            self.kde = params['kde']
        if 'kernel' in params:
            self.kernel = params['kernel']
        if 'bandwidth' in params:
            self.bandwidth = params['bandwidth']
        if 'min_bandwidth' in params:
            self.min_bandwidth = params['min_bandwidth']

    @property
    def priors(self):
        return self.__priors

    @priors.setter
    def priors(self, priors):
        if isinstance(priors, str) and priors != 'uniform':
            raise ValueError("priors must be array-like, None (estimate prior), or 'uniform' (uniform prior)")
        elif priors is not None and not isinstance(priors, str) and (not hasattr(priors, 'shape')
                                                                     or len(priors.shape) != 1 or priors.shape[0] == 0):
            raise ValueError("Class priors must be an array-like of shape (n_classes,)")

        self.__priors = priors

    @property
    def kernel(self):
        return self.__kernel

    @kernel.setter
    def kernel(self, kernel: str):
        # check valid kernel
        if self.kde == 'nd':
            KernelDensity(kernel=kernel, bandwidth=self.bandwidth)
        elif self.kde == '1d':
            KernelDensity1D(kernel=kernel, bandwidth=self.bandwidth, min_bandwidth=self.min_bandwidth)

        self.__kernel = kernel

    @property
    def kde(self):
        return self.__kde

    @kde.setter
    def kde(self, kde: str):
        if kde != "1d" and kde != "nd":
            raise ValueError("kde must be one of '1d', 'nd'")

        self.__kde = kde

    @property
    def bandwidth(self):
        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        # check valid bandwidth
        if self.kde == 'nd':
            KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
        elif self.kde == '1d':
            KernelDensity1D(kernel=self.kernel, bandwidth=bandwidth, min_bandwidth=self.min_bandwidth)

        self.__bandwidth = bandwidth

    @property
    def min_bandwidth(self):
        return self.__min_bandwidth

    @min_bandwidth.setter
    def min_bandwidth(self, min_bandwidth: Optional[float]):
        # check valid min_bandwidth
        if self.kde == "nd":
            _LOGGER.warning("Got a min_bandwidth while kde = nd. n-dimensional KDE does not support min_bandwidth")
        if self.kde == '1d':
            KernelDensity1D(kernel=self.kernel, bandwidth=self.bandwidth, min_bandwidth=min_bandwidth)

        self.__min_bandwidth = min_bandwidth
