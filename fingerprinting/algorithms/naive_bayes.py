import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

# noinspection PyPep8Naming
from fingerprinting.algorithms.kernel import KernelDensity1D
from fingerprinting.util.pipeline import is_sparse_matrix


class KernelDensityNB(BaseEstimator, ClassifierMixin):
    def __init__(self, priors=None, kernel='gaussian', bandwidth=1.0, kde="nd"):
        if isinstance(priors, str) and priors != 'uniform':
            raise ValueError("priors must be array-like, None (estimate prior), or 'uniform' (uniform prior)")
        elif priors is not None and not isinstance(priors, str) and (not hasattr(priors, 'shape')
                                                                     or len(priors.shape) != 1 or priors.shape[0] == 0):
            raise ValueError("Class priors must be an array-like of shape (n_classes,)")

        if kde != "1d" and kde != "nd":
            raise ValueError("kde must be one of '1d', 'nd'")

        # check valid kernel and bandwidth
        if kde == 'nd':
            KernelDensity(kernel=kernel, bandwidth=bandwidth)
        elif kde == '1d':
            KernelDensity1D(kernel=kernel, bandwidth=bandwidth)

        self.priors = priors
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kde = kde

        self.classes_ = np.array([])
        self.class_count_ = np.array([])
        self.class_prior_ = np.array([])
        self.log_prior_ = np.array([])
        self.models_ = np.array([])

    # noinspection PyPep8Naming
    def fit(self, X, y) -> 'KernelDensityNB':
        self.classes_ = np.sort(np.unique(y))

        if is_sparse_matrix(X):
            X = X.todense()

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
            self.models_ = [KernelDensity1D(kernel=self.kernel, bandwidth=self.bandwidth).fit(Xi) for Xi in training]

        return self

    # noinspection PyPep8Naming
    def predict(self, X) -> np.ndarray:
        log_prob = np.array([model.score_samples(X) for model in self.models_]).T
        result = log_prob + self.log_prior_

        return self.classes_[np.argmax(result, axis=1)]
