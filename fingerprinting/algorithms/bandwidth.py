import numpy as np
import scipy.stats as st

ZERO_VAR_BANDWIDTH = .01 / 6


def wmean(x, w):
    """
    Weighted mean
    """
    return np.sum(x * w, axis=0) / np.sum(w, axis=0)


def wvar(x, w):
    """
    Weighted variance
    """
    return np.sum(w * (x - wmean(x, w)) ** 2, axis=0) / (np.sum(w, axis=0) - 1)


def dnorm(x):
    return st.norm.pdf(x)


def weka_precision(sorted_x):
    delta = sorted_x[1:, :] - sorted_x[:-1, :]
    delta_sum = np.sum(delta, axis=0)
    distinct = np.sum(delta > 0, axis=0)

    # 0.01 is the default precision from Weka's NaiveBayes class
    return np.maximum(delta_sum / (distinct + 1e-23), 0.01)


def weka(x, precision=None, weights=None):
    sorted_x = np.sort(x, axis=0)

    if weights is None:
        weights = np.ones(x.shape)
    n = np.sum(weights, axis=0).astype(np.float64)

    x_range = sorted_x[-1, :] - sorted_x[0, :]

    precision = precision if precision is not None else weka_precision(sorted_x)
    h = np.maximum(x_range / np.sqrt(n), precision / 6)

    return h


def silverman(x, weights=None, **kwargs):
    return __iqr_method(x, weights, 0.9)


def scott(x, weights=None, **kwargs):
    return __iqr_method(x, weights, 1.059)


def __iqr_method(x, weights, factor):
    iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    a = np.minimum(np.std(x, ddof=1, axis=0), iqr / 1.349)

    if weights is None:
        weights = np.ones(x.shape)
    n = np.sum(weights, axis=0).astype(np.float64)

    h = factor * a * n ** (-0.2)

    # default bandwidth for zero variance case
    return np.where(h == 0., ZERO_VAR_BANDWIDTH, h)


def norm(x, weights=None, **kwargs):
    """
    Bandwidth estimate assuming f is normal. See paragraph 2.4.2 of
    Bowman and Azzalini[1]_ for details.

    References
    ----------
    .. [1] Applied Smoothing Techniques for Data Analysis: the
        Kernel Approach with S-Plus Illustrations.
        Bowman, A.W. and Azzalini, A. (1997).
        Oxford University Press, Oxford
    """

    if weights is None:
        weights = np.ones(x.shape)

    n = np.sum(weights, axis=0).astype(np.float64)

    sd = np.sqrt(wvar(x, weights))
    return sd * (4 / (3 * n)) ** (1 / 5.0)


def sheather_jones(x, weights=None, **kwargs):
    """
    Sheather-Jones bandwidth estimator [1]_.
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """

    h0 = norm(x)
    v0 = __sj_eq(x, h0)

    hstep = np.where(v0 > 0, 1.1, 0.9)

    h1 = h0 * hstep
    v1 = __sj_eq(x, h1)

    done = ((v1 * v0 <= 0) | (h0 == 0.))
    res_h0 = h0
    res_h1 = h1
    res_v0 = v0
    res_v1 = v1

    while not np.all(done):
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = __sj_eq(x, h1)

        res_h0 = np.where(~done, h0, res_h0)
        res_h1 = np.where(~done, h1, res_h1)
        res_v0 = np.where(~done, v0, res_v0)
        res_v1 = np.where(~done, v1, res_v1)
        done = done | (v1 * v0 <= 0)

    h = res_h0 + (res_h1 - res_h0) * np.abs(res_v0) / (np.abs(res_v0) + np.abs(res_v1))

    # default bandwidth for zero variance case
    return np.where(h == 0., ZERO_VAR_BANDWIDTH, h)


def __sj_eq(x, h):
    """
    Equation 12 of Sheather and Jones [1]_
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """

    def phi6(x_):
        return (x_ ** 6 - 15 * x_ ** 4 + 45 * x_ ** 2 - 15) * dnorm(x_)

    def phi4(x_):
        return (x_ ** 4 - 6 * x_ ** 2 + 3) * dnorm(x_)

    n = x.shape[0]
    f = x.shape[1]
    one = np.ones((1, n))

    lam = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    a = 0.92 * lam * n ** (-1 / 7.0) + 1.e-32
    b = 0.912 * lam * n ** (-1 / 9.0) + 1.e-32

    w = np.transpose(np.broadcast_to(x, (n, n, f)), (2, 0, 1))
    w = w - np.transpose(w, (0, 2, 1))

    w1 = phi6(w / b.reshape((f, 1, 1)))
    tdb = np.squeeze(np.dot(np.dot(one, w1), one.T))
    tdb = -tdb / (n * (n - 1) * b ** 7)

    w1 = phi4(w / a.reshape((f, 1, 1)))
    sda = np.squeeze(np.dot(np.dot(one, w1), one.T))
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (np.abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0) + 1.e-32

    w1 = phi4(w / alpha2.reshape(f, 1, 1))
    sdalpha2 = np.squeeze(np.dot(np.dot(one, w1), one.T).ravel())
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (st.norm.pdf(0, scale=np.sqrt(2)) / (n * np.abs(sdalpha2))) ** 0.2 - h
