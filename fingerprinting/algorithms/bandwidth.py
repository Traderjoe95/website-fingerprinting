import numpy as np


class Normal(object):
    """
    The 1D normal (or Gaussian) distribution.
    """

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(Normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
               0.5 * ((x - mu) ** 2) / (sigma ** 2)

    @staticmethod
    def rvs(mu, sigma, n=1):
        return np.random.normal(mu, sigma, n)


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
    return Normal.pdf(x, 0.0, 1.0)


def silverman(x, weights=None):
    IQR = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    A = np.minimum(np.std(x, ddof=1, axis=0), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = np.sum(weights, axis=0).astype(np.float64)

    return 0.9 * A * n ** (-0.2)


def scott(x, weights=None):
    IQR = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    A = np.minimum(np.std(x, ddof=1, axis=0), IQR / 1.349)

    if weights is None:
        weights = np.ones(x.shape)
    n = np.sum(weights, axis=0).astype(np.float64)

    return 1.059 * A * n ** (-0.2)


def norm(x, weights=None):
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


def sheather_jones(x, weights=None):
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

    return res_h0 + (res_h1 - res_h0) * np.abs(res_v0) / (np.abs(res_v0) + np.abs(res_v1))


def __sj_eq(x, h):
    """
    Equation 12 of Sheather and Jones [1]_
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """
    phi6 = lambda x: (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * dnorm(x)
    phi4 = lambda x: (x ** 4 - 6 * x ** 2 + 3) * dnorm(x)

    n = x.shape[0]
    f = x.shape[1]
    one = np.ones((1, n))

    lam = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    a = 0.92 * lam * n ** (-1 / 7.0) + 1.e-32
    b = 0.912 * lam * n ** (-1 / 9.0) + 1.e-32

    W = np.transpose(np.broadcast_to(x, (n, n, f)), (2, 0, 1))
    W = W - np.transpose(W, (0, 2, 1))

    W1 = phi6(W / b.reshape((f, 1, 1)))
    tdb = np.squeeze(np.dot(np.dot(one, W1), one.T))
    tdb = -tdb / (n * (n - 1) * b ** 7)

    W1 = phi4(W / a.reshape((f, 1, 1)))
    sda = np.squeeze(np.dot(np.dot(one, W1), one.T))
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (np.abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0) + 1.e-32

    W1 = phi4(W / alpha2.reshape(f, 1, 1))
    sdalpha2 = np.squeeze(np.dot(np.dot(one, W1), one.T).ravel())
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (Normal.pdf(0, 0, np.sqrt(2)) / (n * np.abs(sdalpha2))) ** 0.2 - h
