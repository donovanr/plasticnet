import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.solvers.functional import (
    ordinary_least_squares,
    elastic_net,
    general_plastic_net,
)


def test_ordinary_least_squares_explicit(N=200, D=100):
    r"""Test explicitly coded special case OLS numba code in :meth:`plasticnet.solvers.functional.ordinary_least_squares` against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    beta = ordinary_least_squares(X, y, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_elastic_net_explicit_ordinary_least_squares(N=200, D=100):
    r"""Test explicitly coded special case elastic net with :math:`\lambda=0` in :meth:`plasticnet.solvers.functional.elastic_net` against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    beta = elastic_net(X, y, lambda_total=0.0, alpha=0.0, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_elastic_net_explicit(N=200, D=100):
    r"""Test explicitly coded elastic net in :meth:`plasticnet.solvers.functional.elastic_net` against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    elastic_net_lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    elastic_net_lm.fit(X, y)

    beta = elastic_net(
        X, y, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(elastic_net_lm.coef_, beta, decimal=6)


def test_ordinary_least_squares_general(N=200, D=100):
    r"""Test OLS (:math:`\lambda=0` in :meth:`plasticnet.solvers.functional.general_plastic_net`) against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = 0.0
    alpha = 0.0
    xi = np.zeros(D, dtype=np.float64)
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    beta = general_plastic_net(
        X, y, xi, zeta, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_elastic_net_general(N=200, D=100):
    r"""Test elastic net (:math:`\xi=0` and :math:`\zeta=0` in :meth:`plasticnet.solvers.functional.general_plastic_net`) against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    beta = general_plastic_net(
        X, y, xi, zeta, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)
