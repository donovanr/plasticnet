import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.solvers.functional import ols, enet, gpnet


def test_ols_explicit(N=200, D=100):
    """Test explicitly coded special case OLS numba code in solve_ols against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    beta = ols(X, y, tol=1e-8, max_iter=1e3)

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_enet_explicit_ols(N=200, D=100):
    """Test explicitly coded special case elastic net with lambda=0 in solve_enet against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    beta = enet(X, y, lambda_total=0.0, alpha=0.0, tol=1e-8, max_iter=1e3)

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_enet_explicit(N=200, D=100):
    """Test explicitly coded special case elastic net against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    enet_lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1e3
    )
    enet_lm.fit(X, y)

    beta = enet(X, y, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1e3)

    np.testing.assert_almost_equal(enet_lm.coef_, beta, decimal=6)


def test_ols_general(N=200, D=100):
    """Test OLS (lambda=0 in solve_gpnet) against sklearn LinearRegression"""

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

    beta = gpnet(
        X, y, xi, zeta, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1e3
    )

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)


def test_enet_general(N=200, D=100):
    """Test elastic net (xi=0 & zeta=0 in solve_gpnet) against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1e3
    )
    lm.fit(X, y)

    beta = gpnet(
        X, y, xi, zeta, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1e3
    )

    np.testing.assert_almost_equal(lm.coef_, beta, decimal=6)
