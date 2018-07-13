import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.plasticnet.plasticnet import solve_ols, solve_enet


def test_ols_explicit(N=200, D=100):
    """Test explicit OLS numba code in solve_ols against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_ols(X, y, tol=1e-8, max_iter=1e3)

    np.testing.assert_almost_equal(ols.coef_, beta, decimal=6)


def test_enet_explicit_ols(N=200, D=100):
    """Test explicit lambda=0 elastic net in solve_enet against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_enet(X, y, lambda_total=0.0, alpha=0.0, tol=1e-8, max_iter=1e3)

    np.testing.assert_almost_equal(ols.coef_, beta, decimal=6)


def test_enet_explicit(N=200, D=100):
    """Test elastic net against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    enet = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1e3
    )
    enet.fit(X, y)

    beta = solve_enet(
        X, y, lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1e3
    )

    np.testing.assert_almost_equal(enet.coef_, beta, decimal=6)
