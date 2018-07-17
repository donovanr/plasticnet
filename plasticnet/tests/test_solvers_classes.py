import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.solvers.classes import Regression


def test_ols_explicit(N=200, D=100):
    """Test explicitly coded special case OLS numba code in solve_ols against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm_sklearn = linear_model.LinearRegression()
    lm_sklearn.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_ols(tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm_sklearn.coef_, lm_pnet.beta, decimal=6)


def test_enet_explicit(N=200, D=100):
    """Test explicitly coded special case elastic net against sklearn ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    lm_sklearn = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm_sklearn.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_enet(lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm_sklearn.coef_, lm_pnet.beta, decimal=6)
