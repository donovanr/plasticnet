import numpy.testing as npt

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

    beta = solve_ols(X, y, thresh=1e-8, max_iters=1e3)

    npt.assert_almost_equal(ols.coef_, beta, decimal=6)


def test_enet_ols(N=200, D=100):
    """Test explicit lambda=0 elastic net in solve_enet against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_enet(X, y, lambda_total=0.0, alpha=0.0, thresh=1e-8, max_iters=1e3)

    npt.assert_almost_equal(ols.coef_, beta, decimal=6)


def test_enet_sklearn(N=200, D=100):
    """General plastic net tests ElasticNet"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    enet = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5, tol=1e-8, max_iter=1e3)
    enet.fit(X, y)

    beta = solve_enet(X, y, lambda_total=0.5, alpha=0.5, thresh=1e-8, max_iters=1e3)

    npt.assert_almost_equal(enet.coef_, beta, decimal=6)
