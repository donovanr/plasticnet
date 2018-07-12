import numpy as np
import numpy.testing as npt

from sklearn.preprocessing import scale
from sklearn import linear_model

from plasticnet.plasticnet.plasticnet import solve_ols, solve_enet


def test_ols_explicit():
    """Test explicit OLS numba code in solve_ols against sklearn LinearRegression"""

    y = scale(np.array([3, 1, 4, 1], dtype=float))
    X = scale(np.array([[5, 9, 2], [6, 5, 3], [5, 8, 9], [7, 9, 3]], dtype=float))

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_ols(X, y, thresh=1e-8, max_iters=1000)

    npt.assert_almost_equal(ols.coef_, beta, decimal=8)


def test_enet_ols():
    """Test explicit lambda=0 elastic net in solve_enet against sklearn LinearRegression"""

    y = scale(np.array([3, 1, 4, 1], dtype=float))
    X = scale(np.array([[5, 9, 2], [6, 5, 3], [5, 8, 9], [7, 9, 3]], dtype=float))

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_enet(X, y, lambda_total=0.0, alpha=0.0, thresh=1e-8, max_iters=100000)

    npt.assert_almost_equal(ols.coef_, beta, decimal=8)


def test_enet_sklearn():
    """General plastic net tests ElasticNet"""

    y = scale(np.array([3, 1, 4, 1], dtype=float))
    X = scale(np.array([[5, 9, 2], [6, 5, 3], [5, 8, 9], [7, 9, 3]], dtype=float))

    enet = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=100000, tol=1e-8)
    enet.fit(X, y)

    beta = solve_enet(X, y, lambda_total=0.5, alpha=0.5, thresh=1e-8, max_iters=100000)

    npt.assert_almost_equal(enet.coef_, beta, decimal=8)
