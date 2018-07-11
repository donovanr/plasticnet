import numpy as np
import numpy.testing as npt

from sklearn.preprocessing import scale
from sklearn import linear_model

from plasticnet.plasticnet.plasticnet import solve_ols


def test_ols_explicit():
    """General plastic net tests"""

    y = scale(np.array([3, 1, 4, 1], dtype=float))
    X = scale(np.array([[5, 9, 2], [6, 5, 3], [5, 8, 9], [7, 9, 3]], dtype=float))

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    beta = solve_ols(X, y, thresh=1e-8, max_iters=1000)

    npt.assert_almost_equal(ols.coef_, beta, decimal=8)
