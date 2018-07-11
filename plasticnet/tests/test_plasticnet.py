import numpy as np
import numpy.testing as npt

from sklearn.preprocessing import scale
from sklearn import linear_model


def test_gpnet():
    """General plastic net tests"""
    npt.assert_almost_equal(np.array([0.0, 0.0]), np.array([0.0, 0.0]), decimal=1)


def test_ols_regression():
    """General plastic net tests"""

    y = scale(np.array([3, 1, 4, 1], dtype=float))
    X = scale(np.array([[5, 9, 2], [6, 5, 3], [5, 8, 9], [7, 9, 3]], dtype=float))

    ols = linear_model.LinearRegression()
    ols.fit(X, y)

    npt.assert_almost_equal(
        ols.coef_, np.array([-0.69631062, 0.34417626, 0.38806814]), decimal=8
    )
