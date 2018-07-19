import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.classes import Regression


def test_set_beta(N=200, D=100):
    r"""Test beta setter"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm_pnet = Regression(X, y)

    new_beta = np.random.randn(D)
    new_r = y - np.dot(X, new_beta)
    lm_pnet.beta = new_beta

    np.testing.assert_almost_equal(lm_pnet.beta, new_beta, decimal=6)
    np.testing.assert_almost_equal(lm_pnet._r, new_r, decimal=6)


def test_ordinary_least_squares_explicit(N=200, D=100):
    r"""Test explicitly coded special case OLS numba code in :meth:`plasticnet.classes.Regression.fit_ordinary_least_squares` against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_ordinary_least_squares(tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_ridge_explicit(N=200, D=100):
    r"""Test explicitly coded special case ridge numba code in :meth:`plasticnet.classes.Regression.fit_ridge` against sklearn elastic net with l1_ratio=0"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=0.0, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_ridge(lambda_total=lambda_total, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_lasso_explicit(N=200, D=100):
    r"""Test explicitly coded special case lasso numba code in :meth:`plasticnet.classes.Regression.fit_lasso` against sklearn elastic net with `l1_ratio=1`"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=1.0, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_lasso(lambda_total=lambda_total, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_elastic_net_explicit(N=200, D=100):
    r"""Test explicitly coded special case elastic net numba code in :meth:`plasticnet.classes.Regression.fit_elastic_net` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_elastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_general_plastic_net(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_general_plastic_net` with :math:`\xi=0` and :math:`\zeta=0` against sklearn elastic net."""

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

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.zeta = zeta
    lm_pnet.fit_general_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_plastic_ridge(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_plastic_ridge` with :math:`\zeta=0` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=0, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_plastic_ridge(lambda_total=lambda_total, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_plastic_lasso(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_plastic_lasso` with :math:`\xi=0` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=1, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_plastic_lasso(lambda_total=lambda_total, tol=1e-8, max_iter=1000)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_hard_plastic_net(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_hard_plastic_net` with :math:`\xi=0` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_hard_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_soft_plastic_net(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_soft_plastic_net` with :math:`\zeta=0` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_soft_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)


def test_unified_plastic_net(N=200, D=100):
    r"""Test :meth:`plasticnet.classes.Regression.fit_unified_plastic_net` with :math:`\xi=0` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=1e-8, max_iter=1000
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_unified_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=1e-8, max_iter=1000
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=6)
