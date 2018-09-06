import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression

from plasticnet.classes import Regression


def test_set_beta(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test beta property"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm_pnet = Regression(X, y)

    new_beta = np.random.randn(D)
    new_r = y - np.dot(X, new_beta)
    lm_pnet.beta = new_beta

    np.testing.assert_almost_equal(lm_pnet.beta, new_beta, decimal=4)
    np.testing.assert_almost_equal(lm_pnet._r, new_r, decimal=4)


def test_ordinary_least_squares_explicit(N=1500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test explicitly coded special case OLS numba code in :meth:`plasticnet.classes.Regression.fit_ordinary_least_squares` against sklearn LinearRegression"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N, coef=True
    )
    X, y = scale(X), scale(y)

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_ordinary_least_squares(tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=4)


def test_ridge_explicit(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test explicitly coded special case ridge numba code in :meth:`plasticnet.classes.Regression.fit_ridge` against sklearn elastic net with l1_ratio=0"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()

    lm = linear_model.Ridge(alpha=lambda_total * N, tol=tol, max_iter=max_iter)
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_ridge(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=4)


def test_lasso_explicit(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test explicitly coded special case lasso numba code in :meth:`plasticnet.classes.Regression.fit_lasso` against sklearn elastic net with `l1_ratio=1`"""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=1.0, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_lasso(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=4)


def test_elastic_net_explicit(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test explicitly coded special case elastic net numba code in :meth:`plasticnet.classes.Regression.fit_elastic_net` against sklearn elastic net."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.fit_elastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=4)


def test_general_plastic_net(N=500, D=1000, tol=1e-12, max_iter=10000):
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
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.zeta = zeta
    lm_pnet.fit_general_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(lm.coef_, lm_pnet.beta, decimal=4)


def test_plastic_ridge_trivial(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test plastic ridge(:math:`\zeta=0` in :meth:`plasticnet.classes.Regression.fit_plastic_ridge`) against sklearn ElasticNet."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.Ridge(alpha=lambda_total * N, tol=tol, max_iter=max_iter)
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_plastic_ridge(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_plastic_ridge_real(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test :meth:`plasticnet.classes.Regression.fit_plastic_ridge` against sklearn ElasticNet with transformed variables."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    zeta = np.random.randn(D).astype(np.float64)

    X_prime = X
    y_prime = y - np.dot(X, zeta)

    lm = linear_model.Ridge(alpha=lambda_total * N, tol=tol, max_iter=max_iter)
    lm.fit(X_prime, y_prime)
    beta_lm = lm.coef_ + zeta

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_plastic_ridge(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


# test_plastic_lasso_trivial
# test_plastic_lasso_real


def test_plastic_lasso_trivial(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test plastic lasso (:math:`\xi=0` in :meth:`plasticnet.classes.Regression.fit_plastic_lasso`) against sklearn ElasticNet."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=1, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_plastic_lasso(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_plastic_lasso_real(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test :meth:`plasticnet.classes.Regression.fit_plastic_lasso` against sklearn ElasticNet with transformed variables."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    xi = np.random.randn(D).astype(np.float64)

    X_prime = X
    y_prime = y - np.dot(X, xi)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=1, tol=tol, max_iter=max_iter
    )
    lm.fit(X_prime, y_prime)
    beta_lm = lm.coef_ + xi

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_plastic_lasso(lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_hard_plastic_net_trivial(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test hard plastic net (:math:`\xi=0` and in :meth:`plasticnet.classes.Regression.fit_hard_plastic_net`) against sklearn ElasticNet."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_hard_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_hard_plastic_net_limiting_cases(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test hard plastic net :meth:`plasticnet.classes.Regression.fit_hard_plastic_net` against sklearn ElasticNet in limiting cases."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    xi = np.random.randn(D).astype(np.float64)

    X_prime = X
    y_prime = y - np.dot(X, xi)

    alpha = 1.0

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X_prime, y_prime)
    beta_lm = lm.coef_ + xi

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_hard_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)

    alpha = 0.0

    lm = linear_model.Ridge(alpha=lambda_total * N, tol=tol, max_iter=max_iter)
    lm.fit(X, y)

    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_hard_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


# test_soft_plastic_net_trivial
# test_soft_plastic_net_limiting_cases


def test_soft_plastic_net_trivial(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test soft plastic net (:math:`\zeta=0` in :meth:`plasticnet.classes.Regression.fit_soft_plastic_net`) against sklearn ElasticNet."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    zeta = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_soft_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_soft_plastic_net_limiting_cases(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test :meth:`plasticnet.classes.Regression.fit_soft_plastic_net` against sklearn ElasticNet in limiting cases."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    zeta = np.random.randn(D).astype(np.float64)

    alpha = 1.0

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_soft_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)

    alpha = 0.0

    X_prime = X
    y_prime = y - np.dot(X, zeta)

    lm = linear_model.Ridge(alpha=lambda_total * N, tol=tol, max_iter=max_iter)
    lm.fit(X_prime, y_prime)

    beta_lm = lm.coef_ + zeta

    lm_pnet = Regression(X, y)
    lm_pnet.zeta = zeta
    lm_pnet.fit_soft_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


# test_unified_plastic_net_trivial
# test_unified_plastic_net_real


def test_unified_plastic_net_trivial(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test unified plastic net (:math:`\xi=0` in :meth:`plasticnet.classes.Regression.fit_unified_plastic_net`) against sklearn ElasticNet."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.zeros(D, dtype=np.float64)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X, y)
    beta_lm = lm.coef_

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_unified_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)


def test_unified_plastic_net_real(N=500, D=1000, tol=1e-12, max_iter=10000):
    r"""Test :meth:`plasticnet.classes.Regression.fit_unified_plastic_net` against sklearn ElasticNet with transformed variables."""

    X, y, beta_true = make_regression(
        n_samples=N, n_features=D, n_informative=N // 10, coef=True
    )
    X, y = scale(X), scale(y)

    lambda_total = np.random.exponential()
    alpha = np.random.rand()
    xi = np.random.randn(D).astype(np.float64)

    X_prime = X
    y_prime = y - np.dot(X, xi)

    lm = linear_model.ElasticNet(
        alpha=lambda_total, l1_ratio=alpha, tol=tol, max_iter=max_iter
    )
    lm.fit(X_prime, y_prime)
    beta_lm = lm.coef_ + xi

    lm_pnet = Regression(X, y)
    lm_pnet.xi = xi
    lm_pnet.fit_unified_plastic_net(
        lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    np.testing.assert_almost_equal(beta_lm, lm_pnet.beta, decimal=4)
