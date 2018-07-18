import numpy as np

from .in_place import (
    enet_,
    gpnet_,
    hpnet_,
    lasso_,
    ols_,
    plasso_,
    pridge_,
    ridge_,
    spnet_,
    upnet_,
)


class Regression:
    r"""
    This class encapsulates a regression problem.
    It stores the data matrix :math:`X` and the target :math:`\vec{y}`, and provides as methods all of the in-place solvers in :mod:`plasticnet.solvers.in_place`.
    It also stores the coefficient vector :math:`\vec{\beta}`, and the penalized regression target vectors :math:`\vec{\xi}` (L1 target) and :math:`\vec{\zeta}` (L2 target).
    Calling any of the ``fit_*`` methods below will update :math:`\vec{\beta}` in-place.

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.

    Attributes:
        beta (numpy.ndarray): shape (D,) coefficient vector
        xi (numpy.ndarray): shape (D,) L1 coefficient target vector
        zeta (numpy.ndarray): shape (D,) L2 coefficient target vector

    .. warning:: Never set :math:`\vec{\beta}` by hand.  Always use the :meth:`set_beta` method.
                 It's fine to set :math:`\vec{\xi}` and :math:`\vec{\zeta}` by hand.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

        N, D = self.X.shape
        self.beta = np.zeros(D, dtype=np.float64)
        self._r = self.y - np.dot(self.X, self.beta)

        self.xi = np.zeros(D, dtype=np.float64)
        self.zeta = np.zeros(D, dtype=np.float64)

    def set_beta(self, beta):
        r"""Sets :math:`\vec{beta}` to desired value while also updating the residual vector via :math:`\vec{r} = \vec{y} - X\vec{\beta}`."""
        self.beta = beta
        self._r = self.y - np.dot(self.X, self.beta)

    def fit_ols(self, tol=1e-8, max_iter=1000):
        r"""In-place ordinary least squares regression.
        See :meth:`plasticnet.solvers.in_place.ols_` for documentation."""
        ols_(self.beta, self._r, self.X, tol=tol, max_iter=max_iter)

    def fit_ridge(self, lambda_total=1.0, tol=1e-8, max_iter=1000):
        r"""In-place ridge regression.
        See :meth:`plasticnet.solvers.in_place.enet_` for documentation."""
        ridge_(
            self.beta,
            self._r,
            self.X,
            lambda_total=lambda_total,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_lasso(self, lambda_total=1.0, tol=1e-8, max_iter=1000):
        r"""In-place lasso regression.
        See :meth:`plasticnet.solvers.in_place.enet_` for documentation."""
        lasso_(
            self.beta,
            self._r,
            self.X,
            lambda_total=lambda_total,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_enet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place elastic net regression.
        See :meth:`plasticnet.solvers.in_place.enet_` for documentation."""
        enet_(
            self.beta,
            self._r,
            self.X,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_gpnet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place general plastic net regression.
        See :meth:`plasticnet.solvers.in_place.gpnet_` for documentation."""
        gpnet_(
            self.beta,
            self._r,
            self.X,
            self.xi,
            self.zeta,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_pridge(self, lambda_total=1.0, tol=1e-8, max_iter=1000):
        r"""In-place plastic ridge regression.
        See :meth:`plasticnet.solvers.in_place.pridge_` for documentation."""
        pridge_(
            self.beta,
            self._r,
            self.X,
            self.zeta,
            lambda_total=lambda_total,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_plasso(self, lambda_total=1.0, tol=1e-8, max_iter=1000):
        r"""In-place plastic lasso regression.
        See :meth:`plasticnet.solvers.in_place.plasso_` for documentation."""
        plasso_(
            self.beta,
            self._r,
            self.X,
            self.xi,
            lambda_total=lambda_total,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_hpnet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place hard plastic net regression.
        See :meth:`plasticnet.solvers.in_place.hpnet_` for documentation."""
        hpnet_(
            self.beta,
            self._r,
            self.X,
            self.xi,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_spnet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place soft plastic net regression.
        See :meth:`plasticnet.solvers.in_place.spnet_` for documentation."""
        spnet_(
            self.beta,
            self._r,
            self.X,
            self.zeta,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )

    def fit_upnet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place unified plastic net regression.
        See :meth:`plasticnet.solvers.in_place.upnet_` for documentation."""
        upnet_(
            self.beta,
            self._r,
            self.X,
            self.xi,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )
