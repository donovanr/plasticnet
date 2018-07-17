import numpy as np

from .in_place import ols_, enet_


class Regression:
    r"""
    This class encapsulates a regression problem.
    It stores the data matrix :math:`X` and the target :math:`y`, and provides as methods all of the in-place solvers in :mod:`plasticnet.solvers.in_place`.
    It also stores the coefficient vector :math:`\beta`, and the penalized regression target vectors :math:`\xi` (L1 target) and :math:`\zeta` (L2 target).
    Calling any of the fit methods below will update :math:`\beta` in-place.

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.

    Attributes:
        beta (numpy.ndarray): shape (D,) coefficient vector
        xi (numpy.ndarray): shape (D,) L1 coefficient target vector
        zeta (numpy.ndarray): shape (D,) L2 coefficient target vector

    .. warning:: Never set :math:`\beta` by hand.  Always use the :meth:`set_beta` method.
                 It's fine to set :math:`\xi` and :math:`\zeta` by hand.
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
        r"""Sets :math:`\beta` to desired value while also updating the residual vector via :math:`r = y-X\beta`."""
        self.beta = beta
        self._r = self.y - np.dot(self.X, self.beta)

    def fit_ols(self, tol=1e-8, max_iter=1000):
        r"""In-place ordinary least squares.  See :meth:`plasticnet.solvers.in_place.ols_` for documentation."""
        ols_(self.beta, self._r, self.X, tol=tol, max_iter=max_iter)

    def fit_enet(self, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
        r"""In-place elastic net.  See :meth:`plasticnet.solvers.in_place.enet_` for documentation."""
        enet_(
            self.beta,
            self._r,
            self.X,
            lambda_total=lambda_total,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )
