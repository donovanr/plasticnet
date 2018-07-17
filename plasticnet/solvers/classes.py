import numpy as np

from .in_place import ols_, enet_


class Ols:
    r"""
    Ordinary least squares regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N}||\vec{y}-X\vec{\beta}||_2^2

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Methods:
        fit(): calls in-place :meth:`plasticnet.solvers.in_place.ols_`
    """

    def __init__(self, X, y, tol=1e-8, max_iter=1e3):
        self.X = X
        self.y = y

        self.tol = tol
        self.max_iter = max_iter

        N, D = self.X.shape
        self.beta = np.zeros(D, dtype=np.float64)
        self.r = self.y - np.dot(self.X, self.beta)

    def fit(self):
        ols_(self.beta, self.r, self.X, tol=self.tol, max_iter=self.max_iter)


class Enet:
    r"""
    Elastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Methods:
        fit(): calls in-place :meth:`plasticnet.solvers.in_place.enet_`
    """

    def __init__(self, X, y, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
        self.X = X
        self.y = y

        self.lambda_total = lambda_total
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

        N, D = self.X.shape
        self.beta = np.zeros(D, dtype=np.float64)
        self.r = self.y - np.dot(self.X, self.beta)

    def fit(self):
        enet_(
            self.beta,
            self.r,
            self.X,
            lambda_total=self.lambda_total,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
        )
