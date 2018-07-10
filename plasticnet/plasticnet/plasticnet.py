import numpy as np
from numba import jit

from ..utils import math


@jit(nopython=True, nogil=True, cache=True)
def solve_gpnet(
    beta,
    residual,
    X,
    xi,
    zeta,
    lambda_total=1.0,
    alpha=0.75,
    thresh=1e-8,
    max_iters=100,
):
    r"""
    solve_gpnet(beta, residual, X, xi, zeta, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100)

    General plastic net regression.  This function finds the beta that minimizes

    .. math::

        ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha)||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        residual (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        thresh (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **thresh**.
        max_iters (int): maximum number of update passes through all P elements of **beta**, in case **thresh** is never met.

    Note
        **beta** and **residual** are modified in-place.
    """

    N, D = X.shape

    lambda1 = alpha * lambda_total
    lambda2 = (1.0 - alpha) * lambda_total

    delta_b = np.ones_like(beta) * thresh + 1
    iter_num = 0

    while np.max(delta_b) > thresh and iter_num < max_iters:
        iter_num += 1
        for j in np.random.permutation(D):
            b_new = (
                math.soft_thresh(
                    lambda1,
                    np.dot(X[:, j], residual) / N
                    + lambda2 * zeta[j]
                    - (1 + lambda2) * xi[j]
                    + beta[j],
                )
                / (1.0 + lambda2)
                + xi[j]
            )
            delta_b[j] = b_new - beta[j]
            residual -= X[:, j] * delta_b[j]
            beta[j] = b_new
