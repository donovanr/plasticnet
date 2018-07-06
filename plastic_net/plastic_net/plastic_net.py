import numpy as np
from numba import jit

from ..utils.math import soft_thresh


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
    """
    General plastic net regression.  This function finds the beta that minimizes
    ||y-X@beta||_2^2 + alpha*lambda||beta-xi||_1 + (1-alpha)*lambda||beta-zeta||_2^2

    Parameters
    ----------
    beta : Numpy array
        shape (P,) initial guess for the solution to the regression
        modified in-place
    residual : Numpy array
        shape (N,) residual, i.e residual = y - X@beta
        modified in-place
    X : Numpy array
        shape (N,P) data matrix
    xi : Numpy array
        shape (P,) target for L1 penalty
    zeta : Numpy array
        shape (P,) target for L2 penalty
    lambda_total : float
        must be non-negative. total regularization penalty strength
    alpha : float
        must be between zero and one.  mixing parameter between L1 and L1 penalties.
        alpha=0 is pure L2 penalty, alpha=1 is pure L1 penalty
    thresh : float
        convergence criterion for coordinate descent
        coordinate descent runs until the bigest element-wise change in beta is less than thresh
    max_iters : int
        maximum number of update passes through all P elements of beta, in case thesh is never met

    Modifies
    --------
    beta
    residual
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
                soft_thresh(
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
