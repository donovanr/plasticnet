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
