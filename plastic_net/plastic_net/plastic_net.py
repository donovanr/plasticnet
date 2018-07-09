import numpy as np
import networkx as nx
from numba import jit, jitclass, float32, int64

from ..utils import soft_thresh


def pnet_soft_thresh(lam, x):
    """Just wraps the import to test it"""
    return soft_thresh(lam, x)

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


# jit class spec
spec = [
    ("X", float32[:, :]),
    ("y", float32[:]),
    ("xi", float32[:]),
    ("zeta", float32[:]),
    ("beta", float32[:]),
    ("residual", float32[:]),
    ("thresh", float32),
    ("max_iters", int64),
]


# jitted timepoint class that will live on graph nodes
@jitclass(spec)
class TimePoint(object):
    """This class encapsulates a single (X,y) data set, e.g. one time-point"""

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.beta = np.zeros(X.shape[1]).astype(np.float32)
        self.residual = y.copy()
        self.xi = np.zeros_like(self.beta)
        self.zeta = np.zeros_like(self.beta)
        self.zero = np.zeros_like(self.beta)

    def solve_OLS(self, thresh=1e-8, max_iters=100):
        """OLS regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=0,
            alpha=0,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_ridge(self, lambda_total=1.0, thresh=1e-8, max_iters=100):
        """Ridge regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + lambda||beta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=0,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_lasso(self, lambda_total=1.0, thresh=1e-8, max_iters=100):
        """Lasso regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + lambda||beta-xi||_1 + (1-alpha)*lambda||beta-zeta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=1,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_enet(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Elastic net regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + alpha*lambda||beta||_1 + (1-alpha)*lambda||beta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=alpha,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_pridge(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Plastic ridge regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + lambda||beta-zeta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=0,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_plasso(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Plastic Lasso regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + lambda||beta-xi||_1"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=1,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_hpnet(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Hard plastic net regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + lambda||beta-xi||_1 + (1-alpha)*lambda||beta||_2^2"""
        solve_gpnet(
            self.X,
            self.xi,
            self.zero,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=alpha,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_spnet(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Soft plastic net regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + alpha*lambda||beta||_1 + (1-alpha)*lambda||beta-zeta||_2^2"""
        solve_gpnet(
            self.X,
            self.zero,
            self.zeta,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=alpha,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_upnet(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Unified plastic net regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + alpha*lambda||beta-xi||_1 + (1-alpha)*lambda||beta-xi||_2^2"""
        solve_gpnet(
            self.X,
            self.xi,
            self.xi,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=alpha,
            thresh=thresh,
            max_iters=max_iters,
        )

    def solve_gpnet(self, lambda_total=1.0, alpha=0.75, thresh=1e-8, max_iters=100):
        """Hard plastic net regression.  This function finds the beta that minimizes
        ||y-X@beta||_2^2 + alpha*lambda||beta-xi||_1 + (1-alpha)*lambda||beta-zeta||_2^2"""
        solve_gpnet(
            self.X,
            self.xi,
            self.zeta,
            self.beta,
            self.residual,
            lambda_total=lambda_total,
            alpha=alpha,
            thresh=thresh,
            max_iters=max_iters,
        )


# nx code to get nth neighbors, inclusive
def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length <= n]


# methods for computing bias targets
def get_xi(G, n, method="mean"):
    if method == "mean":
        info = [G.nodes[m]["data"].beta for m in G.neighbors(n)]
        mean = np.mean(info, axis=0).astype(np.float32)
        return mean
    elif method == "mean of nonzero neighbors":
        # for each gene, if neighboring time points are both nonzero, use their mean, otherwise use zero
        neighbor_nz = np.array(
            [G.nodes[m]["data"].beta != 0 for m in G.neighbors(n)]
        ).astype(np.float32)
        neighbor_betas = np.array(
            [G.nodes[m]["data"].beta for m in G.neighbors(n)]
        ).astype(np.float32)
        mask = np.prod(neighbor_nz, axis=0)
        xi = np.mean(neighbor_betas, axis=0)
        return mask * xi
    elif method == "mean of nonzero neighbors minus loners":
        # for each gene, if neighboring time points are both nonzero, use their mean, if both are zero, use minus what you are, otherwise use zero
        neighbor_nz = np.array(
            [G.nodes[m]["data"].beta != 0 for m in G.neighbors(n)]
        ).astype(np.float32)
        neighbor_az = np.array(
            [G.nodes[m]["data"].beta == 0 for m in G.neighbors(n)]
        ).astype(np.float32)
        neighbor_betas = np.array(
            [G.nodes[m]["data"].beta for m in G.neighbors(n)]
        ).astype(np.float32)
        nz_mask = np.prod(neighbor_nz, axis=0)
        az_mask = np.prod(neighbor_az, axis=0)
        neighbor_mean = np.mean(neighbor_betas, axis=0)
        beta_self = G.nodes[n]["data"].beta
        return nz_mask * neighbor_mean - 0.5 * az_mask * beta_self
    elif method == "mean of nonzero next nearest neighbors":
        mask = np.prod(
            [G.nodes[m]["data"].beta != 0 for m in neighborhood(G, n, 2) if m != n],
            axis=0,
        ).astype(np.float32)
        xi = np.mean([G.nodes[m]["data"].beta for m in G.neighbors(n)], axis=0).astype(
            np.float32
        )
        return mask * xi
    elif method == "mean of neighbors masked by median":
        info = [G.nodes[m]["data"].beta for m in neighborhood(G, n, 1)]
        mean = np.mean(info, axis=0).astype(np.float32)
        median_mask = (np.median(info, axis=0).astype(np.float32) > 0).astype(
            np.float32
        )
        return mean * median_mask
    elif method == "mean of next neighbors masked by median":
        info = [G.nodes[m]["data"].beta for m in neighborhood(G, n, 2)]
        mean = np.mean(info, axis=0).astype(np.float32)
        median_mask = (np.median(info, axis=0).astype(np.float32) > 0).astype(
            np.float32
        )
        return mean * median_mask
