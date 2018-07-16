import numpy as np
from numba import jit

from ..utils import math


@jit(nopython=True, nogil=True, cache=False)
def ols_(beta, r, X, tol=1e-8, max_iter=1e3):
    r"""
    ols_(beta, r, X, tol=1e-8, max_iter=1e3)

    Ordinary least squares regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N}||\vec{y}-X\vec{\beta}||_2^2

    Args:
        beta (numpy.ndarray): shape (D,) coefficient vector. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,D) data matrix.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape
    rho = np.ones(D, dtype=np.float64) + tol

    iter_num = 0

    while np.max(rho) > tol and iter_num < max_iter:
        iter_num += 1
        for j in range(D):
            rho[j] = np.dot(X[:, j], r) / N
            r -= rho[j] * X[:, j]
            beta[j] += rho[j]


@jit(nopython=True, nogil=True, cache=True)
def enet_(beta, r, X, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    enet_(beta, r, X, y, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Elastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (D,) coefficient vector. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,D) data matrix.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    lambda1 = alpha * lambda_total
    lambda2 = (1.0 - alpha) * lambda_total

    N, D = X.shape
    beta_old = beta.copy()
    delta_beta = np.ones(D, dtype=np.float64) + tol
    rho = np.ones(D, dtype=np.float64) + tol

    iter_num = 0

    while np.max(delta_beta) > tol and iter_num < max_iter:
        iter_num += 1
        for j in range(D):
            rho[j] = np.dot(X[:, j], r) / N
            beta[j] = math.soft_thresh(lambda1, beta_old[j] + rho[j]) / (1 + lambda2)
            delta_beta[j] = beta[j] - beta_old[j]
            r -= X[:, j] * delta_beta[j]
            beta_old[j] = beta[j]


@jit(nopython=True, nogil=True, cache=True)
def gpnet_(beta, r, X, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    gpnet_(beta, r, X, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    General plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    lambda1 = alpha * lambda_total
    lambda2 = (1.0 - alpha) * lambda_total

    N, D = X.shape
    beta_old = beta.copy()
    delta_beta = np.ones(D, dtype=np.float64) + tol
    rho = np.ones(D, dtype=np.float64) + tol

    iter_num = 0

    while np.max(delta_beta) > tol and iter_num < max_iter:
        iter_num += 1
        for j in range(D):
            rho[j] = np.dot(X[:, j], r) / N
            beta[j] = (
                math.soft_thresh(
                    lambda1,
                    beta_old[j] + rho[j] + lambda2 * zeta[j] - (1 + lambda2) * xi[j],
                )
                / (1 + lambda2)
                + xi[j]
            )
            delta_beta[j] = beta[j] - beta_old[j]
            r -= X[:, j] * delta_beta[j]
            beta_old[j] = beta[j]


@jit(nopython=True, nogil=True, cache=True)
def pridge_(beta, r, X, zeta, lambda_total=1.0, tol=1e-8, max_iter=1e3):
    r"""
    pridge_(beta, r, X, zeta, lambda_total=1.0, tol=1e-8, max_iter=1e3)

    Plastic ridge regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape

    gpnet_(
        beta,
        r,
        X,
        np.zeros(D, dtype=np.float64),
        zeta,
        lambda_total=lambda_total,
        alpha=0.0,
        tol=tol,
        max_iter=max_iter,
    )


@jit(nopython=True, nogil=True, cache=True)
def plasso_(beta, r, X, xi, lambda_total=1.0, tol=1e-8, max_iter=1e3):
    r"""
    plasso_(beta, r, X, xi, lambda_total=1.0, tol=1e-8, max_iter=1e3)

    Plastic lasso regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda ||\vec{\beta}-\vec{\xi}||_1

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape

    gpnet_(
        beta,
        r,
        X,
        xi,
        np.zeros(D, dtype=np.float64),
        lambda_total=lambda_total,
        alpha=1.0,
        tol=tol,
        max_iter=max_iter,
    )


@jit(nopython=True, nogil=True, cache=True)
def hpnet_(beta, r, X, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    hpnet_(beta, r, X, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Hard plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape

    gpnet_(
        beta,
        r,
        X,
        xi,
        np.zeros(D, dtype=np.float64),
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )


@jit(nopython=True, nogil=True, cache=True)
def spnet_(beta, r, X, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    spnet_(beta, r, X, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Soft plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape

    gpnet_(
        beta,
        r,
        X,
        np.zeros(D, dtype=np.float64),
        zeta,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )


@jit(nopython=True, nogil=True, cache=True)
def upnet_(beta, r, X, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    upnet_(beta, r, X, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Unified plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\xi}||_2^2 \bigr)

    Args:
        beta (numpy.ndarray): shape (P,) initial guess for the solution to the regression. modified in-place.
        r (numpy.ndarray): shape (N,) residual, i.e :math:`\vec{r} = \vec{y} - X\vec{\beta}`. modified in-place.
        X (numpy.ndarray): shape (N,P) data matrix.
        xi (numpy.ndarray): shape (P,) target for both L1 and L2 penalties.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Note
        **beta** and **r** are modified in-place.  As inputs, if :math:`\beta = 0`, then it *must* be the case that :math:`r = y`, or the function will not converge to the correct answer.  In general, the inputs **beta** and **r** must be coordinated such that :math:`\vec{r} = \vec{y} - X\vec{\beta}`.
    """

    N, D = X.shape

    gpnet_(
        beta,
        r,
        X,
        xi,
        xi,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )
