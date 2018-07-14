import numpy as np
from numba import jit

from .in_place import enet_, gpnet_, hpnet_, ols_, plasso_, pridge_, spnet_, upnet_


@jit(nopython=True, nogil=True, cache=False)
def ols(X, y, tol=1e-8, max_iter=1e3):
    r"""
    ols(X, y, tol=1e-8, max_iter=1e3)

    Ordinary least squares regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N}||\vec{y}-X\cdot\vec{\beta}||_2^2

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    ols_(beta, r, X, tol=tol, max_iter=max_iter)

    return beta


@jit(nopython=True, nogil=True, cache=True)
def enet(X, y, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    enet(X, y, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Elastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,D) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    enet_(
        beta, r, X, lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    return beta


@jit(nopython=True, nogil=True, cache=True)
def gpnet(X, y, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    gpnet(X, y, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    General plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    gpnet_(
        beta,
        r,
        X,
        xi,
        zeta,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    return beta


@jit(nopython=True, nogil=True, cache=True)
def pridge(X, y, zeta, lambda_total=1.0, tol=1e-8, max_iter=1e3):
    r"""
    pridge(X, y, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Plastic ridge regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    pridge_(beta, r, X, zeta, lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    return beta


@jit(nopython=True, nogil=True, cache=True)
def plasso(X, y, xi, lambda_total=1.0, tol=1e-8, max_iter=1e3):
    r"""
    plasso(X, y, xi, lambda_total=1.0, tol=1e-8, max_iter=1e3)

    Plastic lasso regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda ||\vec{\beta}-\vec{\xi}||_1

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    plasso_(beta, r, X, xi, lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    return beta


@jit(nopython=True, nogil=True, cache=True)
def hpnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    hpnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Hard plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        xi (numpy.ndarray): shape (P,) target for L1 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.

    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    hpnet_(
        beta,
        r,
        X,
        xi,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    return beta


@jit(nopython=True, nogil=True, cache=True)
def spnet(X, y, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    spnet(X, y, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Soft plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        zeta (numpy.ndarray): shape (P,) target for L2 penalty.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.
    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    spnet_(
        beta,
        r,
        X,
        zeta,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    return beta


@jit(nopython=True, nogil=True, cache=True)
def upnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3):
    r"""
    upnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1e3)

    Unified plastic net regression.  This function finds the beta that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\cdot\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\xi}||_2^2 \bigr)

    Args:
        X (numpy.ndarray): shape (N,P) data matrix.
        y (numpy.ndarray): shape (N,) target vector.
        xi (numpy.ndarray): shape (P,) target for both L1 and L2 penalties.
        lambda_total (float): must be non-negative. total regularization penalty strength.
        alpha (float): mixing parameter between L1 and L1 penalties. must be between zero and one. :math:`\alpha=0` is pure L2 penalty, :math:`\alpha=1` is pure L1 penalty.
        tol (float): convergence criterion for coordinate descent. coordinate descent runs until the maximum element-wise change in **beta** is less than **tol**.
        max_iter (int): maximum number of update passes through all P elements of **beta**, in case **tol** is never met.
    Returns:
        (numpy.ndarray): shape (D,) coefficient vector.
    """

    N, D = X.shape
    beta = np.zeros(D, dtype=np.float64)
    r = y - np.dot(X, beta)

    upnet_(
        beta,
        r,
        X,
        xi,
        lambda_total=lambda_total,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    return beta
