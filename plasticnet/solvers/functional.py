import numpy as np

from .in_place import (
    elastic_net_,
    general_plastic_net_,
    hard_plastic_net_,
    lasso_,
    ordinary_least_squares_,
    plastic_lasso_,
    plastic_ridge_,
    ridge_,
    soft_plastic_net_,
    unified_plastic_net_,
)


def ols(X, y, tol=1e-8, max_iter=1000):
    r"""
    Ordinary least squares regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N}||\vec{y}-X\vec{\beta}||_2^2

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

    ordinary_least_squares_(beta, r, X, tol=tol, max_iter=max_iter)

    return beta


def ridge(X, y, lambda_total=1.0, tol=1e-8, max_iter=1000):
    r"""
    Ridge regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \tfrac{1}{2} ||\vec{\beta}||_2^2

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

    ridge_(beta, r, X, lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    return beta


def lasso(X, y, lambda_total=1.0, tol=1e-8, max_iter=1000):
    r"""
    Lasso regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda ||\vec{\beta}||_1

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

    lasso_(beta, r, X, lambda_total=lambda_total, tol=tol, max_iter=max_iter)

    return beta


def enet(X, y, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
    r"""
    Elastic net regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

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

    elastic_net_(
        beta, r, X, lambda_total=lambda_total, alpha=alpha, tol=tol, max_iter=max_iter
    )

    return beta


def gpnet(X, y, xi, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
    r"""
    General plastic net regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

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

    general_plastic_net_(
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


def pridge(X, y, zeta, lambda_total=1.0, tol=1e-8, max_iter=1000):
    r"""
    Plastic ridge regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2

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

    plastic_ridge_(
        beta, r, X, zeta, lambda_total=lambda_total, tol=tol, max_iter=max_iter
    )

    return beta


def plasso(X, y, xi, lambda_total=1.0, tol=1e-8, max_iter=1000):
    r"""
    Plastic lasso regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda ||\vec{\beta}-\vec{\xi}||_1

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

    plastic_lasso_(
        beta, r, X, xi, lambda_total=lambda_total, tol=tol, max_iter=max_iter
    )

    return beta


def hpnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
    r"""
    Hard plastic net regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}||_2^2 \bigr)

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

    hard_plastic_net_(
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


def spnet(X, y, zeta, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
    r"""
    Soft plastic net regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\zeta}||_2^2 \bigr)

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

    soft_plastic_net_(
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


def upnet(X, y, xi, lambda_total=1.0, alpha=0.75, tol=1e-8, max_iter=1000):
    r"""
    Unified plastic net regression.  This function finds the :math:`\vec{\beta}` that minimizes

    .. math::

        \tfrac{1}{2N} ||\vec{y}-X\vec{\beta}||_2^2 + \lambda \bigl( \alpha||\vec{\beta}-\vec{\xi}||_1 + (1-\alpha) \tfrac{1}{2} ||\vec{\beta}-\vec{\xi}||_2^2 \bigr)

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

    unified_plastic_net_(
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
