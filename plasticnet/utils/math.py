import numpy as np
from numba import jit


def soft_thresh_nojit(lam, x):
    r"""
    Soft thresholding operator.  Takes a numpy ndarray of floats and returns and element-wise soft-thesholded version.

    Element-wise, the soft-thresholding operator :math:`S_\lambda(x)` is given by:

    .. math::

        S_\lambda(x) =
            \begin{cases}
                x + \lambda, & \text{if} \quad x < -\lambda \\
                0          , & \text{if} \quad -\lambda < x < \lambda \\
                x - \lambda, & \text{if} \quad x > \lambda
            \end{cases}

    where :math:`\lambda` is a scalar tresholding parameter.

    :param float lam: threshold value
    :param ndarry x: numpy array of floats to be soft thresholded
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


@jit(nopython=True, nogil=True, cache=True)
def soft_thresh(lam, x):
    r"""

    soft_thresh(lam, x)

    Soft thresholding operator.  Takes a numpy ndarray of floats and returns and element-wise soft-thesholded version.

    Element-wise, the soft-thresholding operator :math:`S_\lambda(x)` is given by:

    .. math::

        S_\lambda(x) =
            \begin{cases}
                x + \lambda, & \text{if} \quad x < -\lambda \\
                0          , & \text{if} \quad -\lambda < x < \lambda \\
                x - \lambda, & \text{if} \quad x > \lambda
            \end{cases}

    where :math:`\lambda` is a scalar tresholding parameter.

    :param float lam: threshold value
    :param ndarry x: numpy array of floats to be soft thresholded
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
