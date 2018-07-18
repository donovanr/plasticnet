import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def soft_thresh(lam, x):  # pragma: no cover
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

    Args:
        lam (float): threshold value
        x (numpy.ndarray): array of floats to be soft thresholded

    Returns:
        numpy.ndarray: soft thresholded array of floats
    """
    if lam == 0.0:
        return x
    else:
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
