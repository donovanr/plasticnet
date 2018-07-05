from __future__ import absolute_import, division, print_function
import numpy as np
from numba import jit


# TODO test whether math is faster than numpy
@jit(nopython=True, nogil=True, cache=True)
def soft_thresh(lam, x):
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

    Args:
        lam (float): threshold value
        x (ndarry): numpy array of floats to be soft thresholded

    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
