import numpy as np
import numpy.testing as npt

from ..utils.math import soft_thresh


def test_soft_thresh():
    r"""Soft thresholding tests"""

    x = np.arange(-10, 11, dtype=float)
    y = np.concatenate([np.arange(-9, 0), np.zeros(3), np.arange(1, 10)])
    npt.assert_almost_equal(soft_thresh(0, x), x, decimal=10)
    npt.assert_almost_equal(soft_thresh(1, x), y, decimal=10)
    npt.assert_almost_equal(soft_thresh(np.max(x), x), np.zeros_like(x), decimal=10)
