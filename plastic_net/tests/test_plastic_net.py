import numpy as np
import numpy.testing as npt

from plastic_net.plastic_net.plastic_net import solve_gpnet
from plastic_net.plastic_net.plastic_net import pnet_soft_thresh

def test_pnet_soft_thresh():
    """Soft thresholding tests"""

    x = np.arange(-10, 11, dtype=float)
    y = np.concatenate([np.arange(-9, 0), np.zeros(3), np.arange(1, 10)])
    npt.assert_almost_equal(pnet_soft_thresh(1, x), y, decimal=10)
    npt.assert_almost_equal(pnet_soft_thresh(np.max(x), x), np.zeros_like(x), decimal=10)

def test_gpnet():
    """General plastic net tests"""
    pass
