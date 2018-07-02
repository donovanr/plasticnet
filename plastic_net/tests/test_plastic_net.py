from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import plastic_net as pnet


def test_cum_gauss():
    """Dummy test function"""
    X = np.random.randn(100000)
    m = np.mean(X)
    npt.assert_almost_equal(0, m, decimal=2)
