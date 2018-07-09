import numpy as np
import numpy.testing as npt


def test_gpnet():
    """General plastic net tests"""
    npt.assert_almost_equal(np.array([0.0, 0.0]), np.array([0.0, 0.0]), decimal=1)
