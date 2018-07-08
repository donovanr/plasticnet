import numpy as np
import numpy.testing as npt

from plastic_net.plastic_net.plastic_net import solve_gpnet


def test_gpnet():
    """General plastic net tests"""

   npt.assert_almost_equal(0.0, 0.0, decimal=1)
