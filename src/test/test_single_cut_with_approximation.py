import unittest

import numpy as np

from data_classes import DatasetInfo
from data_utils import create_interpolation_data

class SingleCutWithApproximationTest(unittest.TestCase):

    def test_original_1d_interpolation_data(self):
        dsetinfo = DatasetInfo(data_range=[-1, 1, 1],
                               primary_cut_center=[1.1, 1.2, 1.3, 1.4, 1.5],
                               varying_index=0,
                               )

        dset = create_interpolation_data(dsetinfo)
        # ----expectation----
        expectation = np.array([[0.1, 1.2, 1.3, 1.4, 1.5], [1.1, 1.2, 1.3, 1.4, 1.5]])
        self.assertTrue(np.allclose(dset, expectation))

    def test_approximate_1d_interpolation_data(self):
        dsetinfo = DatasetInfo(data_range=[-1, 1, 1],
                               primary_cut_center=[1.1, 1.2, 1.3],
                               varying_index=0,
                               secondary_cut_center=[2.1, 2.2, 2.3],
                               fixed_index=2
                               )

        dset = create_interpolation_data(dsetinfo)
        # ----expectation----
        expectation = np.array([[1.1, 1.2, 2.3], [2.1, 1.2, 2.3]])
        self.assertTrue(np.allclose(dset, expectation))




if __name__ == "__main__":
    unittest.main()
