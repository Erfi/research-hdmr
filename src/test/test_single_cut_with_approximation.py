import unittest

import numpy as np

from data_classes import DatasetInfo
from data_utils import create_interpolation_data, create_data_dict

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


    def test_create_data_dictionary(self):
        cc1 = [0,0,0]
        cc2 = [1,1,1]
        data_range = [-1,2, 1]
        def f(x):
            return np.array([datum[0] * np.sin(datum[1]) + datum[1] + datum[2] for datum in x]).reshape(-1, 1)

        data_dict = create_data_dict(data_range=data_range,
                                     f=f,
                                     primary_cut_center=cc1,
                                     secondary_cut_center=cc2)
        self.assertEqual(len(data_dict), 9)





if __name__ == "__main__":
    unittest.main()
