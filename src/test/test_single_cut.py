import unittest
import numpy as np

from data_utils import create_1d_data, create_2d_data
from algorithms import create_interp_1d_funcs, create_interp_2d_funcs, sigma_f1d, sigma_f2d


class SingleCutTest(unittest.TestCase):

    def test_two_variable_funciton(self):
        def f(x):
            return np.array([datum[0] + 2 * datum[1] for datum in x]).reshape(-1, 1)

        cut_center = [1, 2]
        x_train = create_1d_data(x_range=(-5, 6, 0.5), cut_center=cut_center)
        y = f(x_train)

        # -----functions-----
        f0 = cut_center[0] + 2 * cut_center[1]
        f1s = create_interp_1d_funcs(x=x_train, y=y, cut_center=cut_center)

        # -----calculation----
        x_test = [1.2, 0.6]
        y_test = x_test[0] + 2 * x_test[1]
        y_estimate = f0 + sigma_f1d(f0, f1s, x_test)

        # ----result----
        self.assertAlmostEqual(y_test, y_estimate, delta=0.1)

    def test_three_variable_function(self):
        def f(x):
            return np.array([datum[0] ** 2 + datum[1] ** 2 + datum[2] ** 2 for datum in x]).reshape(-1, 1)

        # ----creating training data----
        cut_center = [0, 0, 0]
        x_1d = create_1d_data(x_range=(-3, 4, 0.5), cut_center=cut_center)
        x_2d = create_2d_data(x1_range=(-3, 4, 0.5), x2_range=(-3, 4, 0.5), cut_center=cut_center)
        y_1d = f(x_1d)
        y_2d = f(x_2d)

        # ----creating the interpolation functions----
        f0 = f(np.array([cut_center]))
        f_1d = create_interp_1d_funcs(x_1d, y_1d, cut_center)
        f_2d = create_interp_2d_funcs(x_2d, y_2d, cut_center)

        # ----Testing----
        test_datum = [3.45, -2.2823, 0.2782]
        true_y = f(np.array([test_datum]))
        y_estimate = f0 + sigma_f1d(f0, f_1d, test_datum) + sigma_f2d(f0, f_1d, f_2d, test_datum)

        # ----result----
        self.assertAlmostEqual(true_y, y_estimate, delta=0.1)

    def test_five_variable_function(self):
        def f(x):
            return np.array(
                [datum[0] * 2 - datum[1] ** 2 + datum[2] * datum[0] + datum[3] / 2 + datum[4] * 3 for datum in x]). \
                reshape(-1, 1)

        # ----creating training data----
        cut_center = [-0.1, 0.4, -0.3, 0.2, 0.28]
        x_1d = create_1d_data(x_range=(-4, 10, 0.5), cut_center=cut_center)
        x_2d = create_2d_data(x1_range=(-4, 10, 0.5), x2_range=(-4, 10, 0.5),
                              cut_center=cut_center)
        y_1d = f(x_1d)
        y_2d = f(x_2d)

        # ----creating the interpolation functions----
        f0 = f(np.array([cut_center]))
        f_1d = create_interp_1d_funcs(x=x_1d, y=y_1d, cut_center=cut_center)
        f_2d = create_interp_2d_funcs(x=x_2d, y=y_2d, cut_center=cut_center)

        # ----testing----
        test_datum = [-2.59, 1, -2.28, -3.29, 0.17]
        y_true = f(np.array([test_datum]))
        sigma_1d = sigma_f1d(f0, f_1d, test_datum)
        sigma_2d = sigma_f2d(f0, f_1d, f_2d, test_datum)
        y_estimate = f0 + sigma_f1d(f0, f_1d, test_datum) + sigma_f2d(f0, f_1d, f_2d, test_datum)

        # ----result----
        self.assertAlmostEqual(y_true, y_estimate, delta=0.1)
        print(f'f0: {f0}')
        print(f'sigma 1d: {sigma_1d}')
        print(f'sigma 2d: {sigma_2d}')
        print(f'True y: {y_true}')
        print(f'Estimated y: {y_estimate}')


if __name__ == "__main__":
    unittest.main()
