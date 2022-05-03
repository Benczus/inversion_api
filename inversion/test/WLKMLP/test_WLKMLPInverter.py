import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor
from WLKMLP.inverter_WLK_init_test import init_default_test_WLK_inverter

from inversion import GAMLPInverter, WLKMLPInverter


class GeneticAlgorithmTests(unittest.TestCase):
    def test_init_WLK(self):
        input_size = 25
        X = np.floor(np.random.random((input_size, input_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]

        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = WLKMLPInverter(
            input_size=input_size, step_size=10, regressor=regressor
        )
        inverter.invert(y)


def check_inverted_y(inverted_y_list, true_y):
    n = len(inverted_y_list)
    actual_elements = 0
    for inverted_y in inverted_y_list:
        if np.isclose(inverted_y, true_y, 0.018):
            actual_elements += 1
    return (actual_elements / n) > 0.8


class InversionTests(unittest.TestCase):
    def test_init_invert(self):
        # given, when
        inverter, _ = init_default_test_WLK_inverter()
        # then
        self.assertIsInstance(inverter, GAMLPInverter)

    def test_seeded_inversion_exp(self):
        # given
        inverter, regressor = init_default_test_WLK_inverter()
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        # when
        inverted_y = regressor.predict(inverted)
        # then
        self.assertEqual(check_inverted_y(inverted_y, true_y), True)


if __name__ == "__main__":
    unittest.main()
