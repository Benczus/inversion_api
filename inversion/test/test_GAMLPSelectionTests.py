import unittest
import numpy as np

from inversion.test.inverter_init_test import init_default_test_inverter


class SelectionTests(unittest.TestCase):

    def test__random_selection(self):
        parent = [np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12]), np.array([13, 14, 15, 16, 17, 18 ])]
        # seeded random uniform array
        expected = [np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12])]
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__random_selection(1 , parent)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__rank_selection(self):
        # given
        parent = [np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12]), np.array([13, 14, 15, 16, 17, 18 ])]
        # seeded random uniform array
        expected = [np.array([1, 2, 3, 4, 5, 6]), np.array([13, 14, 15, 16, 17, 18 ])]
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__rank_selection([0,2,1] , parent)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__tournament_selection(self):
        #given
        parent = [np.array([0, 0, 0, 0, 0, 0]), np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12]), np.array([13, 14, 15, 16, 17, 18 ])]
        # seeded random uniform array
        expected = [np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])]
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__tournament_selection([0,2,1] , parent)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__roulette_selection(self):
        # given
        parent = [np.array([0, 0, 0, 0, 0, 0]), np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12]), np.array([13, 14, 15, 16, 17, 18 ])]
        # seeded random uniform array
        expected = [np.array([13, 14, 15, 16, 17, 18 ]), np.array([1, 2, 3, 4, 5, 6])]
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__roulette_selection([0,2,1,3] , parent)
        # then
        np.testing.assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()