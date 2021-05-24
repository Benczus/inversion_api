import unittest

import numpy as np

from inversion.test.inverter_init_test import init_default_test_inverter


class CrossoverTests(unittest.TestCase):

    def test__one_point_crossover_assert(self):
        # given
        parent1 = np.array([1, 2, 3, 4])
        parent2 = np.array([5, 6, 7, 8])
        expected = np.array([1, 2, 7, 8])
        inverter, _ = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__one_point_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__multi_point_crossover(self):
        # given
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        expected = np.array([1, 2, 7, 8, 5, 6])
        inverter, _ = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__multi_point_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__uniform_crossover(self):
        # given
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        # seeded random uniform array
        expected = np.array([5, 2, 3, 4, 9, 6])
        inverter, _ = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__uniform_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__arithmetic_crossover(self):
        # given
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        # seeded random uniform array
        expected = np.array([3, 4, 5, 6, 7, 8])
        inverter, _ = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__arithmetic_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
