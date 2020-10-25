import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.GAMLPInverter import GAMLPInverter


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_something_else(self):
        self.assertEqual(True, True)

    def test_init_ga_popullation(self):
        population_size = 10
        individual_size = 3
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        self.assertEqual(True, True)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size)
        initial_population = inverter._init_ga_population()
        self.assertEqual(initial_population.shape, (population_size, individual_size))

    def test_init_invert(self):
        population_size = 10
        individual_size = 3
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        self.assertEqual(True, True)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size)
        initial_population = inverter._init_ga_population()
        # self.assertEqual(initial_population.shape, (population_size, individual_size))
        inverter.invert(np.array([0,0]))

if __name__ == '__main__':
    unittest.main()
