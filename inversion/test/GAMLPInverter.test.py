import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.GAMLPInverter import GAMLPInverter


class MyTestCase(unittest.TestCase):

    def test_init_ga_popullation(self):
        population_size = 20
        individual_size = 8
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size)
        initial_population = inverter._init_ga_population()

        self.assertEqual(initial_population.shape, (population_size, individual_size))

    def test_init_invert(self):
        population_size = 20
        individual_size = 8
        max_generaton=10
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size, max_generations=10)
        print(inverter.invert(np.array([0,0])))

if __name__ == '__main__':
    unittest.main()
