import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion import GAMLPInverter


class GeneticAlgorithmTests(unittest.TestCase):

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


class InversionTests(unittest.TestCase):

    def test_init_invert(self):
        population_size = 20
        individual_size = 8
        max_generaton = 10
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size, max_generations=max_generaton)
        print(inverter.invert(np.array([0, 0])))

    def test_seeded_inversion_exp(self):
        population_size = 50
        individual_size = 8
        max_generaton = 100
        seed = 42
        np.random.seed(42)
        X = np.floor(np.random.random((population_size, individual_size)) * 10)
        y = [np.mean(line) ** 2 for line in X]
        regressor = MLPRegressor(random_state=seed)
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size, max_generations=max_generaton,
                                 bounds=(X.max(axis=0), X.min(axis=0)))
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print("Inverted inputs: ", inverted)
        print(np.array(inverted).shape)
        print("True y value:", true_y)
        print("Predicted y values based on the inverted values:\n", regressor.predict(inverted))


if __name__ == '__main__':
    unittest.main()
