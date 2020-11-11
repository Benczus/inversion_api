import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.GAMLPInverter import GAMLPInverter

class CrossoverTests(unittest.TestCase):

    def test__one_point_crossover(self):
        self.fail('Not implemented yet!')

    def test__multi_point_crossover(self):
        self.fail('Not implemented yet!')

    def test__arithmetic_crossover(self):
        self.fail('Not implemented yet!')

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
        max_generaton=10
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor, population_size=population_size, max_generations=max_generaton)
        print(inverter.invert(np.array([0,0])))


    def test_seeded_inversion_exp(self):
        population_size = 50
        individual_size = 8
        max_generaton=100
        seed=42
        np.random.seed(42)
        X= np.floor(np.random.random((population_size, individual_size )) * 10)
        y=[np.mean(line)**2 for line in X]
        regressor=MLPRegressor(random_state=seed)
        regressor.fit(X,y)
        inverter=GAMLPInverter(regressor, population_size=population_size, max_generations=max_generaton, bounds=(X.max(axis=0),X.min(axis=0)))
        true_y=np.mean([n for n in range(1,9)])**2
        print("Inverted inputs: ",inverter.invert([true_y]))
        print("True y value:" ,true_y)
        inverted_values=inverter.invert([true_y])
        print("Predicted y values based on the inverted values:\n", regressor.predict(inverted_values))




if __name__ == '__main__':
    unittest.main()
