import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion import GAMLPInverter
from inversion.test.inverter_init_test import init_default_test_inverter


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


def check_inverted_y(inverted_y_list, true_y):
    n=len(inverted_y_list)
    actual_elements=0
    for inverted_y in inverted_y_list:
      if np.isclose(inverted_y, true_y, 0.018):
            actual_elements+=1
    print(actual_elements)
    print(n)
    print(actual_elements/n)
    if (actual_elements/n)>0.8:
        return True
    else:
        return False


class InversionTests(unittest.TestCase):

    def test_init_invert(self):
        inverter, _ = init_default_test_inverter()
        self.assertIsInstance(inverter, GAMLPInverter)

    def test_seeded_inversion_exp(self):
        inverter, regressor=init_default_test_inverter()
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print(inverted)
        print(true_y)
        inverted_y= regressor.predict(inverted)
        print(inverted_y)

        self.assertEqual(check_inverted_y(inverted_y, true_y), True)
        print("Predicted y values based on the inverted values:\n", regressor.predict(inverted))


if __name__ == '__main__':
    unittest.main()