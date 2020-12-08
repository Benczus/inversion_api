import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion import GAMLPInverter


def init_default_test_inverter():
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
                             crossover_strategy="uniform",
                             bounds=(X.max(axis=0), X.min(axis=0)))
    return inverter


class CrossoverTests(unittest.TestCase):

    def test__one_point_crossover_assert(self):

        parent1 = np.array([1,2,3,4])
        parent2 = np.array([5,6,7,8])
        expected = np.array([1,2,7,8])
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__one_point_crossover(parent1,parent2)
        # then
        np.testing.assert_allclose(actual,expected)

    def test__multi_point_crossover(self):
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        expected = np.array([1,2,7,8,5,6])
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__multi_point_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__uniform_crossover(self):
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        #seeded random uniform array
        expected = np.array([5, 2, 3, 4, 9, 6])
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__uniform_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)

    def test__arithmetic_crossover(self):
        parent1 = np.array([1, 2, 3, 4, 5, 6])
        parent2 = np.array([5, 6, 7, 8, 9, 10])
        # seeded random uniform array
        expected = np.array([3, 4, 5, 6, 7, 8])
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__arithmetic_crossover(parent1, parent2)
        # then
        np.testing.assert_allclose(actual, expected)


class SelectionTests(unittest.TestCase):

    def test__random_selection(self):
        population_size = 50
        individual_size = 8
        max_generaton = 100
        seed = 42
        np.random.seed(42)
        X = np.floor(np.random.random((population_size, individual_size)) * 10)
        y = [np.mean(line) ** 2 for line in X]
        regressor = MLPRegressor(random_state=seed)
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor,
                                 population_size=population_size,
                                 max_generations=max_generaton,
                                 selection_strategy="random",
                                 bounds=(X.max(axis=0), X.min(axis=0)))
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print("Inverted inputs: ", inverted)
        print(np.array(inverted).shape)

    def test__rank_selection(self):
        population_size = 50
        individual_size = 8
        max_generaton = 100
        seed = 42
        np.random.seed(42)
        X = np.floor(np.random.random((population_size, individual_size)) * 10)
        y = [np.mean(line) ** 2 for line in X]
        regressor = MLPRegressor(random_state=seed)
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor,
                                 population_size=population_size,
                                 max_generations=max_generaton,
                                 selection_strategy="rank",
                                 bounds=(X.max(axis=0), X.min(axis=0)))
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print("Inverted inputs: ", inverted)
        print(np.array(inverted).shape)

    def test__tournament_selection(self):
        population_size = 50
        individual_size = 8
        max_generaton = 100
        seed = 42
        np.random.seed(42)
        X = np.floor(np.random.random((population_size, individual_size)) * 10)
        y = [np.mean(line) ** 2 for line in X]
        regressor = MLPRegressor(random_state=seed)
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor,
                                 population_size=population_size,
                                 max_generations=max_generaton,
                                 selection_strategy="tournament",
                                 bounds=(X.max(axis=0), X.min(axis=0)))
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print("Inverted inputs: ", inverted)
        print(np.array(inverted).shape)

    def test__roulette_selection(self):
        population_size = 50
        individual_size = 8
        max_generaton = 100
        seed = 42
        np.random.seed(42)
        X = np.floor(np.random.random((population_size, individual_size)) * 10)
        y = [np.mean(line) ** 2 for line in X]
        regressor = MLPRegressor(random_state=seed)
        regressor.fit(X, y)
        inverter = GAMLPInverter(regressor,
                                 population_size=population_size,
                                 max_generations=max_generaton,
                                 selection_strategy="roulette",
                                 bounds=(X.max(axis=0), X.min(axis=0)))
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        print("Inverted inputs: ", inverted)
        print(np.array(inverted).shape)


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
