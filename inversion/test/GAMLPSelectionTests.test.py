import unittest

from pandas import np
from sklearn.neural_network import MLPRegressor

from inversion import GAMLPInverter
from inversion.test.setup import init_default_test_inverter


class SelectionTests(unittest.TestCase):

    def test__random_selection(self):
        parent = [np.array([1, 2, 3, 4, 5, 6])]
        # seeded random uniform array
        expected = np.array([5, 2, 3, 4, 9, 6])
        inverter = init_default_test_inverter()
        # when
        actual = inverter._GAMLPInverter__random_selection(0, parent)
        # then
        np.testing.assert_allclose(actual, expected)

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
        print("what")
        print("Inverted inputs, RANK SELECTION: ", inverted)
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
