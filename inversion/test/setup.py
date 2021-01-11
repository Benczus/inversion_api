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
