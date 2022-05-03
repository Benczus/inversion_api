import numpy as test_initializer
from sklearn.neural_network import MLPRegressor

from inversion import GAMLPInverter


def init_default_test_GA_inverter():
    population_size = 50
    individual_size = 8
    max_generaton = 100
    seed = 42
    test_initializer.random.seed(42)
    X = test_initializer.floor(
        test_initializer.random.random((population_size, individual_size)) * 10
    )
    y = [test_initializer.mean(line) ** 2 for line in X]
    regressor = MLPRegressor(random_state=seed)
    regressor.fit(X, y)
    inverter = GAMLPInverter(
        regressor,
        population_size=population_size,
        max_generations=max_generaton,
        crossover_strategy="uniform",
        bounds=(X.max(axis=0), X.min(axis=0)),
    )
    return inverter, regressor
