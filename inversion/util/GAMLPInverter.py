from typing import Tuple, List

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.util.MLPInverter import MLPInverter


class GAMLPInverter(MLPInverter):
    population_size: int
    elite_count: int
    mutation_rate: float
    max_generations: int

    def __init__(self,
                 regressor: MLPRegressor,
                 bounds: Tuple[np.ndarray, np.ndarray] = None,
                 population_size=100,
                 elite_count=10,
                 mutation_rate=0.01,
                 max_generations=1e4):
        super().__init__(regressor, bounds)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    def invert(self,
               desired_output: np.ndarray) -> List[np.ndarray]:
        return []

    def __crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray:
        return []

    def __mutate(self, individual : np.ndarray) -> np.ndarray:
        return []

    def __fitness(self, individual : np.ndarray, desired_output : np.ndarray) -> float:
        return np.sum(
            (self.regressor.predict(individual)
            - desired_output)**2)