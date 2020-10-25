from typing import Tuple, List

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.MLPInverter import MLPInverter

LOWER_BOUNDS = 0
UPPER_BOUNDS = 1

class GAMLPInverter(MLPInverter):
    population_size: int
    elite_count: int
    mutation_rate: float
    max_generations: int

    def __init__(self,
                 regressor: MLPRegressor,
                 bounds: Tuple[np.ndarray, np.ndarray] = None,
                 population_size : int =100,
                 elite_count : int =10,
                 mutation_rate : float =0.01,
                 max_generations : int =1e4):
        super().__init__(regressor, bounds)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    def invert(self,
               desired_output: np.ndarray) -> List[np.ndarray]:

        population = self._init_ga_population()
        fitness_values = [ self.__fitness(individual, desired_output) for individual in population]
        sorted_fitness, sorted_population = zip(*sorted(zip(fitness_values, population)))
        print(sorted_fitness,sorted_population)

        return []

    def _init_ga_population(self) -> List[np.ndarray]:
        return np.array([
            [np.random.uniform(self.bounds[LOWER_BOUNDS][i], self.bounds[UPPER_BOUNDS][i])
            for i in np.arange(self.regressor.coefs_[0].shape[0])]
            for p in np.arange(self.population_size)
        ])

    def __crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray:
        return []

    def __mutate(self, individual : np.ndarray) -> np.ndarray:
        return []

    def __fitness(self, individual : np.ndarray, desired_output : np.ndarray) -> float:
        return np.sum(
            (self.regressor.predict([individual])
            - desired_output)**2)