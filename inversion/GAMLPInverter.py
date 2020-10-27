from random import randint
from typing import Tuple, List, Union, Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.MLPInverter import MLPInverter, ga_logger
from util.util import calculate_spherical_coordinates

LOWER_BOUNDS = 0
UPPER_BOUNDS = 1


class GAMLPInverter(MLPInverter):
    '''
    Inverter class for a single Wi-Fi Access Point
    '''
    population_size: int
    elite_count: int
    mutation_rate: float
    max_generations: int

    def __init__(self,
                 regressor: MLPRegressor,
                 bounds: Tuple[np.ndarray, np.ndarray] = None,
                 population_size: int = 100,
                 elite_count: int = 10,
                 # (population_size // 10) if (not population_size // 10 > 0) else (population_size // 2),
                 mutation_rate: float = 0.01,
                 max_generations: int = int(1e4)):
        '''

        :param regressor: The invertible neural network structure
        :param bounds: valid bounds of individual generation used during the invertion
        :param population_size: Size of each population, default= 100
        :param elite_count: number of elites, default= 10
        :param mutation_rate: Rate of mutation in individuals
        :param max_generations: maximum number of generations of individuals
        '''
        super().__init__(regressor, bounds)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    def invert(self,
               desired_output: np.ndarray) -> List[np.ndarray]:
        '''
        Algorithm:
        1. INITIAL FITNESS GENERATION
        2. GENERATE FIRST NEW GENERATION
        3.    GENERATE ELITES
        4.    GENERATE NEW ELEMENTS BESIDES ELITES

        5. CYCLE GENERATE NEW GENERATION UNTIL REACHING GENERATION N
        6.    CROSSOVER
        7.    MUTATION
        8.    EVALUATION
        9. AFTER GENERATION N, THE OUTPUTS ARE IN THE LAST POPULATION
        :param desired_output: The y value to be inverted
        :return: inverted values of self.regressor's desired output
        '''
        population = self.__init_ga_population()
        #print(population)
        fitness_values = [self.__fitness(individual, desired_output) for individual in population]
        sorted_fitness, sorted_population = zip(*sorted(zip(fitness_values, population)))
        elites=sorted_population[0:self.elite_count]
        offsprings=sorted_population[self.elite_count:]
        for _ in range(self.max_generations):
            crossed_offsprings=[self.__crossover(individual, individual) for individual in offsprings] # TODO
            offsprings=[*elites, *crossed_offsprings]
            offsprings = [self.__mutate(individual) for individual in offsprings]
            fitness_values = [self.__fitness(individual, desired_output) for individual in offsprings]
        return offsprings

    def __init_ga_population(self) -> List[List[Union[int, Any]]]:
        ga_logger.info("Started generate_individual method")
        ga_logger.info("Done generate_individual method")
        return [[x := (randint(self.bounds[LOWER_BOUNDS][0], self.bounds[UPPER_BOUNDS][0])),
                 y := randint(self.bounds[LOWER_BOUNDS][1], self.bounds[UPPER_BOUNDS][1]),
                 z := randint(self.bounds[LOWER_BOUNDS][2], self.bounds[UPPER_BOUNDS][2]),
                 x * y,
                 x * y * z,
                 *calculate_spherical_coordinates(x, y, z)]
                for _ in np.arange(self.population_size)
                ]

    def __crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray:
        return parent_1

    def __mutate(self, individual: np.ndarray) -> np.ndarray:
        return individual

    def __fitness(self, individual: np.ndarray, desired_output: np.ndarray) -> float:
        return float(np.sum(
            (self.regressor.predict([individual])
             - desired_output) ** 2))
