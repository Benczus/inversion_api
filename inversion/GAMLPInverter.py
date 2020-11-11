from typing import Tuple, List, Union, Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.MLPInverter import MLPInverter, ga_logger

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
                 mutation_rate: float = 0.1,
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
        2. CYCLE GENERATE NEW GENERATION UNTIL REACHING GENERATION N
        3.  CALCULATE FITNESS VALUE FOR POPULATION
        4.  SELECT BEST PERFORMING INDIVIDUALS BASED ON FITNESS VALUES
        5.  SELECT ELITES AND NON-ELITE OFFSPRINGS
        6.  CROSS OVER NON-ELITE OFFSPRIGS
        7.  CALCULATE FITNESS FOR CROSSED OFFSPRING
        8.  SELECT CROSSED OFFSPRINGS WITH BIGGEST FITNESS
        9.  MUTATE SELECTED OFFSPRINGS
        10. ASSEMBLE NEW POPULATION FROM ELITES AND SELECTED MUTATED, CROSSED OFFSPRINGS
        . AFTER GENERATION N, THE OUTPUTS ARE IN THE LAST POPULATION
        :param desired_output: The y value to be inverted
        :return: inverted values of self.regressor's desired output
        '''
        population = self._init_ga_population()
        for _ in range(self.max_generations):
            fitness_values = [self.__fitness(individual, desired_output) for individual in population]
            sorted_fitnesses, sorted_offsprings = self.__sort_by_fitness(fitness_values, population)
            elites = sorted_offsprings[0:self.elite_count]
            crossed_mutated_offsprings = []
            for i in range(self.population_size - self.elite_count):
                parents = self.__selection(sorted_fitnesses, sorted_offsprings, strategy=self.__rank_selection)
                crossed_mutated_offsprings.append(self.__mutate(self.__crossover(parents[0], parents[1])))
            population = [*elites, *crossed_mutated_offsprings]
        fitness_values, population = self.__sort_by_fitness(fitness_values, population)
        return population

    def _init_ga_population(self) -> np.ndarray:
        ga_logger.info("Started generate_individual method")
        ga_logger.info("Done generate_individual method")
        # return [[x := (randint(self.bounds[LOWER_BOUNDS][0], self.bounds[UPPER_BOUNDS][0])),
        #          y := randint(self.bounds[LOWER_BOUNDS][1], self.bounds[UPPER_BOUNDS][1]),
        #          z := randint(self.bounds[LOWER_BOUNDS][2], self.bounds[UPPER_BOUNDS][2]),
        #          x * y,
        #          x * y * z,
        #          *calculate_spherical_coordinates(x, y, z)]
        #         for _ in np.arange(self.population_size)
        #         ]
        return np.array([
            [np.random.uniform(self.bounds[LOWER_BOUNDS][i], self.bounds[UPPER_BOUNDS][i])
             for i in np.arange(self.regressor.coefs_[0].shape[0])]
            for p in np.arange(self.population_size)])

    def __crossover(self, parent_1: np.ndarray, parent_2: np.ndarray, strategy=None) -> Union[list, Any]:
        if strategy is None:
            return self.arithmetic_crossover(parent_1, parent_2)

        else:
            return strategy(parent_1, parent_2)

    def __one_point_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> List[np.ndarray]:
        return list(np.append(parent_1[:len(parent_1) // 2], parent_2[len(parent_2) // 2:]))

    def __multi_point_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        return list(np.append(np.append(parent_1[:len(parent_1) // 3],
                                        parent_2[len(parent_2) // 3:(len(parent_2) // 3) * 2]),
                              parent_1[(len(parent_1) // 3) * 2:]))

    def __uniform_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        offspring = []
        for index, element in enumerate(parent_1):
            if np.random.random() > 0.5:
                offspring.append(parent_1[index])
            else:
                offspring.append(parent_2[index])
        return offspring

    def arithmetic_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        return list((np.array(parent_1) + np.array(parent_2)) / 2)

    # General mutation strategies do not apply to regressor inversion!
    def __mutate(self, individual: np.ndarray, strategy=None) -> np.ndarray:
        if strategy is None:
            for index, element in enumerate(individual):
                if np.random.random() < self.mutation_rate:
                    individual[index] = element * np.random.uniform(0.8, 1.2)
            return individual
        else:
            return strategy(individual)

    def __selection(self, fitnesses: np.ndarray, population: List[np.ndarray], strategy=None) -> Tuple[
        np.ndarray, np.ndarray]:
        if strategy is None:
            return np.random.choice(population), np.random.choice(population)
        else:
            return strategy(fitnesses, population)

    def __rank_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        sorted_fitnesses, sorted_population = self.__sort_by_fitness(fitnesses, population)
        return sorted_population[0], sorted_population[1]

    # TODO BUGGED
    def __tournament_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        indexes = [np.random.randint(0, len(population) - 1) for i in range(5)]
        fit_ind, pop_ind = [], []
        for index in indexes:
            fit_ind.append(fitnesses[index])
            pop_ind.append((population[index]))
        sorted_fit, sorted_pop = self.__sort_by_fitness(fit_ind, pop_ind)
        return sorted_pop[0], sorted_pop[1]

    def __roulette_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        pass

    def __sort_by_fitness(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        sorted_fitnesses, sorted_population = zip(*sorted(zip(fitnesses, population), key=lambda x: x[0]))
        return sorted_fitnesses, sorted_population

    def __fitness(self, individual: np.ndarray, desired_output: np.ndarray) -> float:
        return float(np.sum(
            (self.regressor.predict([individual])
             - desired_output) ** 2))