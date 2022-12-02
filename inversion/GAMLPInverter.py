from typing import Any, List, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPRegressor

from inversion.MLPInverter import MLPInverter

LOWER_BOUNDS = 0
UPPER_BOUNDS = 1


class GAMLPInverter(MLPInverter):
    """
    Inverter class for a single Wi-Fi Access Point
    """

    population_size: int
    elite_count: int
    mutation_rate: float
    max_generations: int

    def __init__(
        self,
        regressor: MLPRegressor,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
        population_size: int = 100,
        elite_count: int = 10,
        mutation_rate: float = 0.1,
        max_generations: int = 500,
        crossover_strategy=None,
        selection_strategy=None,
    ):
        """

        :param regressor: The invertible neural network structure
        :param bounds: valid bounds of individual generation used during the invertion
        :param population_size: Size of each population, default= 100
        :param elite_count: number of elites, default= 10
        :param mutation_rate: Rate of mutation in individuals
        :param max_generations: maximum number of generations of individuals
        """
        super().__init__(regressor, bounds)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

        if crossover_strategy is None or crossover_strategy == "arithmetic":
            self.crossover_strategy = self.__arithmetic_crossover
        elif crossover_strategy == "one":
            self.crossover_strategy = self.__one_point_crossover
        elif crossover_strategy == "multi":
            self.crossover_strategy = self.__multi_point_crossover
        elif crossover_strategy == "uniform":
            self.crossover_strategy = self.__uniform_crossover

        if selection_strategy is None or selection_strategy == "rank":
            self.selection_strategy = self.__rank_selection
        elif selection_strategy == "roulette":
            self.selection_strategy = self.__roulette_selection
        elif selection_strategy == "random":
            self.selection_strategy = self.__random_selection
        elif selection_strategy == "tournament":
            self.selection_strategy = self.__tournament_selection

    def invert(
        self,
        desired_output: np.ndarray,
        early_stopping: bool = True,
        early_stopping_num: int = 5,
        early_stopping_sensitivity: int = 0.1,
    ) -> List[np.ndarray]:
        """
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
        """
        self.logger.info("GAMLPInverter.invert started")
        population = [self._init_ga_population(len(desired_output)) for a in range(self.population_size)]
        fitness_values = []
        if early_stopping:
            early_values = [0, population]
            i = 0
            while i <= self.max_generations and (
                early_values[0] <= early_stopping_num
            ):
                #FOR 2D instead of just passing one, we need a set of inputs and select the best one!
                fitness_values, population = self.run_generation(
                    desired_output, population
                )
                try:
                    if np.allclose(
                        np.sort(early_values[1])[: int((len(population) / 2))],
                        np.sort(population)[: int((len(population) / 2))],
                        early_stopping_sensitivity,
                    ):
                        early_values[0] += 1
                    else:
                        early_values[0] = 0
                    early_values[1] = population
                except Exception as e:
                    early_values[0] = 0
                    early_values[1] = population
                i+=1
        else:
            for _ in range(self.max_generations):
                fitness_values, population = self.run_generation(
                    desired_output, population
                )
        fitness_values, population = self.__sort_by_fitness(fitness_values, population)
        # self.logger.debug("population: ", population)
        self.logger.info("GAMLPInverter.invert stopped")
        return population

    def run_generation(self, desired_output, population):
        fitness_values = [
            self.__fitness(individual, desired_output) for individual in population
        ]
        sorted_fitnesses, sorted_offsprings = self.__sort_by_fitness(
            fitness_values, population
        )
        elites = sorted_offsprings[0 : self.elite_count]
        crossed_mutated_offsprings = []
        for _ in range(len(sorted_offsprings) - self.elite_count):
            parents = self.__selection(sorted_fitnesses, sorted_offsprings)
            crossed_mutated_offsprings.append(
                self.__mutate(self.__crossover(parents[0], parents[1]))
            )
        population = [*elites, *crossed_mutated_offsprings]
        return fitness_values, population

    def _init_ga_population(self, pop_size: int) -> np.ndarray:
        self.logger.info("Started generate_individual method")

        # return [[x := (randint(self.bounds[LOWER_BOUNDS][0], self.bounds[UPPER_BOUNDS][0])),
        #          y := randint(self.bounds[LOWER_BOUNDS][1], self.bounds[UPPER_BOUNDS][1]),
        #          z := randint(self.bounds[LOWER_BOUNDS][2], self.bounds[UPPER_BOUNDS][2]),
        #          x * y,
        #          x * y * z,
        #          *calculate_spherical_coordinates(x, y, z)]
        #         for _ in np.arange(self.population_size)
        #         ]
        initial_pop = np.array(
            [
                [
                    np.random.uniform(
                        self.bounds[LOWER_BOUNDS][i], self.bounds[UPPER_BOUNDS][i]
                    )
                    for i in np.arange(self.regressor.coefs_[0].shape[0])
                ]
                for _ in np.arange(pop_size)
            ]
        )
        self.logger.info("Done generate_individual method")
        return initial_pop

    def __crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> Union[list, Any]:
        return self.crossover_strategy(parent_1, parent_2)

    def __one_point_crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> List[np.ndarray]:
        return list(
            np.append(parent_1[: len(parent_1) // 2], parent_2[len(parent_2) // 2 :])
        )

    def __multi_point_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        return list(
            np.append(
                np.append(
                    parent_1[: len(parent_1) // 3],
                    parent_2[len(parent_2) // 3 : (len(parent_2) // 3) * 2],
                ),
                parent_1[(len(parent_1) // 3) * 2 :],
            )
        )

    def __uniform_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        offspring = []
        for index, element in enumerate(parent_1):
            if np.random.random() > 0.5:
                offspring.append(parent_1[index])
            else:
                offspring.append(parent_2[index])
        return offspring

    def __arithmetic_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray):
        return list((np.array(parent_1) + np.array(parent_2)) / 2)

    # General mutation strategies do not apply to regressor inversion!
    def __mutate(self, individual: np.ndarray, strategy=None) -> np.ndarray:
        self.logger.info("Mutate started")
        if strategy is None:
            for index, element in enumerate(individual):
                if np.random.random() < self.mutation_rate:
                    individual[index] = element * np.random.uniform(0.8, 1.2)
            return individual

        return strategy(individual)

    def __selection(
        self, fitnesses: np.ndarray, population: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.selection_strategy(fitnesses, population)

    def __random_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        return (
            population[np.random.randint(0, len(population) - 1)],
            population[np.random.randint(0, len(population) - 1)],
        )

    def __rank_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        _, sorted_population = self.__sort_by_fitness(fitnesses, population)
        return sorted_population[0], sorted_population[1]

    def __tournament_selection(
        self, fitnesses: np.ndarray, population: List[np.ndarray]
    ):
        indexes = [np.random.randint(0, len(population) - 1) for _ in range(5)]
        fit_ind, pop_ind = [], []
        for index in indexes:
            fit_ind.append(fitnesses[index])
            pop_ind.append((population[index]))
        _, sorted_pop = self.__sort_by_fitness(fit_ind, pop_ind)
        return sorted_pop[0], sorted_pop[1]

    def __roulette_selection(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        parents = [0, 0]
        for i in range(2):
            max_selected = sum(fitnesses)
            pick = np.random.uniform(0, max_selected)
            current = max_selected
            print(current)
            for index, individual in enumerate(population):
                current -= fitnesses[index]
                print(current, pick)
                if current < pick:
                    parents[i] = individual
                    break
        return parents

    def __sort_by_fitness(self, fitnesses: np.ndarray, population: List[np.ndarray]):
        self.logger.info("Sorting by fitness")
        sorted_fitnesses, sorted_population = zip(
            *sorted(zip(fitnesses, population), key=lambda x: x[0])
        )
        return sorted_fitnesses, sorted_population

    def __fitness(self, individual: np.ndarray, desired_output: np.ndarray) -> float:
        return float(
            np.sum((self.regressor.predict(individual) - desired_output) ** 2)
        )
