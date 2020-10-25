from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor

LOWER_BOUNDS = 0
UPPER_BOUNDS = 1

class MLPInverter:
    regressor: MLPRegressor
    bounds: Tuple[np.ndarray, np.ndarray]

    def __init__(self, regressor: MLPRegressor, bounds: Tuple[np.ndarray, np.ndarray] = None):
        self.regressor: regressor
        INPUT_LAYER_SIZE = regressor.coefs_[0].shape[0]
        if(bounds == None or
                len(bounds) != 2 or
                len(bounds[LOWER_BOUNDS]) != INPUT_LAYER_SIZE or
                len(bounds[UPPER_BOUNDS]) != INPUT_LAYER_SIZE):
            self.bounds = (
                np.full(INPUT_LAYER_SIZE, np.ninf),
                np.full(INPUT_LAYER_SIZE, np.inf)
            )
        else:
            self.bounds = bounds

    def predict(self, regressor_inputs: List[np.ndarray]) -> List[np.ndarray]:
        return [self.regressor.predict(regressor_input) for regressor_input in regressor_inputs]

    def invert(self, desired_output: np.ndarray) -> List[np.ndarray]:
        pass

    def score(self, desired_output: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> float:
        return metric(desired_output, self.predict(self.invert(desired_output)))

    def score_r2(self, desired_output: np.ndarray) -> float:
        return self.score(desired_output, lambda expected, actual: np.sum((expected - actual) ** 2))