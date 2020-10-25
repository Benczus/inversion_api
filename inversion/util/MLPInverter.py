import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List


class MLPInverter:

    def __init__(self, regressor: MLPRegressor):
        self.regressor: regressor

    def invert(self, desired_output: np.ndarray) -> List[np.ndarray]:
        pass
