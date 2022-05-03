import logging
import pickle
from datetime import datetime

from sklearn.neural_network import MLPRegressor

current_datetime = datetime.now()
ann_logger = logging.getLogger("logger")


class WifiRSSIPropagation:
    name: str
    model: MLPRegressor
    scaler: None

    def __init__(self, name, model, scaler):
        ann_logger.info("Instantiated GA_Inverter method")
        self.name = name
        self.model = model
        self.scaler = scaler

    @staticmethod
    def load_by_name(name):
        ann_logger.info("Instantiated GA_Inverter method")
        (model, scaler) = WifiRSSIPropagation.load_model(
            "model/ann_models/{}".format(name)
        )
        return WifiRSSIPropagation(name, model, scaler)

    @staticmethod
    def load_model(filepath):
        with open(filepath, "rb") as fp:
            return pickle.load(fp)

    @staticmethod
    def save_model(filepath, wifi_rssi_prop):
        with open(filepath, "wb") as fp:
            pickle.dump((wifi_rssi_prop.model, wifi_rssi_prop.scaler), fp)
