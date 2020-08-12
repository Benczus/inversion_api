import pickle
from datetime import datetime

from util.util import setup_logger

current_datetime = datetime.now()
ann_logger = setup_logger('ga_invert',
                          "log/ga_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                          current_datetime.day,
                                                          current_datetime.hour))


class WifiRSSIPropagation():
    def __init__(self, scaler, name):
        ann_logger.info("Instantiated GA_Inverter method")
        self.scaler = scaler
        self.name = name
        self.model = WifiRSSIPropagation.load_model("model/ann_models/{}".format(name))

    @staticmethod
    def load_model(filepath):
        with open(filepath, "rb") as fp:
            return pickle.load(fp)

    @staticmethod
    def save_model(filepath, wifirssiprop):
        with open(filepath, "wb") as fp:
            pickle.dump(wifirssiprop, fp)
