import datetime
from util import setup_logger

current_datetime = datetime.datetime.now()
ann_logger = setup_logger('ga_invert',
                      "log/ga_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month, current_datetime.day,
                                                    current_datetime.hour))


class ANN_component():
    def __init__(self, model, scaler):
        ann_logger.info("Instantiated GA_Inverter method")
        self.model= model
        self.scaler=scaler


    