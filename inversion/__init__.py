from datetime import datetime

from inversion.GAMLPInverter import GAMLPInverter
from inversion.MLPInverter import MLPInverter
from inversion.WifiRssiPropagationInverter import WifiRssiPropagationInverter
import logging
import os

__all__ = [
    'GAMLPInverter',
    'MLPInverter'
]

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
if not os.path.exists('log'):
    os.makedirs('log')
current_datetime = datetime.now()
fh = logging.FileHandler("log/{}_{}_{}_{}.log".format(current_datetime.year,
                                                                  current_datetime.month,
                                                                  current_datetime.day,
                                                                  current_datetime.hour) )
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
