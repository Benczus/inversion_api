from inversion import GAMLPInverter
from model import WiFiRSSIPropagation


class WifiRssiPropagationInverter(GAMLPInverter):

    propagation_model : WiFiRSSIPropagation

    def __init__(self,
                 propagation_model : WiFiRSSIPropagation):
        self.propagation_model = propagation_model
        super(WifiRssiPropagationInverter, self).__init__(
            propagation_model.model
        )
