import data_tools
import machine_learning_tools
import show_results
import numpy as np
from config import CONF
import matplotlib.pyplot as plt

class Predictor:
    dataset = data_tools.DataSet()
    data_F = data_tools.Functions()
    neural_networks = machine_learning_tools.NeuralNetworks()
    show_results = show_results.Functions()
    data_processed = {}


    def __init__(self):
        pass

    def data_processing(self, data_processing_params):
        """Acquire and process data according to specified method."""
        
        for function_name in data_processing_params.keys():
            
            self.data_processed[function_name] = getattr(self.data_F, function_name) \
                                                 (self.dataset, data_processing_params[function_name])

    def predict(self, models_params, predictor_params):
        self.neural_networks.create_models(self.data_processed, models_params)
        self.neural_networks.predict(self.data_processed, models_params, predictor_params)

    def results(self, params):
        """Show predict results and compare with real signal"""
        self.show_results.plot(self.data_processed, params)

def main():

    predictor = Predictor()

    for method in CONF['method_list']:
        if CONF['load_bearings_data']:
            predictor.dataset.load_bearing_data(predictor.dataset, method.load_data_params)
        
        predictor.data_processing(method.data_processing_params)
        predictor.predict(method.models_params, method.predictor_params)
        #predictor.results(method.show_results_params)

if __name__ == '__main__':
    main()

# Timer
#import time
#start = time.time()
#end = time.time()
#print(end - start)
