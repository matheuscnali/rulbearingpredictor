"""
Predictors based on papers.
#1 = Mao, W., He, J., Tang, J. and Li, Y., 2018. Predicting remaining useful life of
rolling bearings based on deep feature representation and long short-term memory
neural network. Advances in Mechanical Engineering, 10(12), p.1687814018817184.
"""

from collections import OrderedDict
import torch.nn as nn


class Method:

    def __init__(self, name, load_data_params, data_processing_params, model_params, show_results_params):
        self.name = name
        self.data_processing_params = data_processing_params
        self.model_params = model_params
        self.show_results_params = show_results_params
        self.load_data_params = load_data_params

def generate():

    method_list = []
                        
    conf = {
        'predictor_name': 'Data driven predictor based on #1',
        'dataset_name': 'PHM',
        'method_list': method_list,
        'processing_unit': 'CPU',
        'load_bearings_data': True
    }

    def method1():

        name = '#1'

        """                    Data processing functions parameters                    """

        vibration_signal = 'vib_horizontal'
        
        hht_marginal_spec = {
            'bearings': [0], # Bearings to be processed by this function. See PHM_dataset in data_tools.py to get bearings name.
            'sampling_frequency': 25600, # 25.6 KHz
            'vibration_signal': vibration_signal,
        }

        bearings_fft = {
            'bearings': [0],
            'sampling_frequency': 25600, # 25.6 KHz
            'vibration_signal': vibration_signal
        }

        health_assessment = {
            'bearings': [0],
            'vibration_signal': vibration_signal,
            'norm_interval': [-1, 1],
            'max_qty': 2,
            'base_values_chunk_percentage': [0, 10],
            'hankel_window_size': 4,
            'smoothing_window_size': 7,
            'debug': True
        }

        rms = {
            'bearings': [0],
            'vibration_signal': vibration_signal,
        }

        load_data_params = {
            'bearings': [0],
            'file_chunk_percentage': [0, 100]
        }

        data_processing_params = OrderedDict([
            # ('bearings_fft', bearings_fft),
            ('hht_marginal_spectrum', hht_marginal_spec),
            ('health_assessment', health_assessment),
            ('rms', rms)
        ])

        """                    Models parameters                    """

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) - Default by PyTorch
        cnn_layers = OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)),
            ('relu1', nn.ReLU()),
        ])

        lstm_layers = OrderedDict([])

        model_params = {
            'cnn':cnn_layers,
            'lstm':lstm_layers
        }

        """                    Show results parameters                    """

        show_results_params = {
            'results_to_show': ['hht_marginal_spectrum', 'health_assessment']
        }    
        

        return [name, load_data_params, data_processing_params, model_params, show_results_params]

    conf['method_list'].append(Method(*method1()))

    return conf

CONF = generate()
