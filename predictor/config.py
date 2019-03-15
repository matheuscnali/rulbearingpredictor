"""
Predictors based on papers.
#1 = Mao, W., He, J., Tang, J. and Li, Y., 2018. Predicting remaining useful life of
rolling bearings based on deep feature representation and long short-term memory
neural network. Advances in Mechanical Engineering, 10(12), p.1687814018817184.
"""

from collections import OrderedDict
import torch.nn as nn
import torch


class Method:

    def __init__(self, name, load_data_params, data_processing_params, models_params, predictor_params, show_results_params):
        self.name = name
        self.data_processing_params = data_processing_params
        self.models_params = models_params
        self.predictor_params = predictor_params
        self.show_results_params = show_results_params
        self.load_data_params = load_data_params

def generate():

    method_list = []
                        
    conf = {
        'predictor_name': 'Data driven predictor based on #1',
        'dataset_name': 'PHM',
        'method_list': method_list,
        'processing_unit': 'CPU',
    }

    def method1():

        name = '#1'

        """                    Data processing functions parameters                    """

        vibration_signal = 'vib_horizontal'
        
        hht_marginal_spectrum = {
            'bearings': [0], # Bearings to be processed by this function. See PHM_dataset in data_tools.py to get bearings name.
            'sampling_frequency': 25600, # 25.6 KHz
            'vibration_signal': vibration_signal,
            'imfs_qty': -4 # Using MAX number of imfs - 1.
        }

        bearings_fft = {
            'bearings': [0],
            'sampling_frequency': 25600,
            'vibration_signal': vibration_signal
        }

        health_assessment = {
            'bearings': [0, 1, 2, 3, 4, 5, 6],
            'vibration_signal': vibration_signal,
            'norm_interval': [-1, 1],
            'max_qty': 2,
            'base_values_chunk_percentage': [0, 2],
            'hankel_window_size': 10,
            'smoothing_window_size': 9,
            'manual_threshold': 0.9,
            'correlation_coefficient_method': 'base_values_mean' # or 'correlation_coefficient_values_mean'
        }

        rms = {
            'bearings': [0, 1, 2, 3, 4, 5, 6],
            'smoothing_window_size': 7,
            'vibration_signal': vibration_signal,
        }

        load_data_params = {
            'load_data': True,
            'bearings': [0, 1, 2, 3, 4, 5, 6],
            'file_chunk_percentage': [0, 100]
        }

        data_processing_params = OrderedDict([
            # ('bearings_fft', bearings_fft),
            ('hht_marginal_spectrum', hht_marginal_spectrum),
            ('health_assessment', health_assessment),
            ('rms', rms)
        ])

        """                    Models parameters                    """

        deep_features_qty = 25

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) - Default by PyTorch
        cnn_layers = OrderedDict([
            ('conv1', [1, 64, 2, 1]),    #in_channels, out_channels, kernel_size, stride.
            ('pool', [2]),               #kernel_size.
            ('conv2', [64, 128, 2, 1]),  #in_channels, out_channels, kernel_size, stride.
            ('linear1', [3968, deep_features_qty]),
            ('linear2', [deep_features_qty, 2]),    
        ])

        lstm_layers = OrderedDict([
            ('input', [25]),
            ('hidden', [1]),
            ('batch_first', True),
            ()
        ])

        models_params = {
            'cnn': cnn_layers,
            'cnn_epochs': 30,
            'cnn_batch_size': 20,
            'lstm': lstm_layers 
        }

        """                     Predictor parameters                        """

        predictor_params = {
            'bearings': [0],
            'hht_cnn_shape': [1, 10, 128],
            'return_cnn_model': False, # If you want to recover the model to train more, set False.
            'cuda_available': False #torch.cuda.is_available()
        }

        """                    Show results parameters                    """

        show_results_params = {
            'results_to_show': ['hht_marginal_spectrum', 'health_assessment']
        }    
        

        return [name, load_data_params, data_processing_params, models_params, predictor_params, show_results_params]

    conf['method_list'].append(Method(*method1()))

    return conf

CONF = generate()
