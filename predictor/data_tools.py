import math
import os
import pickle
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import collections

from matplotlib.pyplot import magnitude_spectrum
from pyhht.emd import EMD
import pyhht.utils
from scipy.optimize import curve_fit
from scipy.linalg import svdvals
from scipy.linalg import hankel
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from scipy.misc import derivative
from collections import OrderedDict
from numpy.fft import fft, fftfreq, ifft
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval

from config import CONF


class DataSet:

    bearings_files = {}
    dir_path = None
    bearings = None
    files_path = None
    files_qty = None
    current_dir = os.getcwd()
    next_bearing = True

    PHM_dataset = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5',
                   'Bearing1_6', 'Bearing1_7', 'Bearing2_1', 'Bearing2_2', 'Bearing2_3',
                   'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7', 'Bearing3_1',
                   'Bearing3_2', 'Bearing3_3']
    
    def __init__(self):
        if CONF['dataset_name'] == 'PHM':
            self.dir_path = '/predictor/data/PHM_Vibration/'

    def find_missing_data(self, data, file_path):
        vib_vert_checkings = data['vib_vertical'].isnull()
        vib_hor_checkings = data['vib_horizontal'].isnull()
        
        for check in vib_vert_checkings:
            if check:
                print('Found missing data in:', file_path)
        for check in vib_hor_checkings:
            if check:
                print('Found missing data in:', file_path)

    def save_processed_data(self, data, processed_data_path):

        processed_data_path = self.current_dir + '/predictor/data/Processed_Data/' + processed_data_path

        with open(processed_data_path, 'wb') as file:
            pickle.dump(data, file)

    def load_processed_data(self, dataset, processed_data_path):

        processed_data_path = dataset.current_dir + '/predictor/data/Processed_Data/' + processed_data_path

        if(os.path.isfile(processed_data_path)):
            with open(processed_data_path, 'rb') as file:
                return True, pickle.load(file)
        return False, None

    def load_bearing_data(self, dataset, params):

        processed_data_path = '/bearings_files/bearings_files_' + str(params['file_chunk_percentage'][0]) + '_' +str(params['file_chunk_percentage'][1])

        dataset.bearings_files = dataset.load_processed_data(dataset, processed_data_path)
        bearings_not_loaded = params['bearings']

        if dataset.bearings_files[0]:
            dataset.bearings_files = dataset.bearings_files[1]
            bearings_loaded = list(map(int, list(dataset.bearings_files.keys())))
            bearings_not_loaded = [x for x in params['bearings'] if x not in bearings_loaded]
            params['bearings'] = bearings_not_loaded

            if bearings_not_loaded == []:
                return
        else:
            dataset.bearings_files = {}
        
        dataset.read_data(dataset, params)

    def read_data(self, dataset, params):

        processed_data_path = 'bearings_files/bearings_files_' + str(params['file_chunk_percentage'][0]) + '_' + str(params['file_chunk_percentage'][1])
       
        for current_bearing in params['bearings']:
            bearing_files = []

            bearing_path = dataset.current_dir + str(dataset.dir_path + dataset.PHM_dataset[current_bearing])
            dataset.files_path = [bearing_path + '/' + s for s in sorted(os.listdir(bearing_path))]
            dataset.files_qty = len(dataset.files_path)
            
            # Resizing files_path.
            ini = int(params['file_chunk_percentage'][0]*dataset.files_qty // 100)
            end = int(params['file_chunk_percentage'][1]*dataset.files_qty // 100)
            dataset.files_path = dataset.files_path[ini:end]
            dataset.files_qty = len(dataset.files_path)

            for file_path in dataset.files_path:
                with open(file_path, 'r') as file:
                        # Removed column 3 (micro-seconds).
                        read_data = pd.read_csv(file, usecols=[0, 1, 2, 4, 5], names=['hour', 'min', 'sec', 'vib_horizontal', 'vib_vertical'], header=None, float_precision='high')
                        bearing_files.append(read_data)
                            
            dataset.bearings_files[str(current_bearing)] = bearing_files
            
        # Saving bearing files.
        dataset.save_processed_data(dataset.bearings_files, processed_data_path)

class Functions:

    def bearings_fft(self, dataset, params):
        processed_data_path = 'bearings_fft/bearings_fft'
        bearings_fft = dataset.load_processed_data(dataset, processed_data_path)
        bearings_not_processed = params['bearings']

        if bearings_fft[0]:
            bearings_fft = bearings_fft[1]

            bearings_fft_processed = list(map(int, list(bearings_fft.keys())))
            bearings_fft_not_processed = [x for x in params['bearings'] if x not in bearings_fft_processed]

            if bearings_fft_not_processed == []:
                return bearings_fft
            
        else:
            bearings_fft = OrderedDict()
        
        for current_bearing in bearings_not_processed:      
            bearing_files = dataset.bearings_files[str(current_bearing)]
            bearing_fft = []

            # Calculating fft for each data file.
            fs = params['sampling_frequency']
            for bearing_file in bearing_files:
                data = bearing_file[params['vibration_signal']].values
                
                N = len(data)
                #freqs = fftfreq(N)*(N/fs)
                freqs = np.arange(N)*(fs/N)
                freqs = freqs[0:int(N//2)]
                mask = freqs > 0

                fft_vals = fft(data)
                fft_theo = 2.0*np.abs(fft_vals/N)
                fft_theo = fft_theo[0:int(N//2)]

                bearing_fft.append([freqs[mask], fft_theo[mask]])
            
            bearings_fft[str(current_bearing)] = bearing_fft
        
        dataset.save_processed_data(bearings_fft, processed_data_path)

        return bearings_fft

    def hht_marginal_spectrum(self, dataset, params):

        processed_data_path = 'hht_marginal_spectrum/hht_marginal_spectrum'
        bearings_marginal_spectrum = dataset.load_processed_data(dataset, processed_data_path)
        bearings_not_processed = params['bearings']
        
        if bearings_marginal_spectrum[0]:
            bearings_marginal_spectrum = bearings_marginal_spectrum[1]

            bearings_processed = list(map(int, list(bearings_marginal_spectrum.keys())))
            bearings_not_processed = [x for x in params['bearings'] if x not in bearings_processed]

            if bearings_not_processed == []:
                return bearings_marginal_spectrum

        else:
            bearings_marginal_spectrum = {}

        for current_bearing in bearings_not_processed:
            imfs_files = []
            bearing_marginal_spectrum = []
            bearing_files = dataset.bearings_files[str(current_bearing)]

            # Calculating IMFs for each data file.
            for bearing_file in bearing_files:      
                data = bearing_file[params['vibration_signal']].values
                decomposer = EMD(data)
                imfs_files.append(decomposer.decompose())
            
            # Getting the frequency bins.
            N = len(data); fs = params['sampling_frequency']; freq_bins_step = fs/N
            freq_bins = np.fft.fftfreq(N)[0:N//2]*fs # Timestep = 1. 

            # Calculating Hilbert transform for each IMF.
            imfs_ht_files = []

            for imfs_file in imfs_files:
                imfs_ht_files.append(hilbert(imfs_file))

            # Calculating instantaneous frequency of each data.
            imfs_freqs_files = []
            for imfs_ht_file in imfs_ht_files:
                imfs_freqs_file = []
                for imf_ht_file in imfs_ht_file:
                    imfs_freqs_file.append(pyhht.utils.inst_freq(imf_ht_file)[0] * fs) # [0] to select the frequencies. * fs because the inst_freq return normalized freqs.
                imfs_freqs_files.append(imfs_freqs_file)

            # Calculating absolute value.
            N = len(imfs_ht_file[0])
            imfs_envelope_files = np.abs(imfs_ht_files) / N

            # Putting frequencies into the frequency bins and computing Hilbert Marginal Spectrum.
            imfs_envelope_files_bins = [] 
            for imfs_freqs_file, imfs_envelope_file in zip(imfs_freqs_files, imfs_envelope_files):
                imfs_envelope_file_bins = []
                for imf_freqs_file, imf_envelope_file in zip(imfs_freqs_file, imfs_envelope_file):
                    imfs_envelope_file_ = np.zeros(N//2) 
                    
                    bin_index = [int(freq // freq_bins_step) for freq in imf_freqs_file]
                    
                    for index, abs_val in zip(bin_index, imf_envelope_file):
                        imfs_envelope_file_[index] += abs_val
                
                    imfs_envelope_file_bins.append(imfs_envelope_file_)

                imfs_envelope_files_bins.append(imfs_envelope_file_bins)

            # Summing Hilbert Marginal Spectrum of [0:params['imfs_qty]] imfs.
            for imfs_envelope_file_bins in imfs_envelope_files_bins:
                 bearing_marginal_spectrum.append([sum(x) for x in zip(*imfs_envelope_file_bins[0:params['imfs_qty']])])

            bearings_marginal_spectrum[str(current_bearing)] = [freq_bins, bearing_marginal_spectrum]
            
        dataset.save_processed_data(bearings_marginal_spectrum, processed_data_path)

        return bearings_marginal_spectrum

    def health_assessment(self, dataset, params):

        processed_data_path = 'health_assessment/health_assessment'
        bearings_health_data = dataset.load_processed_data(dataset, processed_data_path)
        bearings_not_processed = params['bearings']

        # Checking data loaded.
        if bearings_health_data[0]:
            bearings_health_data = bearings_health_data[1]
            
            # List of processed bearings.
            bearings_processed = list(map(int, list(bearings_health_data.keys())))
            bearings_not_processed = [x for x in params['bearings'] if x not in bearings_processed] 

            # Checking if it's needed to process a new bearing.
            if bearings_not_processed == []:
                for current_bearing in params['bearings']:
                    bearing_health_data = bearings_health_data[str(current_bearing)]
                    bearing_health_data['correlation_coefficients'] = self.correlation_coefficients(bearing_health_data, params)
                bearings_health_data = self.states_assesment(bearings_health_data, params)

                return bearings_health_data
        
        else:
            bearings_health_data = {}

        # Singular values and correlation coefficient calculation.
        for current_bearing in bearings_not_processed:
            bearing_health_data = {
                'singular_values': [],
                'base_values': [],
                'correlation_coefficients': [],
                'health_states': {},
                'threshold_value': 0
            } 

            bearing_files = dataset.bearings_files[str(current_bearing)]

            # Calculating singular values.
            bearing_health_data['singular_values'] = self.svd_norm_sequences(bearing_files, params)

            # Calculating correlation coefficient for the bearing singular values.
            bearing_health_data['correlation_coefficients'] = self.correlation_coefficients(bearing_health_data, params)

            # Adding processed data to dict.
            bearings_health_data[str(current_bearing)] = bearing_health_data
        
        # Health states assessment
        bearings_health_data = self.states_assesment(bearings_health_data, params)
                    
        # Saving in file all processed data.
        dataset.save_processed_data(bearings_health_data, processed_data_path)

        return bearings_health_data

    def states_assesment(self, bearings_health_data, params):
        
        for _, health_data in bearings_health_data.items():

            index_separator_candidate = -1

            for i, correlation_coefficient in enumerate(health_data['correlation_coefficients']):

                if correlation_coefficient < params['manual_threshold'] and index_separator_candidate == -1:
                    index_separator_candidate = i # i is the position of the first 'fast_degradation' data.

                if correlation_coefficient > params['manual_threshold'] and index_separator_candidate != -1:
                    test_qty = int(len(health_data['correlation_coefficients'])*0.5//100) # Quantity of data to determine if it's back to normal state.
                    test_index = np.arange(i, i + test_qty)
                    test_data = np.array(health_data['correlation_coefficients'])[test_index]

                    data_average = statistics.mean(test_data)

                    # Checking if data is back to normal state.
                    if data_average > params['manual_threshold']:
                        # Reseting index separator.
                        index_separator_candidate = -1
            
            health_data['health_states']['normal'] = [0, index_separator_candidate - 1]
            health_data['health_states']['fast_degradation'] = [index_separator_candidate, len(health_data['correlation_coefficients']) - 1]
                
        return bearings_health_data

    def rms(self, dataset, params):

        processed_data_path = 'rms/rms'
        bearings_rms = dataset.load_processed_data(dataset, processed_data_path)
        bearings_not_processed = params['bearings']

        if bearings_rms[0]:
            bearings_rms = bearings_rms[1]
            
            bearings_processed = list(map(int, list(bearings_rms.keys())))
            bearings_not_processed = [x for x in params['bearings'] if x not in bearings_processed] 
            
            if bearings_not_processed == []:
                return bearings_rms

        else:    
            bearings_rms = OrderedDict()

        for current_bearing in bearings_not_processed:
            bearing_files = dataset.bearings_files[str(current_bearing)]
            bearing_rms = []

            for bearing_file in bearing_files:
                data = bearing_file[params['vibration_signal']].values
                # Calculating RMS.
                bearing_rms.append(math.sqrt(np.mean(data**2)))
            # Smoothing data.
            bearing_rms =  savgol_filter(bearing_rms, params['smoothing_window_size'], 3)
            bearings_rms[str(current_bearing)] = bearing_rms
    
        dataset.save_processed_data(bearings_rms, processed_data_path)

        return bearings_rms

    def correlation_coefficients(self, bearing_health_data, params):
        """Take x% of the initial data, calculate the singular values for each file. 
        For each singular values, calculate the correlation coefficient and then take the mean of x% correlation coefficients."""
     
        correlation_coefficients = []
        
        svd_norm_sequences = bearing_health_data['singular_values']

        data_len = len(svd_norm_sequences)
        ini = data_len*params['base_values_chunk_percentage'][0]//100
        end = data_len*params['base_values_chunk_percentage'][1]//100
        base_values = svd_norm_sequences[ini: end]

        bearing_health_data['base_values'] = base_values
       
        # 1st - Take the mean of the base values.
        if params['correlation_coefficient_method'] == 'base_values_mean':
            base_value = [statistics.mean(k) for k in zip(*base_values)]
            
            for svd_norm_sequence in svd_norm_sequences:
                sum_xy = 0; sum_xx = 0; sum_yy = 0
                for x, y in zip(base_value, svd_norm_sequence):
                    sum_xy += x*y
                    sum_xx += x*x
                    sum_yy += y*y
                sqrt_xx_yy = math.sqrt(sum_xx*sum_yy)
                correlation_coefficients.append(sum_xy / sqrt_xx_yy)

        # 2nd - Take the mean of the correlation coefficients calculated for the set of base values.
        elif params['correlation_coefficient_method'] == 'correlation_coefficient_values_mean':
            for svd_norm_sequence in svd_norm_sequences:
                file_corr_coefs = []
                for base_value in base_values:
                    sum_xy = 0; sum_xx = 0; sum_yy = 0
                    for x, y in zip(base_value, svd_norm_sequence):
                        sum_xy += x*y
                        sum_xx += x*x
                        sum_yy += y*y
                    sqrt_xx_yy = math.sqrt(sum_xx*sum_yy)
                    # Calculating correlation coefficient.
                    file_corr_coefs.append(sum_xy / sqrt_xx_yy)
                # Taking mean of the correlation coefficients highest values.
                correlation_coefficients.append(statistics.mean(sorted(file_corr_coefs, reverse=True)[0:params['max_qty']]))

        # Smoothing correlation coefficients.
        correlation_coefficients = savgol_filter(correlation_coefficients, params['smoothing_window_size'], 2)

        # Normalizing correlation coefficients to [-1, 1].
        correlation_coefficients_norm = self.normalize_data(correlation_coefficients, params)

        return correlation_coefficients_norm

    def normalize_data(self, data, params):

        data = np.array(data)
        data_shape = data.shape
        data_row = np.reshape(data, [1, data.size])[0]

        data_max = np.amax(data_row)
        data_min = np.amin(data_row)
        
        DIFF = data_max - data_min;  a, b = params['norm_interval'];  norm_diff = b - a
            
        data_normalized = []
        for value in data_row:
            data_normalized.append(( norm_diff*(value - data_min) / DIFF ) + a)
        
        data_normalized = np.array(data_normalized)
        data_normalized = np.reshape(data_normalized, data_shape)

        return data_normalized.tolist()

    def svd_norm_sequences(self, bearing_files, params):
        
        svd_sequences = []
        svd_norm_sequences = []
        files_size = len(bearing_files)
        loop_time = 0

        # Calculating hankel matrix for each datafile and estimating remaining time..
        for i, bearing_file in enumerate(bearing_files):
            
            # Initial time.
            ini = time.time()
            
            data = bearing_file[params['vibration_signal']].values

            # Embedding process.
            """Hankel matrix size reference: Mahmoudvand, R. and Zokaei, M., 2012. On the singular values of the Hankel matrix with application in singular spectrum analysis. Chilean Journal of Statistics, 3(1), pp.43-56."""
            N = len(data); L = params['hankel_window_size'] # L = int(N//4) # K = N - L + 1 
            c = data[0:L]; r = data[L-1:N] 
            # Computing hankel matrix.
            hankel_matrix = hankel(c, r)

            # Decomposing hankel matrix.
            svd_sequences.append(svdvals(hankel_matrix))

            # Calculating remaining time.
            end = time.time()
            loop_time = (loop_time*i + (end-ini))/(i+1); remaining_time = loop_time * (files_size - i)

            if i%10 == 0:
                print('SVD - Processed ', i+1,' of ', files_size,' files.', int(remaining_time/60), 'minutes reamining.')

        # Normalizing to [-1, 1]
        svd_norm_sequences = self.normalize_data(svd_sequences, params)

        return svd_norm_sequences

    def rul_stop_threshold(self, data_processed):
        
        bearings_rms, bearings_health_assessment = data_processed['rms'], data_processed['health_assessment']
        rms_stop_threshold = 4.47
        recording_step_time = 10 # Seconds

        # Calculating RUL for bearings that got stop threshold.
        bearing_rul = []
        
        for (_, bearing_rms), (_, bearing_health_assessment) in zip(bearings_rms.items(), bearings_health_assessment.items()):
            # Looking for stop rms values in the degradation set.
            degradation_index = np.arange(bearing_health_assessment['health_states']['fast_degradation'][0], bearing_health_assessment['health_states']['fast_degradation'][1] + 1)
            for index, rms_vals in zip(degradation_index, bearing_rms[degradation_index]):
                if rms_vals > rms_stop_threshold:
                    bearing_rul.append((index - bearing_health_assessment['health_states']['fast_degradation'][0])*recording_step_time)
                    break
            else:
                # Fitting polynomial and calculating stop threshold for bearing which didn't reach the stop threshold.
                N = len(degradation_index)
                x = np.arange(N-N//3) # Adjusting the data for the fitting function.
                y = bearing_rms[degradation_index][N//3:N]
                
                pol_coefs2, [resid2, _, _, _] = Polynomial.fit(x, y, 2, full=True)
                pol_coefs3, [resid3, _, _, _] = Polynomial.fit(x, y, 3, full=True)

                # Selecting the polynomial with lowest fitting error.
                if resid2 > resid3:
                    pol_coefs = pol_coefs3
                else:
                    pol_coefs = pol_coefs2
                
                #plt.plot(x, y)
                #plt.plot(polyval(np.arange(len(x)), pol_coefs.convert().coef))
                #plt.show()

                for val in range(10000):
                    if polyval(val, pol_coefs.convert().coef) > rms_stop_threshold:
                        bearing_rul.append(val*recording_step_time)
                        break
        
        return bearing_rul
                    

