""" CUDA SVD """
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg

N = 6400
Y = np.random.randn(N, N) + 1j*np.random.randn(N, N)
X = np.asarray(Y, np.complex64)
a_gpu = gpuarray.to_gpu(X)

def gpu_skcuda_linalg_svd(input):
        input_gpu = gpuarray.to_gpu(input)
        linalg.init()
        u_gpu, s_gpu, vh_gpu = linalg.svd(input_gpu, 'A', 'A', 'cusolver')
    
gpu_skcuda_linalg_svd(a_gpu)


""" Paralelization of emd """
def split_data(self, alist, parts=1):
    length = len(alist)
    return [ alist[i*length // parts: (i+1)*length // parts] 
            for i in range(parts) ]

decomposer = [EMD(x) for x in data_split]

def enthread(target, args):
    q = queue.Queue()
    def wrapper():
        q.put(target(*args))
    t = threading.Thread(target=wrapper)
    t.start()
    return q

for i in range(4):
    q = enthread(target = decomposer[i].decompose,  args=())
    ...   


""" Testing inst freq """
fs = 25600 # 25.6 KHz
#fs = 6000
f = 20 
x = np.arange(fs) 
y = [ 10*np.sin(2*np.pi*f * (i/fs)) for i in x]
y = np.array(y)


for bearing_marginal_spectrum in bearings_marginal_spectrum:
    for data_file in bearing_marginal_spectrum:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_file[1], data_file[0][0:2559])
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        plt.show()


"""Calculating inst freq and marginal spectrum."""
# Module
imfs_ht_files_module = np.absolute(imfs_ht_files).tolist()

# Frequency
fs = 25600 # 25.6 KHz
imfs_ht_files_InstFreq = []

for imfs_ht_file in imfs_ht_files:
    tmp_InstFreq = []
    for i, imf_ht in enumerate(imfs_ht_file):
        inst_phase = np.unwrap(np.angle(imf_ht))
        tmp_InstFreq.append((np.diff(inst_phase) / (2.0*np.pi)) * fs)

    imfs_ht_files_InstFreq.append(tmp_InstFreq)

# Calculating marginal spectrum.
for imfs_ht_file_module, imfs_ht_file_InstFreq in zip(imfs_ht_files_module, imfs_ht_files_InstFreq):
    freq = [sum(x) for x in zip(*imfs_ht_file_InstFreq)]
    mod = [sum(x) for x in zip(*imfs_ht_file_module)][0:len(freq)]
    bearing_marginal_spectrum.append([mod, freq])


"""Calculating bearing state threshold."""
# Taking the maximum splope point of correlation coefficient for each bearing.
max_slope = []
for bearing_correlation_coefficient in correlation_coefficients:
    bearing_max_slope = np.argmax(np.abs([ x - z for x, z in (zip(bearing_correlation_coefficient[:-1], bearing_correlation_coefficient[1:]))]))
    max_slope.append(bearing_max_slope)


""" Other base value function """
def base_value_mean(self, bearing_files, params):
    """Take x% of the initial data, calculate the singular values for each file and take the mean of those values as base value."""
        
        mean_base_value = [statistics.mean(k) for k in zip(*data['base_values'][current_bearing_index])]
        base_value = mean_base_value

""" Other way to normalize data beetween an interval"""
np.interp(s, (s.min(), s.max()), (-1, +1))


""" Testing signal for hht """
# Test signal
#fs = 6000
#f = 20 
#x = np.arange(fs) 
#y = [10*np.sin(2*np.pi*f * (i/fs)) for i in x]


""" Old svd normalization """
    def singular_val_norm(self, bearing_files, params):
        
        singular_val_norm_sequences = []
        singular_val_sequences = []
        files_size = len(bearing_files)
        avg_duration = 0

        # Calculating hankel matrix for each datafile.
        for i, bearing_file in enumerate(bearing_files):

            ini = time.time()
            
            data = bearing_file[params['vibration_signal']].values
            
            """Hankel matrix size reference: Mahmoudvand, R. and Zokaei, M., 2012. On the singular values of the Hankel matrix with application in singular spectrum analysis. Chilean Journal of Statistics, 3(1), pp.43-56."""

            N = len(data); L = int(N//1.2) #;K = N - L + 1 
            c = data[0:L]
            r = data[L-1:N]

            hankel_matrix = hankel(c, r)

            # Decomposing hankel matrix.
            #svd = TruncatedSVD(n_components=params['num_svd_decompositions'], n_iter=params['num_svd_iterations'])
            s = scipy.linalg.svdvals(hankel_matrix)
            singular_val_sequences.append(s)
            #s = svd.fit(hankel_matrix).singular_values_ 

            #MAX = max(s)   ;   MIN  = min(s)  ;   DIFF = MAX - MIN
            #a, b = params['norm_interval']    ;   norm_diff = b - a
            #
            s_norm = []
            #
            ## Normalizing singular values.
            #for value in s:
            #    s_norm.append((norm_diff*(value - MIN)/DIFF) + a)              

            #singular_val_norm_sequences.append(s_norm)
            
            
            end = time.time()
            # Estimating remaining time do end calculations.
            if avg_duration != 0:
                avg_duration = (avg_duration + (end-ini))/2  
            else:    
                avg_duration = (end-ini)
            
            remaining_time = avg_duration*(files_size-i-1)

            if i%4 == 0:
                print('SVD - Processed ', i+1,' of ', files_size,' files.', int(remaining_time/60), 'minutes reamining.')

        max_s = max(singular_val_sequences[0])
        min_s = min(singular_val_sequences[0])
        
        for singular_val_sequence in singular_val_sequences:
            if max(singular_val_sequence) > max_s:
                max_s = max(singular_val_sequence)
            if min(singular_val_sequence) < min(singular_val_sequence):
                min_s = min(singular_val_sequence)

        DIFF = max_s - min_s
        a, b = params['norm_interval']    ;   norm_diff = b - a

        for singular_val_sequence in singular_val_sequences:
            for value in singular_val_sequence:
                s_norm.append((norm_diff*(value - min_s)/DIFF) + a)
            singular_val_norm_sequences.append(s_norm)


        return singular_val_norm_sequences

""" Old correlation coeffs function """
 def correlation_coefficients(self, data, params):
        """Take x% of the initial data, calculate the singular values for each file. 
        For each singular values, calculate the correlation coefficient and then take the mean of x% correlation coefficients."""
     
        correlation_coefficients = []

        base_values = data['base_values']
        svd_norm_sequences = data['singular_values']

        ## 1st - Take the mean of base values using paper equation.
        base_value = [statistics.mean(k) for k in zip(*base_values)]

        for svd_norm_sequence in svd_norm_sequences:
            sum_xy = 0; sum_xx = 0; sum_yy = 0
            for x, y in zip(base_value, svd_norm_sequence):
                sum_xy += x*y
                sum_xx += x*x
                sum_yy += y*y
            sqrt_xx_yy = math.sqrt(sum_xx*sum_yy)
            correlation_coefficients.append(sum_xy / sqrt_xx_yy)

        # 2nd - Take the mean of correlation coefficients for each base value using np.correlate.
        #for i, singular_val_norm_file in enumerate(singular_val_norm_files):
        #    file_corr_coefs = []
        #    for base_value in base_values:
        #        file_corr_coefs.append(np.correlate(base_value, singular_val_norm_file).item(0))
        #    #correlation_coefficients.append(statistics.mean(file_corr_coefs))
        #    correlation_coefficients.append(statistics.mean(sorted(file_corr_coefs, reverse=True)[0:params['max_qty']]))
        
        ## 3rd - Take the mean of correlation coefficients for each base value using paper equation.
        #for singular_val_norm_file in singular_val_norm_files:
        #    file_corr_coefs = []
        #    for base_value in base_values:
        #        sum_xy = 0; sum_xx = 0; sum_yy = 0
        #        for x, y in zip(base_value, singular_val_norm_file):
        #            sum_xy += x*y
        #            sum_xx += x*x
        #            sum_yy += y*y
        #        sqrt_xx_yy = math.sqrt(sum_xx*sum_yy)
        #        # Calculating correlation coefficient.
        #        file_corr_coefs.append(sum_xy / sqrt_xx_yy)
        #    # Taking mean of the correlation coefficients highest values.
        #    #correlation_coefficients.append(statistics.mean(file_corr_coefs))
        #    correlation_coefficients.append(statistics.mean(sorted(file_corr_coefs, reverse=True)[0:params['max_qty']]))
            
        return correlation_coefficients:
            
""" Debugging """
if function_name == 'health_assessment':
                for i in range(45):
                    data_processing_params[function_name]['smoothing_window_size'] = 4 + (2*i+1)
                    result = self.data_F.health_assessment(self.dataset, data_processing_params[function_name])
                    plt.figure(); plt.plot(result['0']['correlation_coefficients']);plt.xlim(2710,2810); plt.ylim(-1.1,1.1); plt.savefig('B0_W'+str(i+1))
                    plt.figure(); plt.plot(result['1']['correlation_coefficients']);plt.xlim(780,880); plt.ylim(-1.1,1.1); plt.savefig('B2_W'+str(i+1))
                    plt.figure(); plt.plot(result['6']['correlation_coefficients']);plt.xlim(2160,2270); plt.ylim(-1.1,1.1); plt.savefig('B6_W'+str(i+1))


""" Analysing behavior of singular values"""
def window_size_analysis(self, bearings_file, params):
    
    bearing_file = bearings_file
    window_size_list = np.arange(2, 10)

    f, ax = plt.subplots(4,2)
    for i, window_size in enumerate(window_size_list):
        qty_to_show = window_size
        params['window_size'] = window_size
        s_norm = self.svd_norm_sequences(bearing_file, params)
        N = len(s_norm)
        
        l = i%4; c = int(i//4)
        ax[l][c].plot(self.normalized_points(window_size, qty_to_show), s_norm[0:52][0:window_size], color='b')
        ax[l][c].plot(self.normalized_points(window_size, qty_to_show), s_norm[2750:2802][0:window_size], color='r')
        ax[l][c].legend(loc="upper right")
    plt.show()

def normalized_points(self, window_size, qty_to_show):
    x_points = np.arange(1, qty_to_show+1)
    max_s = qty_to_show; min_s = 0
    DIFF = max_s - min_s;  a = 0; b = window_size;  norm_diff = b - a

    x_points_normalized = []
    for value in x_points:
        x_points_normalized.append(( norm_diff*(value - min_s) / DIFF ) + a)
    
    return x_points_normalized


""" HHT """
                for imf_ht_file in imfs_ht_file:
                    freqs, _, spectrum = scipy.signal.spectrogram(np.imag(imf_ht_file), fs)
                    imfs_mag_spec_file.append([freqs, spectrum])
                imfs_mag_spec_files.append(imfs_mag_spec_file)


""" Outliers filtering """
    def remove_outliers(self, bearings_files, m=2):
        
        big_data = []

        for bearing_files in bearings_files:
            for bearing_file in bearing_files:
                data = bearing_file['vib_horizontal']
                big_data.extend(data)

        print(sorted(big_data[0:30]))       
        
        return data[abs(data - np.mean(data)) < m * np.std(data)]


""" HHT """
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
            bearings_marginal_spectrum = OrderedDict()

        for current_bearing in bearings_not_processed:
            imfs_files = []
            bearing_marginal_spectrum = []
            bearing_files = dataset.bearings_files[str(current_bearing)]

            # Calculating IMFs for each data file.
            for bearing_file in bearing_files:      
                data = bearing_file[params['vibration_signal']].values
                decomposer = EMD(data)
                imfs_files.append(decomposer.decompose())

            # Calculating Hilbert transform for each IMF.
            imfs_ht_files = []

            for imfs_file in imfs_files:
                imfs_ht_files.append(hilbert(imfs_file))
                       
            # Calculating Hilbert spectrum of each decomposition.
            fs = params['sampling_frequency']
            imfs_mag_spec_files = []

            for imfs_ht_file in imfs_ht_files:
                imfs_mag_spec_file = []
                N = len(imfs_ht_file[0])
                freqs = np.arange(N)*(fs/N)
                freqs = freqs[0:int(N//2)]
                for imf_ht_file in imfs_ht_file:
                    fft_vals = fft(imf_ht_file)
                    fft_theo = 2.0*np.abs(fft_vals/N)
                    fft_theo = fft_theo[0:int(N//2)]
                    imfs_mag_spec_file.append([freqs, fft_theo])
                imfs_mag_spec_files.append(imfs_mag_spec_file)

            # Calculating Hilbert marginal spectrum
            for imfs_mag_spec_file in imfs_mag_spec_files:
                bearing_marginal_spectrum.append([imfs_mag_spec_file[0][0], sum([x[1] for x in imfs_mag_spec_file])])

            bearings_marginal_spectrum[str(current_bearing)] = bearing_marginal_spectrum

        dataset.save_processed_data(bearings_marginal_spectrum, processed_data_path)

        return bearings_marginal_spectrum


""" Non sense max derivative to calculate threshold """
threshold_values_mean = 0
        for (i), (_, health_data) in enumerate(bearings_health_data.items()):
            derivative_list = abs(np.diff(health_data['correlation_coefficients']))
            max_derivative_index = np.argmax(derivative_list) 
            health_data['threshold_value'] = health_data['correlation_coefficients'][max_derivative_index]
            threshold_values_mean = (threshold_values_mean*i + health_data['threshold_value'])/(i+1)

""" Calculating change point in correlation coefficinet by taking the average of a moving window """

# Calculate the mean value in a moving window of size 'average_window_size'.
average_window_index = i % average_window_size
correlation_coefficient_average_list[average_window_index] = correlation_coefficient

correlation_coefficient_window_mean = statistics.mean(correlation_coefficient_average_list)

if correlation_coefficient_window_mean < params['manual_threshold']:
    for j, window_correlation_coeffient in enumerate(correlation_coefficient_average_list):
        if window_correlation_coeffient < params['manual_threshold']:
            health_data['health_states']['normal'] = [0, i + (j - average_window_size) - 1]
            health_data['health_states']['fast_degradation'] = [i-j, len(health_data['correlation_coefficients'])-1]
            health_data['health_states']['threshold_mean'] = correlation_coefficient_window_mean
            break
    break

""" Changing the weights of loss function to improve imbalanced classes (normal and fast degradation). """
# loss_weight = torch.tensor([1.0, 2.0] + [1]*23)

""" Data normalization """

    def normalize_data(self, data):
        data = np.array(data)
        data_shape = data.shape
        data_row = np.reshape(data, [1, data.size])[0]

        data_max = np.amax(data_row)
        data_min = np.amin(data_row)
        
        DIFF = data_max - data_min;  a, b = -1, 1;  norm_diff = b - a
            
        data_normalized = []
        for value in data_row:
            data_normalized.append(( norm_diff*(value - data_min) / DIFF ) + a)
        
        data_normalized = np.array(data_normalized)
        data_normalized = np.reshape(data_normalized, data_shape)

        return data_normalized.tolist()

        """ Wrong HHT """
                # Calculating Hilbert spectrum of each decomposition.
            #fs = params['sampling_frequency']
            #imfs_mag_spec_files = []
            #
            #for imfs_ht_file in imfs_ht_files:
            #    imfs_mag_spec_file = []
            #    N = len(imfs_ht_file[0])
            #    freqs = np.arange(N)*(fs/N)
            #    freqs = freqs[0:int(N//2)]
            #    for imf_ht_file in imfs_ht_file:
            #        fft_vals = fft(imf_ht_file)
            #        fft_theo = 2.0*np.abs(fft_vals/N)
            #        fft_theo = fft_theo[0:int(N//2)]
            #        imfs_mag_spec_file.append([freqs, fft_theo])
            #    imfs_mag_spec_files.append(imfs_mag_spec_file)
            ## Calculating Hilbert marginal spectrum
            #for imfs_mag_spec_file in imfs_mag_spec_files:
            #    bearing_marginal_spectrum.append([imfs_mag_spec_file[0][0], sum([x[1] for x in imfs_mag_spec_file])])

             files_instantaneous_frequency = [np.int_(x) for x in files_instantaneous_frequency]
            
                        for file_instantaneous_frequency, file_amplitude_envelope in zip(files_instantaneous_frequency, files_amplitude_envelope):
                file_imfs_marginal_spectrum = []
                for imf_instantaneous_frequency, imf_amplitude_envelope in zip(file_instantaneous_frequency, file_amplitude_envelope):
                        dups = collections.defaultdict(list)
                        
                        frequencies = []; spectrum = []
                        for i, e in enumerate(imf_instantaneous_frequency):
                            dups[e].append(i)
                        for freq, index in sorted(dups.items()):
                            time_integral = sum([imf_amplitude_envelope[x] for x in index])
                            frequencies.append(freq); spectrum.append(time_integral)
                        file_imfs_marginal_spectrum.append([frequencies, spectrum])
                
                imfs_frequencies = []; imfs_spectrum = []
                for i, (imf_frequencies, imf_spectrum) in enumerate(file_imfs_marginal_spectrum):
                    if i < 5: #Setting the number of IMFs to calculate.
                        imfs_frequencies.extend(imf_frequencies)
                        imfs_spectrum.extend(imf_spectrum)
            
                dups = collections.defaultdict(list)
                    
                frequencies = []; spectrum = []
                for i, e in enumerate(imfs_frequencies):
                    dups[e].append(i)
                for freq, index in sorted(dups.items()):
                    time_integral = sum([imfs_spectrum[x] for x in index])
                    frequencies.append(freq); spectrum.append(time_integral)
                bearing_marginal_spectrum.append([frequencies, spectrum])

            bearings_marginal_spectrum[str(current_bearing)] = bearing_marginal_spectrum