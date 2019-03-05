import matplotlib.pyplot as plt
import numpy as np
from config import CONF

class Functions:
               
    def plot(self, data_processed, params):

        def fft_plot():
            fft_data = data_processed['bearings_fft']

            plt.figure()
            for bearing_name, bearing_fft in fft_data.items():
                ini = 5; middle = 1750; end = 2765
                freqs1, fft1 = bearing_fft[ini]
                freqs2, fft2 = bearing_fft[middle]
                freqs3, fft3 = bearing_fft[end]

                plt.subplot(3, 1, 1)
                plt.plot(freqs1, fft1)
                plt.ylabel('Bearing ' + bearing_name + ', beggining.')

                plt.subplot(3, 1, 2)
                plt.plot(freqs2, fft2)
                plt.ylabel('Bearing ' + bearing_name + ', middle.')

                plt.subplot(3, 1, 3)
                plt.plot(freqs3, fft3)
                plt.ylabel('Bearing ' + bearing_name + ', end.')

                plt.show()
        
        def hht_marginal_spectrum_plot():
            
            data = data_processed['hht_marginal_spectrum']
    
            for bearing_name, bearing_marginal_spectrum in data.items():
                
                N = len(bearing_marginal_spectrum)
                ini = 10; middle = int(N//2-5); end = int(N-15)
                freqs1, spectrum1 = bearing_marginal_spectrum[ini]
                freqs2, spectrum2 = bearing_marginal_spectrum[middle]
                freqs3, spectrum3 = bearing_marginal_spectrum[end]

                fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
                fig.dpi = 180; fig.figsize = (11,11)
                ax[0].plot(freqs1, spectrum1)
                ax[0].set_title('Bearing ' + bearing_name + ', beggining.')
                ax[0].set_xlim(6000, 12000, 'c')
                ax[0].set_ylim(0, 0.5)

                ax[1].plot(freqs2, spectrum2)
                ax[1].set_title('Bearing ' + bearing_name + ', middle.')
                ax[1].set_xlim(6000, 12000, 'c')
                ax[1].set_ylim(0, 0.5)

                ax[2].plot(freqs3, spectrum3)
                ax[2].set_title('Bearing ' + bearing_name + ', end.')
                ax[2].set_xlim(6000, 12000, 'c')
                ax[2].set_ylim(0, 0.5)
                
                plt.show()
            
        def health_assessment_plot():

            rms_data = data_processed['rms']
            health_assesment_data = data_processed['health_assessment']

            for (rms_key, rms), (_, health_assessment) in zip(rms_data.items(), health_assesment_data.items()):
                fig = plt.figure()
                fig.dpi = 200; fig.figsize = (11,11)
                plt.title('Correlation Coefficient - Bearing '+ str(int(rms_key)+1))
                plt.plot(health_assessment['correlation_coefficients'], 'C2')
                plt.plot(rms, 'C0')
                ax_point = health_assessment['health_states']['fast_degradation'][0]
                plt.axhline(health_assessment['correlation_coefficients'][ax_point], color='red')
                plt.axvline(ax_point, color='red')
                plt.show()

        def rms_plot():
            rms_data = data_processed['rms']
           
            for rms_key, rms in rms_data.items():
                fig = plt.figure()
                plt.title('RMS - Bearing '+ str(int(rms_key)+1))
                fig.dpi = 200; fig.figsize = (11,11)
                plt.plot(rms, 'C0')
                plt.show()


        plot_functions = {
            'fft_plot': fft_plot,
            'hht_marginal_spectrum': hht_marginal_spectrum_plot,
            'health_assessment': health_assessment_plot,
            'rms': rms_plot
        }

        for result_name in params['results_to_show']:
            plot_functions[result_name]()




                
                
                
                


