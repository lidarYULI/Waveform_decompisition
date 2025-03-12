import pandas as pd
import numpy as np
import gaussian_decomposition as GD_decom
import exgaussian_decomposition as exGD_decom
import matplotlib.pyplot as plt
import model_usage
import os
import matplotlib
matplotlib.use('TkAgg')
def test_gaussia_decomposition():

    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file_path)

    test_excel = os.path.join(project_root,'test_data.xlsx')

    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    shot_number = '79650300200248910'

    tx_parameters, rx_parameters = GEDI_waveform_parameters(dataframe, dataframe,shot_number)

    rx_waveform_value = rx_parameters['rx_waveform']

    stddev, gamma, sigma, search_start, search_end = dataframe.loc[shot_number, ['stddev', 'tx_eggamma', 'tx_egsigma', 'search_start', 'search_end']]

    searching_waveform, noise_std = GD_decom.get_smooth_waveform(rx_waveform_value, search_start, search_end)

    times_noise = 3

    stop_threshold = times_noise * noise_std

    #fitted_parameters = exGD_decom.waveform_decompose_exgaussian(searching_waveform, stop_threshold, sigma, gamma)

    fitted_parameters = GD_decom.waveform_decompose_gaussian(searching_waveform, stop_threshold, sigma)

    lmfit_fig, lmfit_ax = plt.subplots(figsize=(10, 6))

    lmfit_ax.plot(range(len(searching_waveform)), searching_waveform, c='gray', label='searching waveform',linewidth=0.5)

    draw_Gaussian_fitted_modes(lmfit_ax, fitted_parameters, len(searching_waveform))
    ########
    lmfit_ax.axvline(rx_parameters['zcross'] - rx_parameters['search_start'], label='ground', c='red')
    lmfit_ax.legend()
    plt.show()
def draw_Gaussian_fitted_modes(ax, fitted_parameters, length):
    x = np.arange(length)
    # plot Gaussian fit for ground
    sum_y = 0
    i = 0
    for index in range(0, len(fitted_parameters), 3):
        amplitude, center, sigma = fitted_parameters[index], fitted_parameters[index + 1], fitted_parameters[index + 2]
        fit_y = model_usage.gaussian(x, amplitude, center, sigma)
        sum_y = sum_y + fit_y
        i = i + 1
        ax.plot(x, fit_y, linestyle='--', label=f'Gaussian_mode_{i}')

    ax.plot(x, sum_y, c='orange', label='fitted waveform', zorder=0)

def GEDI_waveform_parameters(comparison_DataFrame,process_field_DataFrame,shot_number):

    tx_waveform_str = comparison_DataFrame.loc[shot_number, 'txwaveform']

    tx_waveform_value = np.array(tx_waveform_str.split(',')).astype(np.float32)

    rx_waveform_str = comparison_DataFrame.loc[shot_number, 'rxwaveform']

    rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

    process_fields = process_field_DataFrame.loc[shot_number, ['search_start', 'search_end', 'toploc',
                                                               'botloc', 'zcross', 'zcross0', 'mean',
                                                               'selected_l2a_algorithm']].values.tolist()

    search_start, search_end, toploc, botloc, zcross, zcross0, mean, selected_l2a_algorithm = process_fields

    ALS_CC, GEDI_CC = comparison_DataFrame.loc[shot_number, 'ALS_total_CC'], comparison_DataFrame.loc[
        shot_number, 'GEDI_total_CC']

    beam_name = process_field_DataFrame.loc[shot_number, 'beam']

    tx_parameters = {'tx_waveform': tx_waveform_value, 'beam_name': beam_name, 'shot_number': shot_number,
                     'selected_l2a_algorithm': selected_l2a_algorithm}

    rx_parameters = {'rx_waveform': rx_waveform_value, 'search_start': search_start, 'search_end': search_end,
                     'toploc': toploc,
                     'botloc': botloc, 'zcross': zcross, 'zcross0': zcross0, 'mean': mean,
                     'selected_l2a_algorithm': selected_l2a_algorithm,
                     'ALS_total_CC': ALS_CC, 'GEDI_total_CC': GEDI_CC}
    return tx_parameters, rx_parameters
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_gaussia_decomposition()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
