import pandas as pd
import numpy as np
import gaussian_decomposition as GD_decom
import GEDI_waveform_processing as GEDI_processing
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')
from files_access import file_path

def test_gaussia_decomposition():

    test_excel = file_path.manual_sample_excel

    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    shot_number = '79650300200248910'

    waveform_parameters = GEDI_waveform_parameters(dataframe,shot_number)

    rx_waveform_value = waveform_parameters['rx_waveform']

    stddev, gamma, sigma, search_start, search_end = dataframe.loc[shot_number, ['stddev', 'tx_eggamma', 'tx_egsigma', 'search_start', 'search_end']]

    searching_waveform, noise_std = GD_decom.get_smooth_waveform(rx_waveform_value, search_start, search_end)

    times_noise = 3

    stop_threshold = times_noise * noise_std

    #fitted_parameters = exGD_decom.waveform_decompose_exgaussian(searching_waveform, stop_threshold, sigma, gamma)

    fitted_parameters = GD_decom.waveform_decompose_gaussian(searching_waveform, stop_threshold, sigma)

    lmfit_fig, lmfit_ax = plt.subplots(figsize=(10, 6))

    lmfit_ax.plot(range(len(searching_waveform)), searching_waveform, c='gray', label='searching waveform',linewidth=0.5)

    GD_decom.draw_Gaussian_fitted_modes(lmfit_ax, fitted_parameters, len(searching_waveform))
    ########
    lmfit_ax.axvline(waveform_parameters['zcross'] - waveform_parameters['search_start'], label='ground', c='red')
    lmfit_ax.legend()
    plt.show()


def GEDI_waveform_parameters(dataframe,shot_number):

    tx_waveform_str = dataframe.loc[shot_number, 'txwaveform']

    tx_waveform_value = np.array(tx_waveform_str.split(',')).astype(np.float32)

    rx_waveform_str = dataframe.loc[shot_number, 'rxwaveform']

    rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

    process_fields = dataframe.loc[shot_number, ['search_start', 'search_end', 'toploc',
                                                               'botloc', 'zcross', 'zcross0', 'mean',
                                                               'selected_l2a_algorithm','beam','ALS_total_CC','GEDI_total_CC']].to_dict()

    process_fields.update({'rx_waveform': rx_waveform_value,'tx_waveform': tx_waveform_value, 'shot_number': shot_number})

    return process_fields


def draw_rxwaveform():

    test_excel = file_path.manual_sample_excel

    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    shot_number = '79650300200248910'

    wave_parameters = GEDI_waveform_parameters(dataframe,shot_number)

    DEM, lowest_ele, zcross = dataframe.loc[shot_number, ['DEM_NEON_weighted', 'GEDI_lowestmode_height_NAVD', 'zcross']]

    DEM_cross = (lowest_ele - DEM) / 0.15 + zcross

    smooth_waveform = GEDI_processing.rx_waveform_denoise(wave_parameters['rx_waveform'], wave_parameters['search_start'], wave_parameters['search_end'], 2)

    GEDI_fig, GEDI_ax = plt.subplots(figsize=(10, 6))

    GEDI_ax.plot(range(len(wave_parameters['rx_waveform'])), wave_parameters['rx_waveform'], c='gray', zorder=-1,label = 'waveform')
    GEDI_ax.plot(range(len(smooth_waveform)), smooth_waveform, c='black', zorder=0, label = 'smooth waveform')
    GEDI_ax.axvline(DEM_cross, c='cyan', label='ALS DEM', linestyle='dashed', linewidth=2)
    GEDI_ax.axvline(zcross, c='orange', label='lowest_mode', linewidth=2)
    GEDI_ax.axvline(wave_parameters['toploc'], c='tab:blue', label='toploc', linewidth=2)
    GEDI_ax.axvline(wave_parameters['botloc'], c='green', label='botloc', linewidth=2)
    GEDI_ax.scatter([wave_parameters['search_start']], smooth_waveform[int(wave_parameters['search_start'])], marker='*', color=['b'], zorder=2, label='search start',s=80)
    GEDI_ax.scatter([wave_parameters['search_end']], smooth_waveform[int(wave_parameters['search_end'])], marker='*', color=['g'], zorder=2, label='search end ', s=80)
    GEDI_ax.axhline(wave_parameters['mean'], c='tab:gray', label='noise', linewidth=2, linestyle='--')
    GEDI_ax.legend(fontsize=14)

    GEDI_ax.legend()
    plt.show()

if __name__ == '__main__':
    #test_gaussia_decomposition()
    draw_rxwaveform()
