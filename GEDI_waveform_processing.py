import numpy as np
import pandas as pd
from lmfit import Model
from scipy.special import erfc
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import smooth_filter
# refer to Hofton and Blair., 2019, ATBD for GEDI waveform processing

# return the processing parameters: front_threshold, back_threshold, smooth_width(ns)
def GEDI_preprocessing_parameters(alrothm_id):
    # six kinds of thresholds combination are provided
    if alrothm_id == 1:
        return 3,6,6.5
    if alrothm_id == 2:
        return 3,3,3.5
    if alrothm_id == 3:
        return 3,6,3.5
    if alrothm_id == 4:
        return 6,6,6.5
    if alrothm_id == 5:
        return 3,2,3.5
    if alrothm_id == 6:
        return 3,4,3.5

# this is defined for GEDI rx_waveform
def rx_waveform_denoise(rx_waveform, search_start, search_end, algorithm_id):
    # this defines the search_start and search_end
    # signal_threshold = noise_mean + std * preprocessor_threshold # preprocessor_threshold = 4 # default value
    # the first and last position of the rxwaveform value larger than signal_threshold are extended above and below by 100 (ancillary/searchsize)

    search_start, search_end = int(search_start), int(search_end)

    signal_waveform = rx_waveform[search_start:search_end]

    noise_left, noise_right = rx_waveform[0:search_start], rx_waveform[search_end:len(rx_waveform)]

    front_threshold, back_threshold, smooth_sigma = GEDI_preprocessing_parameters(algorithm_id)
    # the smooth_width is dependent on algorithm
    signal_waveform_smooth = smooth_filter.gaussian_smooth(signal_waveform, smooth_width=smooth_sigma)
    # smooth noise by
    noise_left_smooth, noise_right_smooth = smooth_filter.gaussian_smooth(noise_left, smooth_width=6.5), smooth_filter.gaussian_smooth(noise_right,smooth_width=6.5)
    # concat them to a de-noised waveform
    smoothed_rxwaveform = np.hstack([noise_left_smooth, signal_waveform_smooth, noise_right_smooth])

    return smoothed_rxwaveform

