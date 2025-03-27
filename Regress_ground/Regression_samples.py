import numpy as np
import initial_parameters
import pandas as pd
import GEDI_waveform_processing
import os
from RF_ground_identification.gaussian_samples import file_path
from RF_ground_identification import derivative_samples

def new_samples(waveform_info):

    # existed samples are generated from derivative_samples.py
    feature_dataframe = derivative_samples.generate_features(waveform_info)
    # add more samples for regression
    raw_waveform_value, search_start, search_end, mean, stddev = waveform_info

    smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(raw_waveform_value, search_start, search_end, 2)



