import numpy as np
import initial_parameters
import pandas as pd
import GEDI_waveform_processing
import os
from gaussian_samples import file_path

# generate training samples based on ALS elevation
def trainsamples_by_ALS_elevation():

    sample_DataFrame = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    site_groups = sample_DataFrame.groupby('site')

    merge_dataframe = pd.DataFrame()

    for group in site_groups:

        dataframe = group[1]

        random_number = int(len(dataframe) * 0.1) # select 10% of dataset

        randomly_selected_dataframe = dataframe.sample(n = random_number)

        merge_dataframe = pd.concat([merge_dataframe,randomly_selected_dataframe],axis = 0)

    features_dataframe = generate_sample_dataframe(merge_dataframe)

    features_dataframe['is_ground'] = 0

    random_excel_output_path = os.path.join(file_path.elevation_train,'random_selection_samples.xlsx')

    features_dataframe.loc[abs(features_dataframe['zcross_ALS'] - features_dataframe['mode']) < 10,'is_ground'] = 1

    dirpath = os.path.dirname(random_excel_output_path)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    features_dataframe.to_excel(random_excel_output_path, index = False)

def trainsamples_by_manual_selection():

    visual_sample_DataFrame = pd.read_excel(file_path.manually_train, dtype={'shot_number': str}, index_col=0)

    features_dataframe = generate_sample_dataframe(visual_sample_DataFrame)

    features_dataframe['is_ground'] = 0

    random_excel_output_path = os.path.join(file_path.manually_train,'manual_selection_samples.xlsx')
    # the manually selected location may not precisely correspond to a mode (derivative of the waveform)
    # so 5 bins tolerance is used
    features_dataframe.loc[abs(features_dataframe['zcross_manually'] - features_dataframe['mode']) < 5, 'is_ground'] = 1

    dirpath = os.path.dirname(random_excel_output_path)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    features_dataframe.to_excel(random_excel_output_path, index = False)

def generate_sample_dataframe(input_dataframe):
    '''
    :param excel_file: calculate the features for all modes in a waveform
    :return:
    '''

    # columns:  ['mode','stddev', 'distance_ratio', 'percentile_amp']
    # ['distance_to_end', 'distance_to_start','is_second_deri','mode_duration','cumulative_ratio', 'mode_amplitude']
    # ['ma1', 'ma2', 'ma3', 'ma4', 'ma5']
    # ['per_25', 'per_50', 'per_75', 'cumu_25_index', 'cumu_50_index', 'cumu_75_index']
    # ['site','zcross_manually','zcross_ALS','shot_number']

    features_DataFrame = pd.DataFrame()

    for index, i in zip(input_dataframe.index.values.tolist(), range(len(input_dataframe))):
        print(index,i)
        rx_waveform_str, search_start, toploc, search_end, mean, site,stddev = input_dataframe.loc[index, ['rxwaveform', 'search_start', 'toploc','search_end', 'mean', 'site','stddev']].values  # zcross

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        waveform_info = [rx_waveform_value, search_start, search_end, mean, stddev]

        # generate features for all modes in a waveform
        sample_fetures_dataframes = generate_features(waveform_info)

        zcross, GEDI_lowest_height_NAVD, DEM_NEON = input_dataframe.loc[index,['zcross','GEDI_lowestmode_height_NAVD','DEM_NEON']].values

        if 'zcross_manually' in input_dataframe.columns:
            zcross_manually = input_dataframe.loc[index,['zcross_manually']].values[0]
        else:
            zcross_manually = 0

        zcross_ALS = (GEDI_lowest_height_NAVD - DEM_NEON)/0.15 + zcross

        sample_fetures_dataframes[['site','zcross_manually','zcross_ALS','shot_number']] = site, zcross_manually, zcross_ALS,index

        features_DataFrame = pd.concat([features_DataFrame,sample_fetures_dataframes],axis=0)

    return features_DataFrame

# generate features from a waveform
def generate_features(waveform_info):

    raw_waveform_value, search_start, search_end, mean, stddev = waveform_info

    smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(raw_waveform_value, search_start, search_end, 2)

    zcross_top_lim = np.argmax(smooth_waveform)
    # mean is the mean noise
    denoised_waveform = smooth_waveform - mean

    sample_fetures_list,columns = feature_extraction(stddev,denoised_waveform, zcross_top_lim,search_start,search_end)

    feature_dataframe = pd.DataFrame(columns=columns)

    for i in range(len(sample_fetures_list)):

        feature_dataframe.loc[i,columns] = sample_fetures_list[i]

    # columns:  ['mode','stddev', 'distance_ratio', 'percentile_amp']
    # ['distance_to_end', 'distance_to_start','is_second_deri','mode_duration','cumulative_ratio', 'mode_amplitude']
    # ['ma1', 'ma2', 'ma3', 'ma4', 'ma5']
    # ['per_25', 'per_50', 'per_75', 'cumu_25_index', 'cumu_50_index', 'cumu_75_index']

    return feature_dataframe

# calculate the feature for each waveform
def feature_extraction(stddev, denoised_waveform, zcross_top_lim, search_start, search_end):

    normalized_waveform = (denoised_waveform - np.min(denoised_waveform)) / (np.max(denoised_waveform) - np.min(denoised_waveform))

    # first derivative
    derivative_x, derivative_y = np.array(initial_parameters.derivative_points(normalized_waveform))

    derivative_waveform = np.array(normalized_waveform)[derivative_x.astype(int)]

    cumu_energy_ratios = calculate_cumulative_ratio(derivative_waveform)

    potential_modes = derivative_x[(derivative_x >= zcross_top_lim) & (derivative_x < search_end)]

    potential_mode_amplitude = np.array(normalized_waveform)[potential_modes.astype(int)]

    samples_list = []

    # 21 columns in total
    basic_features_columns = ['mode','stddev', 'distance_ratio', 'percentile_amp']
    mode_features_columns = ['distance_to_end', 'distance_to_start','is_second_deri','mode_duration','cumulative_ratio', 'mode_amplitude']
    mode_variation_columns = ['ma1', 'ma2', 'ma3', 'ma4', 'ma5']
    waveform_feature_columns = ['per_25', 'per_50', 'per_75', 'cumu_25_index', 'cumu_50_index', 'cumu_75_index']
    column = basic_features_columns + mode_features_columns + mode_variation_columns + waveform_feature_columns

    for mode, i in zip(potential_modes, range(len(potential_modes))):

        mode = int(mode)

        distance_ratio = (mode - search_start) / (search_end - search_start)

        percentile_amp = calculate_percentile_with_numpy(potential_mode_amplitude, normalized_waveform[mode])
        #
        basic_features = [mode, stddev, distance_ratio, percentile_amp]

        mode_features = mode_features_extraction(mode, normalized_waveform, derivative_x, search_start, search_end, cumu_energy_ratios)

        mode_variation_features = modes_variation_features_extraction(mode, normalized_waveform, derivative_x)

        waveform_features = waveform_feature_extraction(normalized_waveform, derivative_x, search_end)

        all_features = basic_features + mode_features + mode_variation_features + waveform_features

        samples_list.append(all_features)

    return samples_list, column

# 1) basic features of mode
def mode_features_extraction(mode_loc,normalized_waveform,derivative_x,search_start,search_end,derivative_cumu_energy_ratios):

    derivative_index = np.argmin(abs(derivative_x - mode_loc))

    inflation_x, inflation_y = np.array(initial_parameters.inflection_points(normalized_waveform))

    # mode_coor = np.array([mode_loc,normalized_waveform[int(mode_loc)]])

    # 1) calculate the distance to search_end
    distance_to_end = mode_loc - search_end  # np.linalg.norm(mode_coor - botloc_point)
    # 2) calculate the distance to search_satrt
    distance_to_start = mode_loc - search_start

    # 3) second derivative,judge if mode_loc is second derivative point
    derivative_y = normalized_waveform[derivative_x.astype(int)]
    derivative_x2, derivative_y2 = initial_parameters.derivative_points(derivative_y)
    derivative_x2_locations = derivative_x[derivative_x2.astype(int)]
    is_second_derivative = 0
    if mode_loc in derivative_x2_locations:
        is_second_derivative = 1

    # 4) the mode duration
    mode_duration = initial_parameters.derivation_sigma(mode_loc, inflation_x)

    # 5)cumulative ratios amplitude
    cumulative_ratio = derivative_cumu_energy_ratios[derivative_index]

    # 6) amplitude
    mode_amplitude = normalized_waveform[mode_loc] / np.max(normalized_waveform)

    return [distance_to_end, distance_to_start, is_second_derivative, mode_duration, cumulative_ratio, mode_amplitude]

# 2) mode variation features
def modes_variation_features_extraction(mode_loc,normalized_waveform,derivative_x):
    '''
    :param mode_loc: the location of current mode
    :param normalized_waveform:
    :param derivative_x: derivative of the waveform
    :return: features of list
    '''

    derivative_index = np.argmin(abs(derivative_x - mode_loc))

    derivative_y = normalized_waveform[derivative_x.astype(int)]

    mode_as = np.zeros([1,5]).flatten()

    mode_amplitudes = derivative_y[derivative_index+1:derivative_index+1+5].tolist()

    mode_as[0:len(mode_amplitudes)] = mode_amplitudes

    mode_as_list = [mode_as[0],mode_as[1],mode_as[2],mode_as[3],mode_as[4]]

    return mode_as_list

# 3) waveform features
def waveform_feature_extraction(normalized_waveform, derivative_x, search_end):
    # amplitude_feature
    derivative_y = normalized_waveform[derivative_x.astype(int)]
    percentile_25,percentile_50,percentile_75 = np.percentile(derivative_y, 25),np.percentile(derivative_y, 50),np.percentile(derivative_y, 75)
    #
    cumu_energy_ratios = calculate_cumulative_ratio(normalized_waveform)
    cumu_index_25 = np.abs(cumu_energy_ratios - 0.25).argmin()
    cumu_index_50 = np.abs(cumu_energy_ratios - 0.5).argmin()
    cumu_index_75 = np.abs(cumu_energy_ratios - 0.75).argmin()
    # change them to relative distance
    return [percentile_25,percentile_50,percentile_75,cumu_index_25 - search_end,cumu_index_50 - search_end,cumu_index_75 - search_end]

# cumulative energy to each bin of waveform
def calculate_cumulative_ratio(waveform):
    ratios = []
    for i in range(len(waveform)):
        cumulative_energy_ratio = np.sum(waveform[0:i+1]) / np.sum(waveform)
        ratios.append(cumulative_energy_ratio)
    return np.array(ratios)

#calculate the percentile of value
def calculate_percentile_with_numpy(array, value):

    # sort array
    sorted_array = np.sort(array)

    # find the position of value in the array
    rank = np.searchsorted(sorted_array, value, side='right')

    # calculate the percentile of the given value
    percentile = rank / len(sorted_array)
    return percentile

if __name__ == '__main__':
    print('samples generation: select a function to run')
    #trainsamples_by_ALS_elevation()
    #trainsamples_by_manual_selection()