import numpy as np
import initial_parameters
import pandas as pd
import GEDI_waveform_processing
import os
from files_access import file_path

## just generate predictive variables for ground return mode
def generate_ground_mode_features():
    # 32 columns in total
    features_columns = ['mode_zcross', 'distance_to_start', 'distance_to_end', 'distance_ratio',
                              'mode_amplitude', 'mode_duration', 'amplitude_percentile', 'cumulative_ratio',
                              'is_second_deri','ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5',
                              'loc6','per_5', 'per_25', 'per_45', 'per_65', 'per_85','cumu_5_index', 'cumu_25_index', 'cumu_45_index', 'cumu_65_index', 'cumu_85_index',
                                'stddev']

    ground_features_dataframe = pd.DataFrame(columns = features_columns)

    RF_DataFrame = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)
    i = 0
    for shot_number in RF_DataFrame.index.values:
        i = i + 1
        print(f'\r{i}', end='')
        rx_waveform_str, search_start, toploc, search_end, mean, stddev = RF_DataFrame.loc[shot_number, ['rxwaveform', 'search_start', 'toploc', 'search_end', 'mean', 'stddev']].values

        NEON_reference_ele, GEDI_ele, GEDI_zcross = RF_DataFrame.loc[shot_number,['GEDI_lowestmode_height_NAVD', 'DEM_NEON_weighted', 'zcross']]

        ALS_ground_zcross = GEDI_zcross + (GEDI_ele - NEON_reference_ele)/0.15

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

        zcross_top_lim = np.argmax(smooth_waveform)
        # mean is the mean noise
        denoised_waveform = smooth_waveform - mean

        normalized_waveform = (denoised_waveform - np.min(denoised_waveform)) / (
                    np.max(denoised_waveform) - np.min(denoised_waveform))

        # first derivative
        derivative_x, derivative_y = np.array(initial_parameters.derivative_points(normalized_waveform))

        derivative_waveform = np.array(normalized_waveform)[derivative_x.astype(int)]

        cumu_energy_ratios = calculate_cumulative_ratio(derivative_waveform)

        potential_modes = derivative_x[(derivative_x >= zcross_top_lim) & (derivative_x < search_end + 100)]

        potential_mode_amplitude = np.array(normalized_waveform)[potential_modes.astype(int)]

        ground_derivative =  potential_modes[np.argmin(abs(potential_modes - ALS_ground_zcross))]

        basic_features = mode_features_extraction(ground_derivative, normalized_waveform, derivative_x, search_start,
                                                  search_end, cumu_energy_ratios, potential_mode_amplitude)

        amplitude_variation, mode_loc_variation = modes_variation_features_extraction(ground_derivative,normalized_waveform,derivative_x)

        waveform_features = waveform_feature_extraction(normalized_waveform)

        all_features = basic_features + amplitude_variation + mode_loc_variation + waveform_features + [stddev]

        ground_features_dataframe.loc[shot_number,features_columns] = all_features

    ground_features_dataframe = ground_features_dataframe.rename_axis('shot_number')

    ground_features_dataframe.to_excel(file_path.RF_ground_features_excel)

def trainsamples_by_manual_selection():

    visual_sample_DataFrame = pd.read_excel(file_path.manually_derivative_predictive_samples, dtype={'shot_number': str}, index_col=0)

    features_dataframe = generate_sample_dataframe(visual_sample_DataFrame)

    features_dataframe['is_ground'] = 0

    # the manually selected location may not precisely correspond to a mode (derivative of the waveform)
    # so 5 bins tolerance is used
    features_dataframe.loc[abs(features_dataframe['zcross_manually'] - features_dataframe['mode']) < 5, 'is_ground'] = 1

    dirpath = os.path.dirname(file_path.manually_derivative_predictive_samples)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    features_dataframe.to_excel(file_path.manually_derivative_predictive_samples, index = False)

def generate_sample_dataframe(input_dataframe):
    '''
    :param excel_file: calculate the features for all modes in a waveform
    :return:
    '''

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

    smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(raw_waveform_value, search_start, search_end, 4)

    zcross_top_lim = np.argmax(smooth_waveform)
    # mean is the mean noise
    denoised_waveform = smooth_waveform - mean

    sample_features_list,columns = feature_extraction(stddev,denoised_waveform, zcross_top_lim,search_start,search_end)

    feature_dataframe = pd.DataFrame(columns=columns)

    for i in range(len(sample_features_list)):

        feature_dataframe.loc[i,columns] = np.around(sample_features_list[i],3)

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

    # 32 columns in total

    basic_features_columns = ['Mode location', 'Front distance', 'Back distance','distance_ratio',
                              'Mode amplitude','Mode width','Mode amplitude percentile','Cumulative integral','is_second_deri']

    mode_variation_columns = ['ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6']

    waveform_feature_columns = ['p5','p25', 'p45', 'p65', 'p85', 'c5', 'c25', 'c45', 'c65', 'c85', 'stddev']

    feature_columns = basic_features_columns + mode_variation_columns + waveform_feature_columns

    for mode_zcross, i in zip(potential_modes, range(len(potential_modes))):

        basic_features = mode_features_extraction(mode_zcross, normalized_waveform, derivative_x, search_start, search_end, cumu_energy_ratios,potential_mode_amplitude)

        amplitude_variation, mode_loc_variation = modes_variation_features_extraction(mode_zcross, normalized_waveform, derivative_x)

        waveform_features = waveform_feature_extraction(normalized_waveform)

        all_features = basic_features + amplitude_variation + mode_loc_variation + waveform_features + [stddev]

        samples_list.append(all_features)

    return samples_list, feature_columns

# 1) basic features of mode
def mode_features_extraction(mode_loc,normalized_waveform,derivative_x,search_start,search_end,cumu_energy_ratios,potential_amplitudes):

    derivative_index = np.argmin(abs(derivative_x - mode_loc))

    inflation_x, inflation_y = np.array(initial_parameters.inflection_points(normalized_waveform))

    # mode_coor = np.array([mode_loc,normalized_waveform[int(mode_loc)]])

    # distance features
    distance_to_end = mode_loc - search_end  # np.linalg.norm(mode_coor - botloc_point)
    # distance features
    distance_to_start = mode_loc - search_start
    # distance features
    distance_ratio = (mode_loc - search_start) / (search_end - search_start)

    # amplitude
    mode_amplitude = normalized_waveform[int(mode_loc)]

    # the mode duration
    mode_duration = initial_parameters.derivation_sigma(mode_loc, inflation_x)

    ## mode amplitude percentile
    amplitude_percentile = calculate_percentile_with_numpy(potential_amplitudes,normalized_waveform[int(mode_loc)])

    # cumulative ratios amplitude
    cumulative_ratio = cumu_energy_ratios[derivative_index]

    # second derivative,judge if mode_loc is second derivative point
    derivative_y = normalized_waveform[derivative_x.astype(int)]
    derivative_x2, derivative_y2 = initial_parameters.derivative_points(derivative_y)
    derivative_x2_locations = derivative_x[derivative_x2.astype(int)]
    is_second_derivative = 0
    if mode_loc in derivative_x2_locations:
        is_second_derivative = 1

    basic_features = [mode_loc,distance_to_start,distance_to_end,distance_ratio,
                     mode_amplitude,mode_duration,amplitude_percentile,cumulative_ratio,is_second_derivative]

    return basic_features

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

    amplitude_variation = select_values_around_index(derivative_y,derivative_index)

    modeloc_variaiton = select_values_around_index(derivative_x,derivative_index,0,1000)

    return amplitude_variation,modeloc_variaiton

def select_values_around_index(arr, index,left_replace = 0, right_replace = 0):
    arr = np.array(arr)  # 转换为 numpy 数组
    n = len(arr)

    # 计算索引范围
    left_indices = np.arange(index - 3, index)
    right_indices = np.arange(index + 1, index + 4)

    # 生成结果列表
    left_values = [arr[i] if 0 <= i < n else left_replace for i in left_indices]
    right_values = [arr[i] if 0 <= i < n else right_replace for i in right_indices]

    return left_values + right_values

# 3) waveform features
def waveform_feature_extraction(normalized_waveform):

    cumu_energy_ratios = calculate_cumulative_ratio(normalized_waveform)

    def percentile_from_existing_values(arr, percentile):
        arr_sorted = np.sort(arr)
        index = int(np.ceil(percentile / 100 * len(arr))) - 1
        return arr_sorted[max(0, min(index, len(arr) - 1))]

    percentile_list = []
    cumu_index_list = []
    for percentile in np.arange(5,100,20):

        percentile_list.append(percentile_from_existing_values(normalized_waveform,percentile))

        cumu_index_list.append(np.abs(cumu_energy_ratios - percentile/100).argmin())

    # change them to relative distance
    return percentile_list + cumu_index_list

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

## generate prediction variable based on derivatives of the waveform
def generate_predicted_samples(excel_path,save_excel, elevation_column = 'ALS'):

    RF_dataframe = pd.read_excel(excel_path,dtype={'shot_number':str},index_col=0)

    features_dataframe = pd.DataFrame()
    i = 0
    for shot_number in RF_dataframe.index.values.tolist():
        i = i + 1
        print(f'\r{i}', end='')
        rx_waveform_str, search_start, toploc, search_end, mean, stddev = RF_dataframe.loc[shot_number, ['rxwaveform', 'search_start', 'toploc', 'search_end', 'mean','stddev']].values

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        waveform_info = [rx_waveform_value, search_start, search_end, mean, stddev]

        sample_fetures_dataframes = generate_features(waveform_info)

        sample_fetures_dataframes['shot_number'] = shot_number

        sample_fetures_dataframes['NLCD'] = RF_dataframe.loc[shot_number, 'NLCD']

        features_dataframe = pd.concat([features_dataframe,sample_fetures_dataframes],axis=0)

    features_dataframe.set_index('shot_number',inplace=True)
    #'manual_elevation','zcross_manually'
    dataframe = features_dataframe.merge(RF_dataframe[['site','zcross','DEM_NEON_weighted','GEDI_lowestmode_height_NAVD','kmeans_labels']], left_index=True, right_index=True,how='left')
    #
    if elevation_column == 'manual':
        dataframe.loc[:, 'mode_height'] = (dataframe['zcross'] - dataframe['mode_zcross']) * 0.15 + dataframe['GEDI_lowestmode_height_NAVD'] - dataframe['manual_elevation']
        dataframe['is_ground'] = 0
        dataframe.loc[abs(dataframe['zcross_manually'] - dataframe['mode_zcross']) < 5, 'is_ground'] = 1
    if elevation_column == 'ALS':
        dataframe.loc[:, 'mode_height'] = (dataframe['zcross'] - dataframe['mode_zcross']) * 0.15 + dataframe['GEDI_lowestmode_height_NAVD'] - dataframe['DEM_NEON_weighted']
    if elevation_column == 'GEDI':
        dataframe.loc[:, 'mode_height'] = (dataframe['Mode location'] - dataframe['zcross']) * 0.15
    #dataframe.to_excel(r'C:\Users\lrvin\OneDrive\Desktop\test_figs\temporal.xlsx')

if __name__ == '__main__':
    print('samples generation: select a function to run')

    #trainsamples_by_manual_selection()
    generate_predicted_samples(file_path.RF_excel, file_path.RF_derivative_features_excel)
    #
    # generate_ground_mode_features()


