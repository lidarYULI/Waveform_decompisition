import numpy as np
import os
import pandas as pd
from files_access import file_path
import warnings

# 将 RuntimeWarning 转换为异常
warnings.simplefilter("error", RuntimeWarning)

def samples_generation_on_GAU_results():
    ###
    shot_number_List, gaussian_decom_results = open_gaussian_decomposition_result(file_path.Iterative_GAU_txt)

    samples_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    feature_dataframe = generate_samples_by_shot_number(shot_number_List, gaussian_decom_results, samples_dataframe)

    output_excel = os.path.join(file_path.Gau_result_path,'Gaussian_modes_features_RF_excel.xlsx')

    feature_dataframe.to_excel(output_excel,index=False)

# generate sample for RF regression based on Gaussian decomposition
def generate_samples_by_shot_number(shot_number_list,decomposition_results,RF_dataframe):

    # define a sample array，12 features + 4 properties +  1 label
    columns = ['normalized_amplitude','mode_duration','cumulative_ratio','mode_percentile','distance_to_max',
               'ma1', 'ma2', 'ma3', 'ma4', 'ma5','ma6', 'loc1','loc2','loc3','loc4','loc5','loc6','mode_zcross']

    all_features_dataframe = pd.DataFrame()
    # calculate the features:
    i = 0
    for shot_number,Gau_result in zip(shot_number_list,decomposition_results):
        i = i + 1
        print(f'\r{i}', end='')
        GEDI_elevation, ALS_elevation, GEDI_zcross, rxwaveform_str = \
        RF_dataframe.loc[shot_number, ['GEDI_lowestmode_height_NAVD', 'DEM_NEON_weighted', 'zcross', 'rxwaveform']]

        rx_waveform_value = np.array(rxwaveform_str.split(',')).astype(np.float32)

        Gau_modes_str_list = Gau_result.strip().split(',')

        features_array = generate_Gau_features(Gau_modes_str_list,len(rx_waveform_value))

        feature_dataframe = pd.DataFrame(features_array, columns=columns)

        # change to elevation
        mode_elevation =  (GEDI_zcross - features_array[:, -1]) * 0.15 + GEDI_elevation

        # change to height
        mode_height =  mode_elevation - ALS_elevation

        feature_dataframe['mode_height'] = mode_height

        feature_dataframe.insert(0, "shot_number", shot_number)

        all_features_dataframe = pd.concat([all_features_dataframe,feature_dataframe])

    return all_features_dataframe

#calculate the features
# def generate_Gau_features(waveform_info,decomposition_result_list):
#
#     # 12 features + 1 center of mode
#     features_array = np.empty((0, 13))
#
#     waveform_str, toploc, search_start, search_end = waveform_info
#
#     waveform = np.array(waveform_str.split(',')).astype(np.float32)
#
#     waveform_length = int(search_end) - int(search_start)
#
#     Gau_modes_array = np.array(decomposition_result_list).astype(np.float32)
#
#     sort_modes_array, fited_waveform = sort_gaussian_decomposition_results(Gau_modes_array, waveform_length)
#
#     # transpose
#     transposed_Gau_modes = sort_modes_array.T  # amplitude,center(mode loca), sigma
#
#     amplitude_array = transposed_Gau_modes[:, 0]
#
#     maximum_amplitude = np.max(amplitude_array)
#
#     # get the derivative
#     derivative_x2, derivative_y2 = initial_parameters.derivative_points(maximum_amplitude)
#
#     derivative_mode_location = transposed_Gau_modes[:, 1][derivative_x2.astype(int)]
#
#     maximum_amplitude_index = np.argmax(amplitude_array)
#
#     ## calculate the featrues for each mode between maximum amplitude and the last mode
#     for i in np.arange(maximum_amplitude_index,len(amplitude_array)):
#         ### add distance features
#         amplitude = transposed_Gau_modes[i, 0]
#
#         center = transposed_Gau_modes[i, 1]
#
#         # 1
#         distance_to_search_start = center - search_start
#         # 2
#         distance_to_search_end = search_end - center
#         # 3
#         distance_ratio = distance_to_search_start / (search_end - search_start)
#         # 4
#         normalized_amplitude = amplitude / maximum_amplitude
#         # 5
#         sigma = transposed_Gau_modes[i, 2]
#         # nearby modes' amplitude
#         amplitude_after = np.zeros([1, 5]).flatten()
#
#         near_amplitudes = amplitude_array[i + 1:i + 1 + 5].tolist()
#
#         amplitude_after[0:len(near_amplitudes)] = near_amplitudes
#
#         # amplitude percentile; 5 features
#         amplitude_percentile = calculate_percentile_with_numpy(amplitude_array, amplitude)
#
#         # cumulative energy ratio
#         cumulative_energy_ratio = np.sum(waveform[0:int(center)]) / np.sum(waveform)
#
#         is_second_derivative = 0
#         if center in derivative_mode_location:
#             is_second_derivative = 1
#
#         features_list = [distance_to_search_start, distance_to_search_end, distance_ratio, normalized_amplitude,sigma] + \
#                         amplitude_after.tolist() + [amplitude_percentile, cumulative_energy_ratio, is_second_derivative,center]
#
#         # total 12 features, and 1 column of label, the last one is the center of mode
#         features_array = np.vstack((features_array, np.array(features_list)))
#
#     return features_array

def generate_Gau_features(decomposition_result_list,waveform_length):

    # 15 features + 1 center of mode
    features_array = np.empty((0, 18))

    Gau_modes_array = np.array(decomposition_result_list).astype(np.float32)

    sort_modes_array, fitted_waveform = sort_gaussian_decomposition_results(Gau_modes_array, waveform_length)

    # transpose
    transposed_Gau_modes = sort_modes_array.T  # amplitude,center(mode loca), sigma

    ## normalization
    col_min = np.min(transposed_Gau_modes[:, 0])
    col_max = np.max(transposed_Gau_modes[:, 0])
    denominator = col_max - col_min

    if denominator == 0:
        transposed_Gau_modes[:, 0] = 1
    else:
        transposed_Gau_modes[:, 0] = (transposed_Gau_modes[:, 0] - col_min) / denominator

    amplitude_array = transposed_Gau_modes[:, 0]
    # get the derivative
    maximum_amplitude_index = np.argmax(amplitude_array)

    ## calculate the featrues for each mode between maximum amplitude and the last mode
    for i in np.arange(maximum_amplitude_index,len(amplitude_array)):
        ### add distance features
        # 1)
        normalized_amplitude = transposed_Gau_modes[i, 0]

        center = transposed_Gau_modes[i, 1]

        # 2)
        mode_duration = transposed_Gau_modes[i, 2]

        # 3) cumulative energy ratio
        gau_flatten_mode = transposed_Gau_modes[:i+1].flatten()

        partial_waveform = calculated_fitted_waveform(gau_flatten_mode,waveform_length)

        cumulative_energy_ratio = np.sum(partial_waveform) / np.sum(fitted_waveform)

        # 4) amplitude percentile;
        amplitude_percentile = calculate_percentile_with_numpy(amplitude_array, normalized_amplitude)

        # 5) distance to maximum amplitude
        distance_to_maximum_amplitude = center - transposed_Gau_modes[maximum_amplitude_index, 1]

        # 6) mode amplitude variation
        amplitude_variation, location_variation = mode_variation(i,transposed_Gau_modes)

        features_list = [normalized_amplitude, mode_duration, cumulative_energy_ratio, amplitude_percentile, distance_to_maximum_amplitude] + \
                        amplitude_variation + location_variation + [center]

        # total 15 features, and 1 column of height, the last one is the center of mode
        features_array = np.vstack((features_array, np.array(features_list)))

    return features_array

def mode_variation(i,transposed_Gau_modes):

    amplitude_array, mode_location = transposed_Gau_modes[:,0].flatten(),transposed_Gau_modes[:,1].flatten()

    amplitude_variation = select_values_around_index(amplitude_array,i)

    location_variation = select_values_around_index(mode_location,i,left_replace = -999, right_replace = 999)

    return amplitude_variation, location_variation


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

def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))

def calculated_fitted_waveform(Gau_flatten_modes,waveform_length):
    x = np.arange(waveform_length)

    fitted_waveform = 0

    for index in range(0, len(Gau_flatten_modes), 3):

        amplitude, center, sigma = Gau_flatten_modes[index], Gau_flatten_modes[index + 1], Gau_flatten_modes[index + 2]

        fit_y = gaussian(x, amplitude, center, sigma)

        fitted_waveform = fitted_waveform + fit_y

    return fitted_waveform

def sort_gaussian_decomposition_results(Gau_flatten_mode,waveform_length):

    modes_array = np.zeros((3,int(len(Gau_flatten_mode)/3)))

    for index in range(0, len(Gau_flatten_mode), 3):
        amplitude, center, sigma = Gau_flatten_mode[index], Gau_flatten_mode[index + 1], Gau_flatten_mode[index + 2]

        modes_array[:,int(index/3)] = [amplitude, center, sigma]

    #### sort
    sort_modes_array = modes_array[:, np.argsort(modes_array[1, :])]

    fited_waveform = calculated_fitted_waveform(Gau_flatten_mode,waveform_length)

    return sort_modes_array, fited_waveform

def open_gaussian_decomposition_result(GAU_result_txt):

    outPutFile = open(GAU_result_txt, 'r')  # read only

    shot_number_list = [line.strip().split(',')[0] for line in outPutFile.readlines()]

    outPutFile.seek(0)

    decomposition_parameters = [line.strip().split(',',1)[1] for line in outPutFile.readlines()]

    return shot_number_list, decomposition_parameters

#calculate the percentile of value
def calculate_percentile_with_numpy(array, value):

    # sort array
    sorted_array = np.sort(array)

    # find the position of value in the array
    rank = np.searchsorted(sorted_array, value, side='right')

    # calculate the percentile of the given value
    percentile = rank / len(sorted_array)
    return percentile

## add classification result to RF mode features excel
def add_waveform_label():

    # waveform_classification_excel = file_path.RF_waveform_classification_excel # GMM labels

    mode_features_excel = file_path.RF_derivative_features_excel

    RF_dataframe = pd.read_excel(file_path.RF_excel,dtype = {'shot_number':str},index_col = 0)

    mode_feature_dataframe = pd.read_excel(mode_features_excel,dtype = {'shot_number':str},index_col = 0)

    dataframe = mode_feature_dataframe.merge(RF_dataframe['kmeans_labels'], left_index=True, right_index=True, how='left')

    dataframe.to_excel(mode_features_excel)


if __name__ == '__main__':
    print('gaussian_samples.py: run something if needed')
    #samples_generation_on_GAU_results()
    #add_waveform_label()