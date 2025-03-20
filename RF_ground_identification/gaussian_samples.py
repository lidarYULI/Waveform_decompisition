import numpy as np
import os
import pandas as pd
import initial_parameters
from files_access import file_path


def Gaussian_samples_generation():
    ###
    shot_number_List, gaussian_decom_results = open_gaussian_decomposition_result()

    samples_dataframe = pd.read_excel(file_path.manual_sample_excel, dtype={'shot_number': str}, index_col=0)

    feature_dataframe = generate_gaussian_samples_by_shot_number(shot_number_List, gaussian_decom_results,
                                                                samples_dataframe)

    output_excel = os.path.join(file_path.Gau_result_path,'Gaussian_modes_features.xlsx')

    feature_dataframe.to_excel(output_excel)

# generate sample based on Gaussian decomposition, with label
def generate_gaussian_samples_by_shot_number(shot_number_list,decomposition_results,samples_dataframe):

    # define a sample array，12 features + 4 properties +  1 label
    columns = ['distance_to_start', 'distance_to_end', 'distance_ratio', 'normalized_amplitude','mode_duration',
               'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'cumulative_ratio', 'mode_amplitude', 'is_second_derivative', 'mode', 'shot_number', 'site', 'is_ground']
    all_features_dataframe = pd.DataFrame(columns = columns)
    # calculate the features:
    for shot_number in samples_dataframe.index.values.tolist():

        GEDI_elevation, ALS_elevation, GEDI_zcross, search_start, search_end, toploc, rxwaveform_str, site = \
        samples_dataframe.loc[shot_number, ['GEDI_lowestmode_height_NAVD', 'DEM_NEON', 'zcross',
                                            'search_start', 'search_end', 'toploc', 'rxwaveform','site']]

        ALS_zcross = GEDI_zcross + (GEDI_elevation - ALS_elevation) / 0.15

        waveform_info = [rxwaveform_str, toploc, search_start, search_end]

        decomposition_index = shot_number_list.index(shot_number)

        Gau_modes_str_list = decomposition_results[decomposition_index].strip().split(',')

        features_array_without_label = generate_Gau_features(waveform_info,Gau_modes_str_list)

        # add label
        centers = features_array_without_label[:, -1]

        min_index = np.argmin(abs(centers - ALS_zcross))

        return_mode_label = np.zeros([len(centers), 1])

        return_mode_label[min_index, 0] = 1

        features_array_with_label = np.hstack([features_array_without_label, return_mode_label])

        feature_dataframe = pd.DataFrame(features_array_with_label, columns=columns)

        feature_dataframe[['shot_number','site']] = shot_number,site

        all_features_dataframe = pd.concat([all_features_dataframe,feature_dataframe])

    return all_features_dataframe

#calculate the features
def generate_Gau_features(waveform_info,decomposition_result_list):

    # 12 features + 1 center of mode
    features_array = np.empty((0, 13))

    waveform_str, toploc, search_start, search_end = waveform_info

    waveform = np.array(waveform_str.split(',')).astype(np.float32)

    waveform_length = int(search_end) - int(search_start)

    Gau_modes_array = np.array(decomposition_result_list).astype(np.float32)

    sort_modes_array, fited_waveform = sort_gaussian_decomposition_results(Gau_modes_array, waveform_length)

    # transpose
    transposed_Gau_modes = sort_modes_array.T  # amplitude,center(mode loca), sigma

    amplitude_array = transposed_Gau_modes[:, 0]

    maximum_amplitude = np.max(amplitude_array)

    # get the derivative
    derivative_x2, derivative_y2 = initial_parameters.derivative_points(maximum_amplitude)

    derivative_mode_location = transposed_Gau_modes[:, 1][derivative_x2.astype(int)]

    maximum_amplitude_index = np.argmax(amplitude_array)

    ## calculate the featrues for each mode between maximum amplitude and the last mode
    for i in np.arange(maximum_amplitude_index,len(amplitude_array)):
        ### add distance features
        amplitude = transposed_Gau_modes[i, 0]

        center = transposed_Gau_modes[i, 1]

        # 1
        distance_to_search_start = center - search_start
        # 2
        distance_to_search_end = search_end - center
        # 3
        distance_ratio = distance_to_search_start / (search_end - search_start)
        # 4
        normalized_amplitude = amplitude / maximum_amplitude
        # 5
        sigma = transposed_Gau_modes[i, 2]
        # nearby modes' amplitude
        amplitude_after = np.zeros([1, 5]).flatten()

        near_amplitudes = amplitude_array[i + 1:i + 1 + 5].tolist()

        amplitude_after[0:len(near_amplitudes)] = near_amplitudes

        # amplitude percentile; 5 features
        amplitude_percentile = calculate_percentile_with_numpy(amplitude_array, amplitude)

        # cumulative energy ratio
        cumulative_energy_ratio = np.sum(waveform[0:int(center)]) / np.sum(waveform)

        is_second_derivative = 0
        if center in derivative_mode_location:
            is_second_derivative = 1

        features_list = [distance_to_search_start, distance_to_search_end, distance_ratio, normalized_amplitude,sigma] + \
                        amplitude_after.tolist() + [amplitude_percentile, cumulative_energy_ratio, is_second_derivative,center]

        # total 12 features, and 1 column of label, the last one is the center of mode
        features_array = np.vstack((features_array, np.array(features_list)))

    return features_array

def sort_gaussian_decomposition_results(Gau_modes_array,waveform_length):
    fited_waveform = 0
    i = 0
    x = np.arange(waveform_length)

    def gaussian(x, amplitude, center, sigma):
        return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))

    modes_array = np.zeros((3,int(len(Gau_modes_array)/3)))
    for index in range(0, len(Gau_modes_array), 3):
        amplitude, center, sigma = Gau_modes_array[index], Gau_modes_array[index + 1], Gau_modes_array[index + 2]

        modes_array[:,int(index/3)] = [amplitude, center, sigma]

        fit_y = gaussian(x, amplitude, center, sigma)

        fited_waveform = fited_waveform + fit_y

        i = i + 1
    #### sort
    sort_modes_array = modes_array[:, np.argsort(modes_array[1, :])]

    return sort_modes_array, fited_waveform

def open_gaussian_decomposition_result():

    outPutFile = open(file_path.Gau_Decomposition_txt, 'r')  # read only

    shot_number_list = [line.strip().split(',')[-1] for line in outPutFile.readlines()]

    outPutFile.seek(0)

    decomposition_parameters = outPutFile.read().splitlines()

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