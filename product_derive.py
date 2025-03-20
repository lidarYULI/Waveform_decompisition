import numpy as np
import pandas as pd
import GEDI_waveform_processing as GEDI_processing
import model_usage
from scipy.optimize import least_squares
from files_access import file_path

# # get the fitted waveform based on exgaussian decomposed parameters
def fitted_waveform_exgaussian(parameters,waveform_length):
    x = np.arange(0, len(waveform_length), 1)
    canopy_waveform = np.zeros(len(x))
    if len(parameters) >4:
        for index in range(0, len(parameters) - 4, 3):
            amplitude, center, sigma = parameters[index], parameters[index + 1], parameters[index + 2]
            fit_y = model_usage.gaussian(x, amplitude, center, sigma)
            canopy_waveform = canopy_waveform + fit_y

    amplitude, center, sigma, gamma = parameters[-4], parameters[-3], parameters[-2], parameters[-1]

    ground_waveform = model_usage.ex_gaussian(x, amplitude, center, sigma, gamma)
    #fitted_waveform = canopy_waveform + ground_waveform
    # for extended gaussian function, amplitude is the integrated energy
    canopy_energy, ground_energy = np.sum(canopy_waveform), np.sum(ground_waveform)

    return canopy_waveform, ground_waveform, canopy_energy, ground_energy

 # get the fitted waveform based on gaussian decomposed parameters
def fitted_waveform_gaussian(parameters,waveform_length):

    x = np.arange(0, waveform_length, 1)

    canopy_waveform = np.zeros(len(x))

    for index in range(0, len(parameters) - 3, 3):
        amplitude, center, sigma = parameters[index], parameters[index + 1], parameters[index + 2]
        fit_y = model_usage.gaussian(x, amplitude, center, sigma)
        canopy_waveform = canopy_waveform + fit_y

    amplitude, center, sigma = parameters[-3], parameters[-2], parameters[-1]

    ground_waveform = model_usage.gaussian(x, amplitude, center, sigma)

    canopy_energy, ground_energy = np.sum(canopy_waveform), np.sum(ground_waveform)

    return canopy_waveform, ground_waveform, canopy_energy, ground_energy

# calculate CC based on Gaussian decomposition parameters
# toploc is a parameter provided by GEDI L2A product
def cumulative_CC_gaussian_decom(parameters, waveform_length, toploc):

    hight_bins = 5 / 0.15 # 5 m interval, 0.15 m is the distance of one bin in waveform
    A, rg_center, sigma = parameters[-3:]  # ground return parameter
    canopy_waveform, ground_waveform, RV, RG = fitted_waveform_gaussian(parameters,waveform_length)
    #Rv_list = []
    CC_list = []

    for height in np.arange(rg_center, toploc, -hight_bins):

        vertical_canopy_waveform = canopy_waveform[int(toploc):int(height)]

        RV_z = np.sum(vertical_canopy_waveform)

        #Rv_list.append(RV_z)

        CC = RV_z / (RV + 1.5 * RG)

        CC_list.append(CC)

    return CC_list

#directly fit the ground by the location of identified location and calculate CC
def decompose_based_selected_zcross(excel,ground_zcross_field,is_gaussian = True):

    dataframe = pd.read_excel(excel, dtype={'shot_number': str})

    for i in dataframe.index.tolist():
        print(i)
        # randomforest_zcross
        rx_waveform_str, select_zcross, gamma,sigma, toploc, search_start, search_end = dataframe.loc[i, ['rxwaveform', ground_zcross_field,'tx_eggamma', 'tx_egsigma',
                                                                                                          'toploc', 'search_start', 'search_end']].values
        # is_RF = WRD_manually_dataframe.loc[i,'is_RF']
        # if is_RF<2:
        #     continue
        select_zcross = int(select_zcross)

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        initial_waveform = GEDI_processing.rx_waveform_denoise(rx_waveform_value,search_start,search_end,2)

        mean_noise = np.mean(np.concatenate([initial_waveform[0:search_start], initial_waveform[search_end:]]))

        normalized_waveform = initial_waveform - mean_noise

        # find two nearest vally points around the lowest mode for fitting
        # min_indices = argrelextrema(normalized_waveform, np.less)[0]
        # difference = select_zcross - min_indices
        # left_part_x, right_part_x = min_indices[difference > 0], min_indices[difference < 0]
        # fit_start, fit_end = left_part_x[-1], right_part_x[0]
        try:
            if is_gaussian:
            # Gaussian fit
                popt = model_usage.ground_fit_Gau(normalized_waveform,select_zcross,sigma)
            else:
                # extended Gaussian fit
                popt = model_usage.ground_fit_exgaussian(normalized_waveform, select_zcross, sigma, gamma)
                # # #A, u, sigma, gamma = popt  # A is the integration of Rg because of extended Gaussian waveform
            RV_list, CC_list = cumulative_CC_calculation(normalized_waveform, popt, toploc, is_gaussian = is_gaussian)
            parameters = ','.join([str(np.around(q, 3)) for q in popt])

        except:
            # if fit is failed
            ground_y_below = normalized_waveform[select_zcross:select_zcross + 15]
            RV, RG = np.sum(normalized_waveform[int(toploc):int(select_zcross)]) - np.sum(ground_y_below), np.sum(ground_y_below) * 2
            parameters = 'np.nan'
            RV_list, CC_list = cumulative_CC_without_fit(normalized_waveform, RG, select_zcross, toploc)
            print('fitting defeat', i)

        rv_z_str = ','.join([str(rv) for rv in RV_list])
        cc_z_str = ','.join([str(cc) for cc in CC_list])

        # for Gaussian decomposition
        if is_gaussian:
            dataframe.loc[i, ['RV_RF_z', 'CC_RF_z']] = rv_z_str, cc_z_str
            dataframe.loc[i, 'Fitted_parameters_Rg_RF_GAU'] = parameters
        else:
            # for extended Gaussian decomposition
            dataframe.loc[i, ['RV_RF_z_ex', 'CC_RF_z_ex']] = rv_z_str, cc_z_str
            dataframe.loc[i, 'Fitted_parameters_Rg_RF_GAU_ex'] = parameters
        #
        # WRD_manually_dataframe.loc[i, 'CC_WRD_ex'] = RV/ (RV + 1.5*RG)

    dataframe.to_excel(excel, index=False)

def cumulative_CC_calculation(waveform, rg_fit_ppot, toploc, is_gaussian = True):
    # calculate CC based on extended_Gaussian decomposition function
    hight_bins = 5 / 0.15
    if is_gaussian:
        A, rg_center, sigma = rg_fit_ppot
        half_rg_waveform = model_usage.gaussian(np.arange(toploc, rg_center, 1), A, rg_center, sigma)
        RG = np.sum(half_rg_waveform) * 2
    else:
        A, rg_center, sigma, gamma = rg_fit_ppot
        half_rg_waveform = model_usage.ex_gaussian(np.arange(toploc, rg_center, 1), A, rg_center, sigma, gamma)
        RG = np.sum(model_usage.ex_gaussian(np.arange(toploc, rg_center + 500, 1), A, rg_center, sigma, gamma))

    RV = np.sum(waveform[int(toploc):int(rg_center)]) - np.sum(half_rg_waveform)
    Rv_list = []
    CC_list = []
    for height in np.arange(rg_center, toploc, -hight_bins):
        canopy_waveform = waveform[int(toploc):int(height)]
        overlap_waveform = half_rg_waveform[0:int(height - toploc)]
        RV_z = np.sum(canopy_waveform) - np.sum(overlap_waveform)
        Rv_list.append(RV_z)
        CC = RV_z / (RV + 1.5 * RG)
        CC_list.append(CC)
    return Rv_list, CC_list

def cumulative_CC_without_fit(waveform, Rg, rg_center, toploc):
    # directly calculate the integration of waveform
    hight_bins = 5 / 0.15
    RV = np.sum(waveform[int(toploc):rg_center]) - 1 / 2 * Rg
    Rv_list = []
    CC_list = []
    for height in np.arange(rg_center, toploc, -hight_bins):
        # consider possible overlap of canopy with ground
        if (height < rg_center) & (height > rg_center - 10):
            overlap_rg = 1 / 2 * Rg
        else:
            overlap_rg = 0
        canopy_waveform = waveform[int(toploc):int(height)]
        RV_z = np.sum(canopy_waveform) - overlap_rg
        Rv_list.append(RV_z)
        CC = RV_z / (RV + 1.5 * Rg)
        CC_list.append(CC)
        return Rv_list, CC_list

if __name__ == '__main__':

    RF_excel = file_path.RF_excel
    decompose_based_selected_zcross(RF_excel,'randomforest_zcross')