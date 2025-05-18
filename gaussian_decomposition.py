import numpy as np
from scipy.optimize import least_squares
import initial_parameters
import model_usage
import GEDI_waveform_processing as GEDI_processing
import smooth_filter
import pandas as pd
from files_access import file_path
import re
from Regress_ground.Samples_generation import gaussian_samples


# this script is mostly based on (Hofton, 2000);

def waveform_decompose_gaussian(valid_waveform, stddev, sigma,points='derivative'):
    # normalized_waveform[normalized_waveform < 0] = 0

    inflation_x, inflation_y = initial_parameters.inflection_points(valid_waveform)

    derivation_x, derivation_y = initial_parameters.derivative_points(valid_waveform)

    filtered_parameters = initial_parameters.initial_Gaussian_parameters(valid_waveform, inflation_x, derivation_x, stddev, sigma, points=points)

    ranked_parameters = initial_parameters.flag_gaussian(filtered_parameters, stddev)

    # fit least square
    # Gaussian fit
    result = LM_fitted(valid_waveform, ranked_parameters,stddev)

    fitted_parameters = result.x

    return fitted_parameters

def LM_fitted(y, init_modes, noise_std, method='trf'):

    x = np.arange(0, len(y))
    result = 0
    important_mode = np.empty((0,3))
    for i in range(len(init_modes)):
        new_modes = init_modes.loc[init_modes['flag'] == i, ['amplitude', 'center', 'sigma']].values
        # iteratively add an unimportant mode
        important_mode = np.vstack([important_mode, new_modes])
        # lp_bounds, up_bounds = self.generate_bound_GEDI(important_params, init_parameters)
        sort_params_flatten, lp_bounds, up_bounds = model_usage.generate_bound(important_mode)
        result = least_square_modeling(sort_params_flatten, lp_bounds, up_bounds, x, y, noise_std, method = method)

        fitted_parameters = result.x
        residual_mean = np.mean((y - model_usage.model_Gaussian(x,fitted_parameters)) ** 2)
        square_root = np.sqrt(residual_mean)
        if square_root < noise_std:
            return result
    return result

def least_square_modeling(params, lp_bounds, up_bounds, x, y, noise_std, method='trf'):

    def cost_effective(params, x, y_obser):
        # return (np.sum((model(params,x) - y_obser)/noise_std))**2
        return ((model_usage.model_Gaussian(x, params) - y_obser) / noise_std) ** 2

    if method == 'lf':
        result = least_squares(cost_effective, params, args=(x, y), method='lf')
    else:
        result = least_squares(cost_effective, params, args=(x, y), method='trf', bounds=(lp_bounds, up_bounds))

    return result

# get used parameters and data
# take the waveform between "search_start" and "search_end"
def get_smooth_waveform(GEDI_waveform,search_start,search_end):

    smooth_waveform = GEDI_processing.rx_waveform_denoise(GEDI_waveform,search_start,search_end,2)
    # 1) smooth waveform, it is unnecessary
    noise_mean, noise_std = smooth_filter.mean_noise_level(smooth_waveform)

    normalized_waveform = smooth_waveform - noise_mean

    normalized_waveform[normalized_waveform< 0 ] = 0
    # get searching waveform
    searching_waveform = np.array(normalized_waveform[int(search_start):int(search_end)])

    return searching_waveform, noise_std

#### take the mode most biased from the waveform as the important mode;
#### this is a new method different from LM_fitted
def LM_fitted_sort_by_std(ori_waveform, init_modes, noise_std, method='trf'):

    result_dict = {}
    x = np.arange(0, len(ori_waveform))

    i = len(init_modes.loc[init_modes['flag'] == 0])

    if i == 0:
        init_modes.loc[init_modes['amplitude'].idxmax == 0,'flag'] = 0
    j = 0
    while i <= len(init_modes):

        used_modes = init_modes.loc[init_modes['flag'] == 0, ['amplitude', 'center', 'sigma']].values

        sort_params_flatten, lp_bounds, up_bounds = model_usage.generate_bound_simple(used_modes)

        result = least_square_modeling(sort_params_flatten, lp_bounds, up_bounds, x, ori_waveform, noise_std, method=method)

        fitted_parameters = result.x

        fitted_waveform = model_usage.model_Gaussian(x, fitted_parameters)

        residual_mean = np.mean((ori_waveform - fitted_waveform) ** 2)

        square_root = np.sqrt(residual_mean)

        result_dict.update({f'{j}':{'criterion':square_root, 'fitted_result':result}})
        j = j + 1
        if square_root < noise_std:
            return result_dict
        else:
            unused_modes = init_modes.loc[init_modes['flag'] != 0, ['amplitude', 'center', 'sigma']].values
            # if no available modes can be added
            if len(unused_modes) == 0:
                return result_dict
            unused_derivative = unused_modes[:,1]

            used_derivative = used_modes[:,1]

            used_derivative_metrics = derivative_metrics(unused_derivative,used_derivative,ori_waveform,fitted_waveform)

            sorted_indices = np.lexsort((-used_derivative_metrics[:, 2], -used_derivative_metrics[:, 1])) #sort by mean_std and deviated amplitude

            sorted_arr = used_derivative_metrics[sorted_indices]

            selected_derivative = sorted_arr[0, 0]

            init_modes.loc[init_modes['center'] == selected_derivative,'flag'] = 0

        i = len(init_modes.loc[init_modes['flag'] == 0])

    return result_dict

def derivative_metrics(unused_derivative, existed_mode, ori_waveform, fitted_waveform):
    #
    metric_array = np.empty((len(unused_derivative), 3))

    for i, deri in zip(range(len(unused_derivative)), unused_derivative):

        deri_value_ori = ori_waveform[int(deri)]

        deri_value_fit = fitted_waveform[int(deri)]

        try:
            left = int(existed_mode[existed_mode < deri][-1])
        except:
            left = 0
        try:
            right = int(existed_mode[existed_mode > deri][0])
        except:
            right = len(ori_waveform)
        # calculate metrics
        segment_waveform = ori_waveform[left:right]

        segment_fitted_waveform = fitted_waveform[left:right]

        mean_std = np.std(segment_waveform - segment_fitted_waveform)

        distance = np.abs(deri_value_fit - deri_value_ori)

        metric_array[i, :] = [deri, mean_std, distance]

    return metric_array

def iterative_waveform_decomposition(valid_waveform,noise_std,sigma):
    # normalized_waveform[normalized_waveform < 0] = 0

    inflation_x, inflation_y = initial_parameters.inflection_points(valid_waveform)

    derivation_x, derivation_y = initial_parameters.derivative_points(valid_waveform)

    filtered_parameters = initial_parameters.initial_Gaussian_parameters(valid_waveform, inflation_x, derivation_x,
                                                                         noise_std, sigma, points='derivative')

    ranked_parameters = initial_parameters.flag_gaussian(filtered_parameters, noise_std,important_sigma = 3)

    # fit least square
    # Gaussian fit
    results_dict = LM_fitted_sort_by_std(valid_waveform, ranked_parameters, noise_std)

    return results_dict

### stepwise add mode to fit waveform by gaussian
def stepwise_waveform_decomposition(valid_waveform,noise_std,sigma):

    inflection_point_x, inflection_point_y = initial_parameters.inflection_points(valid_waveform)

    # filter_inflation_x, filter_inflation_y = initial_parameters.filter_inflection(inflection_point_x, inflection_point_y,valid_waveform)

    derivation_point_x, derivation_point_y = initial_parameters.derivative_points(valid_waveform)

    potential_modes = initial_parameters.initial_Gaussian_parameters(valid_waveform, inflection_point_x,
                                                                 derivation_point_x, noise_std, sigma,
                                                                 points='derivative')
    potential_modes['flag'] = 1

    potential_modes.loc[potential_modes['amplitude'].idxmax(),'flag'] = 0

    x = np.arange(len(valid_waveform))

    square_root, fitted_parameters, fitted_waveform = None, None, None
    ## stepwise add one mode mostly deviated from observed waveform
    for i in np.arange(len(potential_modes) - 1):

        fit_dataframe = potential_modes[potential_modes['flag'] == 0]

        fit_dataframe = fit_dataframe.sort_values(by='center', ascending=True)
        # print(fit_dataframe)

        fitted_parameters = LM_fitted(valid_waveform,fit_dataframe, noise_std)

        fitted_waveform = model_usage.model_Gaussian(x, fitted_parameters.x)

        square_root = np.sqrt(np.mean((valid_waveform - fitted_waveform) ** 2))

        if square_root > noise_std:

            center_array = potential_modes.loc[potential_modes['flag']!=0, 'center'].values

            amplitude_array = potential_modes.loc[potential_modes['flag']!=0, 'amplitude'].values

            center_waveform_value = fitted_waveform[center_array.astype(int)]

            diff_value = abs(amplitude_array - center_waveform_value)

            max_diff_index = np.argmax(diff_value)

            # add new modes
            potential_modes.loc[potential_modes['center'] == center_array[max_diff_index],'flag'] = 0

        else:
            return square_root, fitted_parameters, fitted_waveform

    return square_root, fitted_parameters, fitted_waveform

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

def output_iterative_GD_results(excel):

    outPutFile = open(file_path.Iterative_GAU_txt, 'a+')  # a+ 读写模式，没有文件,会创建
    outPutFile.seek(0)  # 移动到文件开头
    existed_shot_number_List = [line.strip().split(',')[0] for line in outPutFile.readlines()]
    outPutFile.seek(0, 2)  # 移动到文件末尾

    dataframe = pd.read_excel(excel, dtype={'shot_number': str}, index_col=0)
    i = 0
    for shot_number in dataframe.index.values.tolist():
        if shot_number in existed_shot_number_List:
            i = i + 1
            progress = r'progress = %.2f%%' % (i / len(dataframe) * 100,)
            print(progress)
            continue

        rx_waveform_str, sigma, search_start, search_end, zcross = dataframe.loc[
            shot_number, ['rxwaveform', 'tx_egsigma', 'search_start', 'search_end', 'zcross']]

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        searching_waveform, noise_std = get_smooth_waveform(rx_waveform_value, search_start, search_end)
        try:
            resutls_dict = iterative_waveform_decomposition(searching_waveform, 1.5 * noise_std, sigma)
        except:
            print(f'{shot_number} is error')
            continue

        fianl_result = resutls_dict[str(len(resutls_dict.keys()) - 1)]['fitted_result']

        fitted_parameters = fianl_result.x

        Gaussian_decom_modes_str = ''

        for index in range(0, len(fitted_parameters), 3):

            amplitude, center, sigma = fitted_parameters[index], fitted_parameters[index + 1], fitted_parameters[index + 2]

            center = center + search_start  # recover to GEDI waveform bins

            Gaussian_decom_modes_str = Gaussian_decom_modes_str + f'{amplitude:.2f},' + f'{center:.2f},' + f'{sigma:.2f},'

        output_str = shot_number + ',' + Gaussian_decom_modes_str

        outPutFile.write(output_str + '\n')
        outPutFile.flush()
        i = i + 1

        print(shot_number + r' progress = %.2f%%' % (i / len(dataframe) * 100,))


        # stepwise_fig, stepwise_ax = plt.subplots(figsize=(10, 6))
        #
        # stepwise_ax.plot(range(len(searching_waveform)), searching_waveform, c='gray', label='searching waveform',
        #                  linewidth=0.5)
        #
        # draw_Gaussian_fitted_modes(stepwise_ax, fianl_result.x, len(searching_waveform))
        # ########
        # stepwise_ax.axvline(zcross - search_start, label='ground', c='red')
        #
        # stepwise_ax.legend()
        #
        # plt.show()

    outPutFile.close()

def result_organized():
    outPutFile = open(file_path.Iterative_GAU_txt, 'a+')  # a+ 读写模式，没有文件,会创建
    outPutFile.seek(0)  # 移动到文件开头
    existed_shot_number_List = [line.strip() for line in outPutFile.readlines()]

    def format_numbers(data):
        # 使用正则表达式匹配小数点后超过两位的数字，并在其后插入逗号
        formatted_data = re.sub(r'(\d+\.\d{2})(\d+)', r'\1,\2', data)
        return formatted_data

    outPutFile = open(r'D:\Pycharm_Projects\Waveform_decompisition\result_output\gaussian_decomposition\Gau_decom_iterative_organ.txt', 'a+')  # a+ 读写模式，没有文件,会创建

    for result in existed_shot_number_List:
        parts = result.split(',', 1)  # 按第一个逗号分割
        before_comma = parts[0]
        after_comma = parts[1]
        organized_result = format_numbers(after_comma)
        final_result = before_comma + ',' + organized_result
        outPutFile.write(final_result + '\n')
        outPutFile.flush()

def save_fitted_wave(GAU_results_txt,save_txt_file):

    ##
    shot_number_list, decomposition_results = gaussian_samples.open_gaussian_decomposition_result(GAU_results_txt)
    i = 0
    waveform_template = np.zeros((len(shot_number_list), 1000), dtype=float)

    for GAU_parameter in decomposition_results:

        Gau_modes_str_list = GAU_parameter.strip().split(',')

        Gau_modes_array = np.array(Gau_modes_str_list).astype(np.float32)

        sort_modes_array, fitted_waveform = gaussian_samples.sort_gaussian_decomposition_results(Gau_modes_array, 1000)

        waveform_template[i, :] = fitted_waveform

        i = i + 1

    np.savetxt(save_txt_file, waveform_template, delimiter=',', fmt='%.3f')


if __name__ == '__main__':

    print('')
    #WRD_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\WRD\Manually_WRD.xlsx'

    #output_iterative_GD_results(file_path.RF_excel)

    #result_organized()
    GAU_results_txt = file_path.Iterative_GAU_txt
    fitted_waveform_result = file_path.RF_fitted_waveform_txt
    save_fitted_wave(GAU_results_txt,fitted_waveform_result)