import numpy as np
from scipy.optimize import least_squares
import initial_parameters
import model_usage
import GEDI_waveform_processing as GEDI_processing
import smooth_filter
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

    # get searching waveform
    searching_waveform = np.array(normalized_waveform[int(search_start):int(search_end)])

    return searching_waveform, noise_std
