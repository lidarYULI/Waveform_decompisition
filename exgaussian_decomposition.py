from scipy.special import erfc
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import initial_parameters
import pandas as pd
import numpy as np
import model_usage


def waveform_decompose_exgaussian(valid_waveform, stddev, sigma, gamma, points='derivative'):
    # valid_waveform[valid_waveform < 0] = 0
    inflation_x, inflation_y = initial_parameters.inflection_points(valid_waveform)

    derivation_x, derivation_y = model_usage.derivative_points(valid_waveform)

    filtered_parameters = initial_Gaussian_parameters_exgaussian(valid_waveform, inflation_x, derivation_x, stddev, sigma, gamma,points=points)

    ranked_parameters = initial_parameters.flag_gaussian(filtered_parameters, stddev)

    # fit least square
    # Gaussian fit
    result = LM_fitted_eg_ground(valid_waveform, ranked_parameters, stddev)

    fitted_parameters = result.x

    return fitted_parameters

    # this decomposition method is for WRD process

# Gaussian decomposition using extended Gaussian function for ground return modeling
# maybe this is not the practice in GEDI product
def initial_Gaussian_parameters_exgaussian(smoothed_waveform, inflation_x, derivation_x, noise_std, tx_sigma, tx_gamma, points='derivative'):

    amplitudes, center_x, sigmas = [], [], []
    if points == 'inflection':
        amplitudes, center_x, sigmas = initial_parameters.initial_paras_inflection(inflation_x, smoothed_waveform)
    if points == 'derivative':
        amplitudes, center_x, sigmas = initial_parameters.initial_paras_derivative(derivation_x, inflation_x, tx_sigma, smoothed_waveform)

    initial_pandas = pd.DataFrame({'amplitude': amplitudes, 'center': center_x, 'sigma': sigmas,'gamma': tx_gamma})
    initial_pandas = initial_pandas.astype({'amplitude': 'float64', 'center': 'float64', 'sigma': 'float64', 'gamma': 'float64'})
    amplitude_params = initial_parameters.init_amplitude_ex(initial_pandas, smoothed_waveform)

    parameters = amplitude_params.params

    for i in initial_pandas.index.tolist():
        prefix = 'g' + str(i) + '_'
        initial_pandas.loc[i, ['amplitude', 'center', 'sigma']] = parameters[prefix + 'amplitude'].value, parameters[prefix + 'center'].value, parameters[prefix + 'sigma'].value

    # filter if multiple modes exist
    if len(initial_pandas) <= 1:
        return initial_pandas
    else:
        filter_parameters = initial_parameters.filter_initial_parameter(initial_pandas, noise_std)
        filter_parameters['gamma'] = tx_gamma
        return filter_parameters

def LM_fitted_eg_ground(y, init_modes, stddev, method='trf'):
    init_gamma = init_modes.loc[len(init_modes) - 1, 'gamma']
    # generate the bound
    x = np.arange(0, len(y))
    # get flatten parameters and its boundary
    important_mode = np.empty((0, 3))
    result = None
    ## add more Gaussian components; the smaller the value of "flag", the more important the mode
    for i in range(len(init_modes)):
        new_modes = init_modes.loc[init_modes['flag'] == i, ['amplitude', 'center', 'sigma']].values
        # iteratively add an unimportant mode
        important_mode = np.vstack([important_mode, new_modes])
        # lp_bounds, up_bounds = self.generate_bound_GEDI(important_params, init_parameters)
        sort_params_flatten, lp_bounds, up_bounds = set_boundary_with_gamma(important_mode,init_gamma)
        result = least_square_modeling(sort_params_flatten, lp_bounds, up_bounds, x, y, stddev, method=method)
        fitted_parameters = result.x
        residual_mean = np.mean((y - model_usage.model_exGaussian(x,fitted_parameters)) ** 2)
        square_root = np.sqrt(residual_mean)
        if square_root < stddev:
            return result
    return result

def least_square_modeling(params,lp_bounds,up_bounds,x, y, noise_std,method = 'trf'):

    def cost_effective(params, x, y_obser):
        # return (np.sum((model(params,x) - y_obser)/noise_std))**2
        return ((model_usage.model_exGaussian(x, params) - y_obser) / noise_std) ** 2

    if method == 'lf':
        result = least_squares(cost_effective, params, args=(x, y), method='lf')
    else:
        result = least_squares(cost_effective, params, args=(x, y), method='trf',bounds=(lp_bounds, up_bounds))

    return result

# add gamma coefficient and its bound to ground return parameters
def set_boundary_with_gamma(modes,init_gamma):
    # lp_bounds, up_bounds = self.generate_bound_GEDI(important_params, init_parameters)
    sort_params_flatten, lp_bounds, up_bounds = model_usage.generate_bound(modes)

    sort_params_flatten = np.append(sort_params_flatten, init_gamma)
    lp_bounds.append(init_gamma * 0.9)  # boundary setting is based on result of GEDI products
    up_bounds.append(init_gamma * 1.2)

    return sort_params_flatten, lp_bounds, up_bounds

