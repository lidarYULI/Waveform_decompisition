import numpy as np
import pandas as pd
from lmfit import Model
import gaussian_decomposition
import model_usage
# this script is mostly based on (Hofton, 2000);
# some parameters are designed for GEDI waveform
def initial_Gaussian_parameters(smoothed_waveform, inflation_x, derivation_x, noise_std, tx_sigma, points='derivative'):

    amplitudes, center_x, sigmas = [], [], []
    if points == 'inflection':
        amplitudes, center_x, sigmas = initial_paras_inflection(inflation_x, smoothed_waveform)
    if points == 'derivative':
        amplitudes, center_x, sigmas = initial_paras_derivative(derivation_x, inflation_x, tx_sigma, smoothed_waveform)

    initial_pandas = pd.DataFrame({'amplitude': amplitudes, 'center': center_x, 'sigma': sigmas})

    initial_pandas = initial_pandas.astype({'amplitude': 'float64', 'center': 'float64', 'sigma': 'float64'})

    amplitude_params = init_amplitude(initial_pandas, smoothed_waveform)

    parameters = amplitude_params.params

    for i in initial_pandas.index.tolist():
        prefix = 'g' + str(i) + '_'
        initial_pandas.loc[i, ['amplitude', 'center', 'sigma']] = parameters[prefix + 'amplitude'].value, \
                                                                  parameters[prefix + 'center'].value, parameters[
                                                                      prefix + 'sigma'].value
    if len(initial_pandas) <= 1:
        return initial_pandas
    else:
        filter_parameters = filter_initial_parameter(initial_pandas, noise_std)
        return filter_parameters

# flag some components as the important components; they will input to the modelling at first
def flag_gaussian(initial_parameters,noise_std,important_sigma = 6):

    # flag the important gaussian by 0
    importance_centers = []
    unimportance_centers = []
    for i in range(0,len(initial_parameters)):
        amplitude,center,sigma = initial_parameters.loc[i,['amplitude','center','sigma']]
        # in hofton's paper; > 3*noise_std and sigma >= half_width of the laser impulse; for GEDI pulse (14 ns long) the half_width is 6
        if (amplitude>= 3*noise_std) & (sigma >= important_sigma):
            initial_parameters.loc[i,['flag']] = 0
            importance_centers.append(center)
        else:
            unimportance_centers.append(center)
    if len(importance_centers) == 0:
        maximum_index = initial_parameters['amplitude'].idxmax()
        initial_parameters.loc[maximum_index, ['flag']] = 0
        importance_centers.append(initial_parameters.loc[maximum_index,'center'])
        unimportance_centers.remove(initial_parameters.loc[maximum_index,'center'])
    # record the distant ot the most closed important component
    min_distances = []
    for center_un in unimportance_centers:
        # to the most closed mode
        distance_init = abs(center_un - importance_centers[0])
        # sort
        for center_im in importance_centers[1:]:
            distance_init = abs(center_un - center_im) if abs(center_un - center_im)<distance_init else distance_init

        min_distances.append(distance_init)
    # set the rank to each component by its distance to the important one.
    min_distance_index = np.argsort(np.array(min_distances))
    rank = np.arange(len(min_distance_index)) + 1
    rank[min_distance_index] = rank
    initial_parameters.loc[initial_parameters['flag'] != 0, 'flag'] = rank

    return initial_parameters

# initiate amplitudes of all waveform components; based on Hofton, 2000
def init_amplitude(init_parameters,waveform):

    index_list = init_parameters.index.tolist()
    model = None
    for i in index_list:
        prefix = 'g' + str(i) + '_'
        gaussian_model = Model(model_usage.gaussian, prefix=prefix)
        if model == None:
            model = gaussian_model
        else:
            model = model + gaussian_model

    params = model.make_params()
    for i in index_list:
        prefix = 'g' + str(i) + '_'
        amplitude,center,sigma = init_parameters.loc[i,'amplitude'],init_parameters.loc[i,'center'],init_parameters.loc[i,'sigma']
        params.add(prefix+'amplitude',value = amplitude, min = 0, max = np.inf)
        # these two parameters are fixed
        params.add(prefix+'center',value = center,vary = False)
        params.add(prefix+'sigma',value = sigma, vary = False)

    initial_amplitude_parameters = model.fit(waveform,params,x = np.array(range(len(waveform))))

    return initial_amplitude_parameters

# initiate amplitudes of all waveform components; based on Hofton, 2000
## model the least mode by extended gaussian; based on Tang, 2019, ATBD
def init_amplitude_ex(init_parameters, waveform):
    # 确定初始拟合幅度; Gaussian models
    # at least two modes existed
    index_list = init_parameters.index.tolist()
    model = None

    if len(index_list) == 1:
        prefix = 'g' + str(index_list[-1]) + '_'
        model = Model(model_usage.ex_gaussian, prefix=prefix)
    else:
        for i in index_list[0:-1]:
            prefix = 'g' + str(i) + '_'
            gaussian_model = Model(model_usage.gaussian, prefix=prefix)
            if model == None:
                model = gaussian_model
            else:
                model = model + gaussian_model
        # add the extended Gaussian models
        prefix = 'g' + str(index_list[-1]) + '_'
        model = model + Model(model_usage.ex_gaussian, prefix=prefix)

    params = model.make_params()
    for i in index_list:
        prefix = 'g' + str(i) + '_'

        amplitude, center, sigma = init_parameters.loc[i, 'amplitude'], init_parameters.loc[i, 'center'], init_parameters.loc[i, 'sigma']

        params.add(prefix + 'amplitude', value=amplitude, min=0, max=np.inf)

        # these two parameter don't change
        params.add(prefix + 'center', value=center, vary=False)

        params.add(prefix + 'sigma', value=sigma, vary=False)
    ## add gamma parameter to the model
    prefix = 'g' + str(index_list[-1]) + '_'
    ground_row = init_parameters.iloc[-1]
    ground_gamma = ground_row['gamma']
    params.add(prefix + 'gamma', value=ground_gamma, vary=False)

    amplitude_result = model.fit(waveform, params, x = np.array(range(len(waveform))))

    return amplitude_result

def filter_initial_parameter(initial_pandas,noise_std,lowest_sigma = 1):

    # filter some unreasonable initial parameters
    filter_initparameters = pd.DataFrame({'amplitude': [], 'center': [], 'sigma': []})
    # save the initial position
    i = 0
    for index in initial_pandas.index.to_list():
        amplitude, center, sigma = initial_pandas.loc[index, ['amplitude','center','sigma']]
        # remove a mode with small amplitude or sigma | (sigma < 3*self.lowest_sigma)
        if ((amplitude <= 3*noise_std) | (sigma < 3*lowest_sigma)):
            continue
        else:
            filter_initparameters.loc[i, :] = amplitude, center, sigma
            i = i + 1
    return filter_initparameters

# Gaussian decomposition initial parameters
def initial_paras_inflection(waveform,inflection_x):
    center_x = []
    amplitudes = []
    sigmas = []
    for i in range(0, len(inflection_x), 2):
        # the continuous two inflection points
        x1, x2 = inflection_x[i], inflection_x[i + 1]
        # the initial Gaussian position
        center = (x1 + x2) / 2
        center_amplitude = waveform[int(center)]
        center_x.append(center)
        # the half-width. Hofton said the half-width is sigma;
        sigma = abs(x1 - x2) / 2
        sigmas.append(sigma)
        amplitudes.append(center_amplitude)
    return amplitudes, center_x, sigmas

def initial_paras_derivative(derivation_x, inflection_x, tx_sigma, waveform):
    # tx_sigma; this can be from GEDI L2A product; it is the sigma used in txwaeform fitting
    center_x = []
    amplitudes = []
    sigmas = []
    for center in derivation_x:
        amplitudes.append(waveform[int(center)])
        try:
            sigma = derivation_sigma(center, inflection_x)
        except:
            sigma = tx_sigma
        sigmas.append(sigma)
        center_x.append(center)

    return amplitudes, center_x, sigmas

# inflection points generation from a waveform
def inflection_points(waveform):
    x = np.arange(len(waveform))
    x_inflection = []
    y_inflection = []
    k = 3
    S1 = waveform[k - 2] + waveform[k] - 2 * waveform[k - 1]
    for xi, yi in zip(x[3:-3], waveform[3:-3]):
        S2 = waveform[k - 1] + waveform[k + 1] - 2 * waveform[k]
        if S1 * S2 < 0:
            x_inflection.append(xi)
            y_inflection.append(yi)
        S1 = S2
        k = k + 1
    ## filtering inflection points, the inflection points should be removed if the number is not even
    # x_inflection_filter, y_inflection_filter = self.filter_inflection(x_inflection,y_inflection,y)
    return np.array(x_inflection), np.array(y_inflection)

# the filtering may be not right;
# I consider the center's value between two nearby inflections on both side should be at least larger than one of them
# if not, an inflection may exist alone. this inflection can not be used to calculate sigma in Gaussian component
def filter_inflection(x_inflection,y_inflection,y):

    if len(x_inflection)%2 == 0:
        return x_inflection,y_inflection
    # judge the wrong inflection points
    i = 0
    x_inflection_copy = x_inflection.copy()
    y_copy = y.copy()

    while i < len(x_inflection_copy) - 1:
        inflection_left = x_inflection_copy[i]
        inflection_right = x_inflection_copy[i + 1]
        x_center = (inflection_left + inflection_right) / 2
        y_center = y_copy[int(x_center)]
        if (y_center <= y_copy[int(inflection_left)]) | (y_center <= y_copy[int(inflection_right)]):
            x_inflection = x_inflection[x_inflection!=inflection_left]
            y_inflection = y_inflection[y_inflection!=y_copy[int(inflection_left)]]
            i = i + 1
        else:
            i = i + 2
    return x_inflection,y_inflection

# add initial sigma for derivation initial Gaussian parameters
# the sigma is calculated by the distance between two inflection around the center
def derivation_sigma(center,inflation_points):

    offset = center - inflation_points
    left_part_count = len(offset[offset > 0])
    right_part_count = len(offset[offset < 0])
    if (left_part_count > 0) & (right_part_count > 0):
        left_inflation = inflation_points[left_part_count - 1]
        right_inflation = inflation_points[len(offset) - right_part_count]
        sigma = (right_inflation - left_inflation) / 2
        # set a default sigma if inflections are not paired
        if sigma < 0:
            sigma = 6
            return sigma
        else:
            return sigma
    else:
        return 6

# the first derivation of curve is equal to 0,
# another way to determine the initial component Gaussian
def derivative_points(waveform):
    maxima = (waveform[1:-1] > waveform[:-2]) & (waveform[1:-1] > waveform[2:])
    # the index of fist derivative
    extrema_idx = np.where(maxima)[0] + 1  # add 1 to correct the index
    return extrema_idx, waveform[extrema_idx]