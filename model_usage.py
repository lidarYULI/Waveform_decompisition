import numpy as np
from scipy.special import erfc
from lmfit import Model
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
# define the modeling function
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

# build a mode to fit waveform; overlap  gaussian components all together
def model_Gaussian(x,params):
    # x is a sequential array with length of waveform
    fit_y = 0
    for i in np.arange(0, len(params), 3):
        amplitude, center, sigma = params[i], params[i + 1], params[i + 2]
        fit_y = fit_y + amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    return fit_y

# set boundary for the parameters used in model fitting
def generate_bound(params):
    # params: n*3
    # sort by mode center
    params_sort = params[np.argsort(params[:,1])]
    # generate bound used in modeling
    lp_bounds = []
    up_bounds = []
    for i in range(0, len(params_sort)):

        amplitude, center, sigma = params_sort[i,0], params_sort[i,1], params_sort[i,2]
        ## first mode
        if i == 0:
           left_center = center - 10
           if left_center < 0:
                left_center = 0
        else:
            left_center = params_sort[i - 1, 1]
        if i == len(params_sort) - 1:
            right_center = center + 10
        else:
            right_center = params_sort[i+1,1]
        # the bound for sigma is also variable, paper recommend the smallest sigma is the sigma of transmitted waveform
        lp_bounds = lp_bounds + [0, left_center, sigma]
        up_bounds = up_bounds + [np.inf, right_center, sigma * 5]
    # tuple_bound = tuple([lp_bounds,up_bounds])
    return params_sort.flatten(), lp_bounds, up_bounds


def generate_bound_simple(params):
    # params: n*3
    # sort by mode center
    params_sort = params[np.argsort(params[:,1])]
    # generate bound used in modeling
    lp_bounds = []
    up_bounds = []
    for i in range(0, len(params_sort)):

        amplitude, center, sigma = params_sort[i,0], params_sort[i,1], params_sort[i,2]
        ## first mode
        if i == 0:
           left_center = center - 10
           if left_center < 0:
                left_center = 0
        else:
            left_center = params_sort[i - 1, 1] if center - params_sort[i - 1, 1] < 10 else center - 10
        if i == len(params_sort) - 1:
            right_center = center + 10
        else:
            right_center = params_sort[i + 1, 1] if params_sort[i + 1, 1] - center < 10 else center + 10
        # the bound for sigma is also variable, paper recommend the smallest sigma is the sigma of transmitted waveform
        lp_bounds = lp_bounds + [0, left_center, sigma]
        up_bounds = up_bounds + [np.inf, right_center, sigma * 5]
    # tuple_bound = tuple([lp_bounds,up_bounds])
    return params_sort.flatten(), lp_bounds, up_bounds

# arbitrary settings for GEDI waveform
def generate_bound_GEDI(params):
    # generate bound used in modeling
    lp_bounds = []
    up_bounds = []
    for i in range(0, len(params), 3):
        amplitude, center, sigma = params[i], params[i + 1], params[i + 2]
        center_lp = center - 10
        center_rp = center + 10
        if center_lp < 0:
            center_lp = 0
        lp_bounds = lp_bounds + [0, center_lp, sigma]
        up_bounds = up_bounds + [np.inf, center_rp, np.inf]
    # tuple_bound = tuple([lp_bounds,up_bounds])
    return lp_bounds, up_bounds

# the first derivation of curve is equal to 0,
# another way to determine the initial component Gaussian
def derivative_points(waveform):
    maxima = (waveform[1:-1] > waveform[:-2]) & (waveform[1:-1] > waveform[2:])
    # the index of fist derivative
    extrema_idx = np.where(maxima)[0] + 1  # add 1 to correct the index
    return extrema_idx, waveform[extrema_idx]

# GEDI takes the lowest gaussian mode as the ground return mode
# they fit the ground return mode by Extended Gaussian function; add a gamma parameter
def model_exGaussian(x,params):
    # x is a sequential array with length of waveform
    fit_y = 0
    for i in np.arange(0, len(params) - 4, 3):
        amplitude, center, sigma = params[i], params[i + 1], params[i + 2]
        fit_y = fit_y + amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    # just in case the ground return mode is fitted by extended Gaussian function
    last_mode = ex_gaussian(x,params[-4], params[-3], params[-2],params[-1])

    return fit_y + last_mode

# below is used in GEDI L2B product, Tang and Armston, 2019, ATBD
def ex_gaussian(x, amplitude, center, sigma, gamma):
    # this formula is from other resources instead of GEDI ATBD, but it fits rightly
    erfca = erfc((center + gamma * (sigma ** 2) - x) / (np.sqrt(2) * sigma))  # ERROR part
    input_to_exp = gamma * (center - x + gamma * sigma ** 2 / 2)
    input_to_exp = np.clip(input_to_exp, -700, 700)
    exponential_part = np.exp(input_to_exp)
    return amplitude * gamma * 2 * exponential_part * erfca


# fit ground return based on extended Gaussian function
def ground_fit_exgaussian(normalized_waveform,select_zcross,sigma,gamma):

    def cost_effective(params, x, y_obser):
        # return (np.sum((model(params,x) - y_obser)/noise_std))**2
        amplitude, center, sigma, gamma = params
        return (ex_gaussian(x,amplitude, center, sigma, gamma) - y_obser) **2

    ground_amplitude = normalized_waveform[select_zcross]

    # fit_amplitude = ground_amplitude,
    fit_amplitude = calculate_ex_initial_amplitude(ground_amplitude,select_zcross,sigma,gamma)

     # below boundary setting is set by many times of trail;
    lp_bounds = [0, select_zcross - 4, sigma * 0.8, gamma * 0.98]
    up_bounds = [np.inf, select_zcross + 4, sigma * 10, gamma * 1] #gamma * 1.05

    # transect_waveform is also unknown; set arbitrarily
    transect_wavefrm = normalized_waveform[select_zcross + 4:select_zcross + 20]

    x = np.arange(select_zcross + 4, select_zcross + 4 + len(transect_wavefrm), 1)

    initial_paras = np.array([fit_amplitude, select_zcross, sigma, gamma])

    result = least_squares(cost_effective, initial_paras, args=(x, transect_wavefrm), method='trf',bounds=(lp_bounds, up_bounds))

    fitted_parameters = result.x

    return fitted_parameters

# fit ground return based on Gaussian function
def ground_fit_Gau(normalized_waveform,select_zcross,sigma):

    ground_amplitude = normalized_waveform[select_zcross]
    # below boundary setting is set by many times of trail;
    lp_bounds = [ground_amplitude * 0.8, select_zcross - 4, sigma*0.8]

    up_bounds = [ground_amplitude * 2, select_zcross + 4, sigma * 3]

    # transect_wavefrm = normalized_waveform[fit_start:fit_end]
    #
    # x = np.arange(fit_start, fit_start + len(transect_wavefrm), 1)
    transect_wavefrm = normalized_waveform[select_zcross - 12:select_zcross + 12]

    x = np.arange(select_zcross - 12, select_zcross - 12 + len(transect_wavefrm), 1)

    initial_paras = [ground_amplitude, select_zcross, sigma]

    popt, pcov = curve_fit(gaussian, x, transect_wavefrm, p0=initial_paras, bounds=[lp_bounds, up_bounds])

    return popt


def calculate_ex_initial_amplitude(y,center,sigma,gamma):
    erfca = erfc((center + gamma * (sigma ** 2) - center) / (np.sqrt(2) * sigma))  # 计算误差函数部分
    input_to_exp = gamma * (center - center + gamma * sigma ** 2 / 2)
    exponential_part = np.exp(input_to_exp)
    amplitude = y/ (gamma * 2 * exponential_part * erfca)
    return amplitude