import numpy as np
from scipy.special import erfc
from lmfit import Model

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