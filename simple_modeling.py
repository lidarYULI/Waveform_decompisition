import numpy as np
from scipy.optimize import curve_fit
import model_usage

# curve fit, this method seems more easy and effective
def curve_fit_sci_extended_gau(y, params):

    non_gamma_params = params[0:-1]

    gamma = params[-1]
    # left_bounds,right_bounds = self.generate_bound(non_gamma_params,init_parameters)
    left_bounds, right_bounds = model_usage.generate_bound_GEDI(non_gamma_params)
    # if ground return is moulded by ex_Gaussian
    left_bounds.append(gamma * 0.9)
    right_bounds.append(gamma * 1.2)
    bounds = (left_bounds ,right_bounds)
    x_data = np.arange(len(y))
    # y[y<0] = 0
    # print(params)
    results = curve_fit(model_usage.model_exGaussian,x_data, y, p0 = params ,bounds=bounds)
    popt, pcov = results[0] ,results[1]
    return popt, pcov

# curve fit, this method seems more easy and effective
def curve_fit_sci_gau(y, params):

    left_bounds, right_bounds = model_usage.generate_bound_GEDI(params)
    bounds = (left_bounds, right_bounds)
    x_data = np.arange(len(y))

    results = curve_fit(model_usage.model_Gaussian, x_data, y, p0=params, bounds=bounds)
    popt, pcov = results[0], results[1]
    ### popt includes the fitted parameters
    return popt, pcov

