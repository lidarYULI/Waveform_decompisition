import numpy as np

# smooth to denoise
def gaussian_smooth(waveform, smooth_width=None):
    '''
    in hofton,2001, they take half_width as sigma
    :param x: location/time inforamtion
    :param y: amplitude
    :param sigma: sigma has a great effect on smoothing
    :return: smooth waveform
    '''
    x = np.arange(0, len(waveform), 1)
    y_g = []
    if smooth_width is None:
        # using GEDI default smooth for noise
        sigma = 6.5 / (2 * np.sqrt(2 * np.log(2)))
    else:
        sigma = smooth_width / (2 * np.sqrt(2 * np.log(2)))
    for i in x:
        filter_g = gaussian_filter(x, i, sigma)

        yi = np.sum(filter_g * waveform)
        y_g.append(yi)

    return y_g

# one dimensional gaussian filter
def gaussian_filter(x,u,sigma):
    # half width and sigma are different
    c = (-1/2)*((x-u)/sigma)**2

    g = 1/(sigma*np.sqrt(2*np.pi))*np.power(np.e,c)

    g = g/np.sum(g)

    return g


# can be used to calculate the mean and standard deviation of a GEDI waveform
def mean_noise_level(waveform):
    '''
    :param waveform: the full waveform after smoothing
    :return: mean and standard deviation of noise
    '''
    # take the first and last 100 bins as noise waveform, only be possible for GEDI waveform
    last_10bins = np.concatenate([waveform[0:100],waveform[-100:]])
    return np.mean(last_10bins),np.std(last_10bins)