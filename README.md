This repository is to implement gaussian decomposition for lidar waveform; 
The code is mostly based on (Hofton et al., 2000); Hofton M A, Minster J B, & Blair J B. Decomposition of laser altimeter waveforms[J]. Ieee Transactions on Geoscience and Remote Sensing. 2000, 38, 1989-1996. https://doi.org/10.1109/36.851780 
The code has been partially tested on GEDI (Global Ecosystem Dynamics Investigation) received waveforms.

Two kinds of models were developed to fit the GEDI waveform. 
The first fits all component waveforms using a Gaussian function (Gaussian decomposition). 
The second additionally uses an extended Gaussian function specifically for ground return fitting, as in the GEDI L2B product. However, the second model may not fully follow GEDI's practices because I do not know whether GEDI directly fits the ground return mode (located at the lowest peak of the waveform) using an extended Gaussian function or if they first perform Gaussian decomposition and then further fit the lowest component result using an extended Gaussian function. In my code, I replaced the Gaussian function with an extended Gaussian function to fit the lowest mode. Apart from this, all details are the same as in the Gaussian decomposition.


This is a preliminary version.

Usage:

I provide a test_data.xlsx, it includes waveforms and some fields derived from GEDI products
download this repository, run main.py;  you will see a fig of decomposition result; below is the test function


def test_gaussia_decomposition():

    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file_path)

    test_excel = os.path.join(project_root,'test_data.xlsx')

    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    shot_number = '79650300200248910'

    tx_parameters, rx_parameters = GEDI_waveform_parameters(dataframe, dataframe,shot_number)

    rx_waveform_value = rx_parameters['rx_waveform']

    stddev, gamma, sigma, search_start, search_end = dataframe.loc[shot_number, ['stddev', 'tx_eggamma', 'tx_egsigma', 'search_start', 'search_end']]

    searching_waveform, noise_std = GD_decom.get_smooth_waveform(rx_waveform_value, search_start, search_end)

    times_noise = 3

    stop_threshold = times_noise * noise_std

    #fitted_parameters = exGD_decom.waveform_decompose_exgaussian(searching_waveform, stop_threshold, sigma, gamma)

    fitted_parameters = GD_decom.waveform_decompose_gaussian(searching_waveform, stop_threshold, sigma)

    lmfit_fig, lmfit_ax = plt.subplots(figsize=(10, 6))

    lmfit_ax.plot(range(len(searching_waveform)), searching_waveform, c='gray', label='searching waveform',linewidth=0.5)

    draw_Gaussian_fitted_modes(lmfit_ax, fitted_parameters, len(searching_waveform))
    ########
    lmfit_ax.axvline(rx_parameters['zcross'] - rx_parameters['search_start'], label='ground', c='red')
    lmfit_ax.legend()
    plt.show()






