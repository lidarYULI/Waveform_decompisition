import numpy as np
import initial_parameters
import pandas as pd
import GEDI_waveform_processing
import os
from RF_ground_identification.gaussian_samples import file_path
from RF_ground_identification import derivative_samples
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def waveform_PCA():
    # classify all of waveforms
    rf_excel = file_path.RF_excel

    rf_dataframe = pd.read_excel(rf_excel, dtype={'shot_number': str}, index_col=0)

    waveform_template = np.zeros((len(rf_dataframe), 1000))

    for i, shot_number in zip(np.arange(0, len(rf_dataframe)), rf_dataframe.index.values.tolist()):

        rx_waveform_str, search_start, search_end = rf_dataframe.loc[
            shot_number, ['rxwaveform', 'search_start', 'search_end']].values  # zcross

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 2)

        normalized_waveform = (smooth_waveform - np.min(smooth_waveform)) / (
                    np.max(smooth_waveform) - np.min(smooth_waveform))

        if (len(normalized_waveform) > 1000):
            exceed_length = len(normalized_waveform) - 1000

            waveform_template[i, :] = normalized_waveform[int(exceed_length / 2):1000 + int(exceed_length / 2)]
    # PCA reduces waveform to 50 dimensionality

    waveform_transformed = PCA(n_components=0.95).fit_transform(waveform_template)

    PCA_txt_file = file_path.RF_waveform_PCA_txt

    np.savetxt(PCA_txt_file, waveform_transformed, delimiter = ',', fmt = '%.3f')

def waveform_classification():

    PCA_txt_file = file_path.RF_waveform_PCA_txt

    waveform_features_PCA = np.loadtxt(PCA_txt_file, delimiter=',', dtype=float)

    # GMM classification; using variational bayesian GM to avoid the singularities found in expectation-maximization solutions
    bic_list = []
    for i in range(2,20):

        GMM = GaussianMixture(n_components = i,covariance_type="full")

        GMM.fit(waveform_features_PCA)

        bic_list.append(GMM.bic(waveform_features_PCA))

    GMM_fig, GMM_ax = plt.subplots(figsize=(10, 6))
    GMM_ax.plot(np.arange(2,20),bic_list)
    plt.show()
        # label = bgm.predict(waveform_features_PCA)
        #
        # proba = bgm.predict_proba(waveform_features_PCA)

    #print(bgm.weights_)

if __name__ == '__main__':
    waveform_classification()
    #waveform_PCA()


