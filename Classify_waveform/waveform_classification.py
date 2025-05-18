import os.path
import numpy as np
import pandas as pd
import GEDI_waveform_processing
from files_access import file_path
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from Regress_ground.Samples_generation import gaussian_samples
from sklearn.cluster import KMeans

def waveform_PCA(rf_excel,save_txt_file):
    # classify waveforms
    rf_dataframe = pd.read_excel(rf_excel, dtype={'shot_number': str}, index_col=0)

    waveform_template = waveform_array(rf_dataframe)
    # top 10 components can account for variance more than 95%
    pca = PCA(n_components = 0.98)

    principle_component = pca.fit_transform(waveform_template)

    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

    np.savetxt(save_txt_file, principle_component, delimiter = ',', fmt = '%.3f')

    return explained_variance_ratio

def fitted_waveform_PCA(GAU_results_txt,save_txt_file):

    ##
    shot_number_list, decomposition_results = gaussian_samples.open_gaussian_decomposition_result(GAU_results_txt)
    i = 0
    waveform_template = np.zeros((len(shot_number_list), 1000), dtype=float)

    for GAU_parameter in decomposition_results:

        Gau_modes_str_list = GAU_parameter.strip().split(',')

        Gau_modes_array = np.array(Gau_modes_str_list).astype(np.float32)

        sort_modes_array, fitted_waveform = gaussian_samples.sort_gaussian_decomposition_results(Gau_modes_array, 1000)

        normalized_waveform = (fitted_waveform - np.min(fitted_waveform)) / (
                np.max(fitted_waveform) - np.min(fitted_waveform))

        waveform_template[i, :] = normalized_waveform

        i = i + 1
        # top 10 components can account for variance more than 95%
    pca = PCA(n_components=0.98)

    principle_component = pca.fit_transform(waveform_template)

    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

    np.savetxt(save_txt_file, principle_component, delimiter=',', fmt='%.3f')

    return explained_variance_ratio

def waveform_array(rf_dataframe):
    # classify waveforms
    waveform_template = np.zeros((len(rf_dataframe), 1000), dtype=float)

    for i, shot_number in zip(np.arange(0, len(rf_dataframe)), rf_dataframe.index.values.tolist()):

        rx_waveform_str, search_start, search_end, mean = rf_dataframe.loc[
            shot_number, ['rxwaveform', 'search_start', 'search_end', 'mean']].values  # zcross

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

        array_saving = np.full((1, 1000), mean, dtype=float)

        if (len(smooth_waveform) > 1000):
            exceed_length = len(smooth_waveform) - 1000
            array_saving[0, :] = smooth_waveform[int(exceed_length / 2):int(exceed_length / 2) + 1000]
        else:
            array_saving[0, 0:len(smooth_waveform)] = smooth_waveform

        # normalized_waveform = (array_saving - np.min(array_saving)) / (
        #         np.max(array_saving) - np.min(array_saving))

        normalizer = Normalizer(norm="l2")

        normalized_waveform = normalizer.fit_transform(array_saving)

        waveform_template[i, :] = normalized_waveform

    return waveform_template

def cluster_number(PCA_txt_file, criteria = 'BIC'):

    waveform_features_PCA = np.loadtxt(PCA_txt_file, delimiter=',', dtype=float)

    #scaler = StandardScaler()

    #waveform_array_scaled = scaler.fit_transform(waveform_features_PCA)
    waveform_array_scaled = waveform_features_PCA
    # GMM classification; using variational bayesian GM to avoid the singularities found in expectation-maximization solutions
    criteria_list = []
    for i in range(2,20):

        GMM = GaussianMixture(n_components = i,covariance_type="full")
        GMM.fit(waveform_array_scaled)

        if criteria == 'BIC':
            criteria_list.append(GMM.bic(waveform_array_scaled))
        elif criteria == 'silhouette':
            labels = GMM.predict(waveform_array_scaled)  # 预测簇标签
            score = silhouette_score(waveform_array_scaled, labels)  # 计算轮廓系数
            criteria_list.append(score)

    GMM_fig, GMM_ax = plt.subplots(figsize=(10, 6))
    GMM_ax.plot(np.arange(2,20),criteria_list)
    GMM_ax.set_xticks(np.arange(2,20))
    plt.show()
        # label = bgm.predict(waveform_features_PCA)
        #
        # proba = bgm.predict_proba(waveform_features_PCA)

    #print(bgm.weights_)

def classify_GMM(excel,cluster_num):

    rf_dataframe = pd.read_excel(excel, dtype={'shot_number': str}, index_col=0)

    waveform_array_value = waveform_array(rf_dataframe)

    scaler = StandardScaler()

    waveform_array_scaled = scaler.fit_transform(waveform_array_value)

    pca = PCA(n_components = 0.98)

    principle_component = pca.fit_transform(waveform_array_scaled)

    GMM = GaussianMixture(n_components = cluster_num, covariance_type="full")

    GMM.fit(principle_component)

    labels = GMM.predict(principle_component)

    probabilities = GMM.predict_proba(principle_component)

    probabilities[:, :] = np.around(probabilities[:, :], 3)

    labels_array = labels.reshape(-1,1).astype(int)

    output_dataframe  = pd.DataFrame()

    output_dataframe['shot_number'] = rf_dataframe.index.values.tolist()

    output_dataframe['label'] = labels_array

    for i in np.arange(cluster_num):

        output_dataframe[f'label_{i}'] = probabilities[:,i]

    output_name = os.path.splitext(os.path.basename(excel))[0]

    output_dataframe.to_excel(os.path.dirname(excel),f'{output_name}_GMM_result.xlsx')

def classify_KMEANS(PCA_txt_file,n = 8):

    waveform_features_PCA = np.loadtxt(PCA_txt_file, delimiter=',', dtype=float)

    scaler = StandardScaler()

    waveform_array_scaled = scaler.fit_transform(waveform_features_PCA)

    kmeans = KMeans(n_clusters = n, random_state=42)

    labels = kmeans.fit_predict(waveform_array_scaled)

    rf_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    rf_dataframe['kmeans_labels'] = labels

    rf_dataframe.to_excel(file_path.RF_excel)

############### classification for ground return mode features

def ground_return_features_PCA(excel,save_txt_file):
    # classify waveforms
    rf_dataframe = pd.read_excel(excel, dtype={'shot_number': str}, index_col=0)

    all_array = rf_dataframe.values
    # top 10 components can account for variance more than 95%
    pca = PCA(n_components = 0.98)

    principle_component = pca.fit_transform(all_array)

    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

    np.savetxt(save_txt_file, principle_component, delimiter = ',', fmt = '%.3f')

    return explained_variance_ratio

if __name__ == '__main__':

    #ground_return_features_PCA(file_path.RF_ground_features_excel,file_path.RF_ground_features_PCA_txt)

    #fitted_waveform_PCA(file_path.Iterative_GAU_txt,file_path.RF_fitted_waveform_PCA_txt)

    #waveform_PCA(file_path.RF_excel,file_path.RF_waveform_PCA_txt)

    classify_KMEANS(file_path.RF_waveform_PCA_txt,n = 4)



