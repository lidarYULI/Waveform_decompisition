import pandas as pd
from files_access import file_path
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import r2_score
import GEDI_waveform_processing as GEDI_process
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
def RF_regression_proportion(feature_excel, feature_columns, groupby = 'site',proportion = 0.1):
    '''
    :param feature_excel:
    # feature_excel:
    # file_path.RF_modes_features_excel when using gaussian decomposition mode derived features
    # file_path.RF_derivative_features_excel when using derivative derived features
    :param feature_columns: features used in regression
    :param groupby: how to classify dataset when training; kmeans_labels/site
    :return:
    '''
    # 定义参数网格
    param_grid = {
        'n_estimators': randint(50, 500),
        'max_depth': [5, 10, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    mode_feature_dataframe = pd.read_excel(feature_excel ,dtype = {'shot_number' :str} ,index_col = 0)

    group_classification = mode_feature_dataframe.groupby(groupby)

    features_importance = pd.DataFrame(columns = feature_columns[:-1])

    with open(file_path.RF_regression_model_record_txt, 'a') as optimal_txt:

        for group in group_classification:

            label, dataframe = group[0], group[1]

            print(label,proportion)

            randomly_selected_dataframe = dataframe.sample(frac=proportion, random_state=42)

            remaining_dataframe = dataframe.drop(randomly_selected_dataframe.index)

            samples_regression_array,remaining_array = randomly_selected_dataframe[feature_columns].values, remaining_dataframe[feature_columns].values

            samples_array = samples_regression_array[~np.isnan(samples_regression_array).any(axis=1)]

            test_array = remaining_array[~np.isnan(remaining_array).any(axis=1)]
            # Random forest regression
            rf_regression = RandomForestRegressor(random_state=42)
            # 网格搜索
            grid_search = RandomizedSearchCV(rf_regression, param_distributions = param_grid, n_iter = 50, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

            X, y = samples_array[:,:-1], samples_array[:,-1]

            grid_search.fit(X, y)

            # fit
            R2, RMSE, regression_model = RF_regression(samples_array, grid_search.best_params_)

            # test
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            y_predict = regression_model.predict(x_test)

            test_RMSE = np.sqrt(mean_squared_error(y_test, y_predict))

            test_R2 = r2_score(y_test, y_predict)

            params = regression_model.get_params()

            output_str = (f'site {label}, proportion: {proportion:.2f}: fit_R2: {R2:.2f},RMSE: {RMSE:.2f},test_R2: {test_R2:.2f},test_RMSE: {test_RMSE:.2f}' +
                          f'\n {params}')

            features_importance.loc[label,:] = regression_model.feature_importances_

            optimal_txt.write(output_str + '\n')

            optimal_txt.flush()

    features_importance.to_excel(file_path.variable_importance_save_excel)

def RF_regression(feature_dependent,RF_parameters):

    rf_regression = RandomForestRegressor(
        n_estimators = RF_parameters['n_estimators'],  # the number of decision tree

        max_depth = RF_parameters['max_depth'],  # maximum depth of a tree

        min_samples_leaf = RF_parameters['min_samples_leaf'],

        min_samples_split = RF_parameters['min_samples_split'],

        max_features=RF_parameters['max_features'],

        random_state=42,  # random
    )
    x,y = feature_dependent[:,:-1], feature_dependent[:,-1]

    rf_regression.fit(x,y)

    y_predict = rf_regression.predict(x)

    RMSE = np.sqrt(mean_squared_error(y, y_predict))

    R2 = r2_score(y, y_predict)

    return R2, RMSE, rf_regression

def RF_regression_proportion_test(feature_excel, feature_columns, groupby = 'site'):
    '''
    :param feature_excel:
    # feature_excel:
    # file_path.RF_modes_features_excel when using gaussian decomposition mode derived features
    # file_path.RF_derivative_features_excel when using derivative derived features
    :param feature_columns: features used in regression
    :param groupby: how to classify dataset when training; kmeans_labels/site
    :return:
    '''
    # 定义参数网格
    param_grid = {
        'n_estimators': randint(50, 500),
        'max_depth': [5, 10, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    mode_feature_dataframe = pd.read_excel(feature_excel ,dtype = {'shot_number' :str} ,index_col = 0)

    RF_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    group_classification = mode_feature_dataframe.groupby(groupby)

    with open(file_path.RF_regression_proportion_output_txt, 'a') as optimal_txt:

        for group in group_classification:

            label, dataframe = group[0], group[1]

            for proportion in np.arange(0.02,0.52,0.03):
                print(label,proportion)
                randomly_selected_dataframe = dataframe.sample(frac=proportion, random_state=42)

                remaining_dataframe = dataframe.drop(randomly_selected_dataframe.index)

                samples_regression_array,remaining_array = randomly_selected_dataframe[feature_columns].values, remaining_dataframe[feature_columns].values

                samples_array = samples_regression_array[~np.isnan(samples_regression_array).any(axis=1)]

                test_array = remaining_array[~np.isnan(remaining_array).any(axis=1)]
                # Random forest regression
                rf_regression = RandomForestRegressor(random_state=42)
                # 网格搜索
                grid_search = RandomizedSearchCV(rf_regression, param_distributions = param_grid, n_iter = 50, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

                X, y = samples_array[:,:-1], samples_array[:,-1]

                grid_search.fit(X, y)

                # fit
                R2, RMSE, regression_model = RF_regression(samples_array, grid_search.best_params_)

                # test
                x_test, y_test = test_array[:, :-1], test_array[:, -1]

                y_predict = regression_model.predict(x_test)

                test_RMSE = np.sqrt(mean_squared_error(y_test, y_predict))

                test_R2 = r2_score(y_test, y_predict)

                ### measure if the ground mode can be effectively determined
                test_dataframe = RF_dataframe.loc[remaining_dataframe.index.unique(), ['rxwaveform','search_start','search_end','GEDI_lowestmode_height_NAVD',
                                                                              'DEM_NEON_weighted', 'zcross','NLCD']]
                height_diff = []

                for shot_number in test_dataframe.index.values.tolist():

                    predictive_features = remaining_dataframe.loc[shot_number,feature_columns].values

                    footprint_x_features, footprint_y_height = predictive_features[:,:-1],predictive_features[:,-1]

                    y_predict = regression_model.predict(footprint_x_features)

                    mode_zcross = remaining_dataframe.loc[shot_number,'mode_zcross'].values

                    rx_waveform_str, search_start, search_end, GEDI_ele,NEON_reference_ele, GEDI_zcross, NLCD = test_dataframe.loc[shot_number, :]

                    ALS_ground_zcross = GEDI_zcross + (GEDI_ele - NEON_reference_ele) / 0.15

                    smooth_waveform, mean, noise = get_noise_waveform(rx_waveform_str, search_start, search_end)

                    mode_amplitudes = smooth_waveform[mode_zcross.astype(int)]

                    predict_height_zcross_amplitude = np.array([y_predict, mode_zcross, mode_amplitudes]).T

                    predict_height, predict_zcross = select_ground(predict_height_zcross_amplitude, mean, noise)

                    height_difference = abs((ALS_ground_zcross - predict_zcross) * 0.15)

                    height_diff.append(height_difference)

                MAE_accuracy,Std_accuracy = np.nanmean(np.array(height_diff)), np.nanstd(np.array(height_diff))

                output_str = f'site {label}, proportion: {proportion:.2f}: {R2:.2f},{RMSE:.2f},{test_R2:.2f},{test_RMSE:.2f},{MAE_accuracy:.2f},{Std_accuracy:.2f}'

                optimal_txt.write(output_str + '\n')

                optimal_txt.flush()

def get_noise_waveform(rx_waveform_str,search_start,search_end):

    rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

    smooth_waveform = GEDI_process.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

    last_10bins = np.concatenate([smooth_waveform[0:200], smooth_waveform[-200:]])

    mean, noise = np.mean(last_10bins), np.std(last_10bins)

    return smooth_waveform, mean, noise

def select_ground(predictive_array, mean, noise, NLCD):

    if NLCD == 'Broadleaf forest':
        k = 1.5
    elif NLCD == 'Mixed forest':
        k = 2.5
    elif NLCD == 'Needleleaf forest':
        k = 3.5
    else:
        k = 3.5

    filtered_zcross = predictive_array[predictive_array[:,2] > k * noise + mean]

    if filtered_zcross.ndim == 1:
        return filtered_zcross[0:2]
    else:
        ground_row = filtered_zcross[np.abs(filtered_zcross[:, 0]).argmin()]
        return ground_row[0:2]

def amplitude_threshold_test():
    # determine the amplitude filter threshold based on the maximum probability of selecting true zcross as well as the minimum probability of excluding noise mode
    mode_feature_dataframe = pd.read_excel(file_path.RF_derivative_features_excel, dtype={'shot_number': str}, index_col=0)

    RF_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    thresholds = np.arange(1,10.5,0.5)

    stats_array = np.empty((0,len(thresholds)))
    for forest_type in ['Broadleaf forest','Mixed forest','Needleleaf forest','other']:

        if forest_type == 'other':
            forest_dataframe = RF_dataframe[(RF_dataframe['NLCD'] != 'Broadleaf forest') & (RF_dataframe['NLCD'] != 'Mixed forest') & (RF_dataframe['NLCD'] != 'Needleleaf forest')]
        else:
            forest_dataframe = RF_dataframe[RF_dataframe['NLCD'] == forest_type]
        i = 0

        indicator_array = np.empty((len(forest_dataframe),len(thresholds)))

        for shot_number in forest_dataframe.index.values.tolist():

            rx_waveform_str, search_start, search_end = forest_dataframe.loc[shot_number, ['rxwaveform', 'search_start', 'search_end']].values

            smooth_waveform, mean, noise = get_noise_waveform(rx_waveform_str, search_start, search_end)

            mode_zcross_value = mode_feature_dataframe.loc[shot_number, 'mode_zcross'].values

            amplitude = smooth_waveform[mode_zcross_value.astype(int)]

            print(f'\r{forest_type}_{i}', end='')

            DEM_NEON_weighted, GEDI_ele, GEDI_zcross = forest_dataframe.loc[shot_number,['DEM_NEON_weighted', 'GEDI_lowestmode_height_NAVD', 'zcross']].values

            ALS_ground_zcross = GEDI_zcross + (GEDI_ele - DEM_NEON_weighted) / 0.15

            true_zcross = mode_zcross_value[np.argmin(abs(mode_zcross_value - ALS_ground_zcross))]

            for j, threshold in zip(np.arange(len(thresholds)),thresholds):

                selected_zcross = mode_zcross_value[amplitude > threshold * noise + mean]

                noise_modes = selected_zcross[selected_zcross > true_zcross]

                if true_zcross in selected_zcross:
                    _indicator = 1 if len(noise_modes) == 0 else 1 / len(noise_modes)
                else:
                    _indicator = -1
                indicator_array[i,j] = _indicator

            i = i + 1

        mean_by_col = np.mean(indicator_array, axis=0)

        std_by_col = np.std(indicator_array, axis=0)

        stats = np.vstack([mean_by_col, std_by_col])

        stats_array = np.vstack([stats_array,stats])

    np.savetxt(file_path.amplitude_thresholds_metrics_output_txt, stats_array, fmt="%.4f", header="Mean\nStd")


def feature_num_test(feature_excel, groupby = 'RF'):

    param_grid = {
        'n_estimators': randint(50, 500),
        'max_depth': [5, 10, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    important_columns = ['Back distance','Cumulative integral','Mode amplitude','ma4','ma3','ma6','ma5','Front distance','Mode location',
    'ma2','c85','loc4','Mode amplitude percentile','loc3','c5','loc2','loc1','c25','loc5','c65','ma1',
                          'p45','p25','c45','loc6','p5','p85','p65','stddev','Mode width']

    mode_feature_dataframe = pd.read_excel(feature_excel ,dtype = {'shot_number' :str} ,index_col = 0)

    RF_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    group_classification = mode_feature_dataframe.groupby(groupby)

    with open(file_path.RF_regression_feature_num_output_txt, 'a') as optimal_txt:

        for group in group_classification:

            label, dataframe = group[0], group[1]

            # test how mode identification accuracy changes with number of features used in regression
            for feature_num in np.arange(1,2,1):

                print(feature_num)

                feature_columns = important_columns[0:feature_num]

                feature_columns.append('mode_height')

                randomly_selected_dataframe = dataframe.sample(frac = 0.2, random_state=42)

                remaining_dataframe = dataframe.drop(randomly_selected_dataframe.index)

                samples_regression_array,remaining_array = randomly_selected_dataframe[feature_columns].values, remaining_dataframe[feature_columns].values

                samples_array = samples_regression_array[~np.isnan(samples_regression_array).any(axis=1)]

                test_array = remaining_array[~np.isnan(remaining_array).any(axis=1)]
                # Random forest regression
                rf_regression = RandomForestRegressor(random_state=42)
                # 网格搜索
                grid_search = RandomizedSearchCV(rf_regression, param_distributions = param_grid, n_iter = 50, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

                X, y = samples_array[:,:-1], samples_array[:,-1]

                grid_search.fit(X, y)

                # fit
                R2, RMSE, regression_model = RF_regression(samples_array, grid_search.best_params_)

                # test
                x_test, y_test = test_array[:, :-1], test_array[:, -1]

                y_predict = regression_model.predict(x_test)

                test_RMSE = np.sqrt(mean_squared_error(y_test, y_predict))

                test_R2 = r2_score(y_test, y_predict)

                ### measure if the ground mode can be effectively determined
                test_dataframe = RF_dataframe.loc[remaining_dataframe.index.unique(), ['rxwaveform','search_start','search_end','GEDI_lowestmode_height_NAVD',
                                                                              'DEM_NEON_weighted', 'zcross','NLCD']]
                ground_height_predict = []

                for shot_number in test_dataframe.index.values.tolist():

                    predictive_features = remaining_dataframe.loc[shot_number,feature_columns].values

                    footprint_x_features, footprint_y_height = predictive_features[:,:-1],predictive_features[:,-1]

                    y_predict = regression_model.predict(footprint_x_features)

                    mode_zcross = remaining_dataframe.loc[shot_number,'Mode location'].values

                    rx_waveform_str, search_start, search_end, GEDI_ele, NEON_reference_ele, GEDI_zcross, NLCD = test_dataframe.loc[shot_number, :]

                    ALS_ground_zcross = GEDI_zcross + (GEDI_ele - NEON_reference_ele) / 0.15

                    smooth_waveform, mean, noise = get_noise_waveform(rx_waveform_str, search_start, search_end)

                    mode_amplitudes = smooth_waveform[mode_zcross.astype(int)]

                    predict_height_zcross_amplitude = np.array([y_predict, mode_zcross, mode_amplitudes]).T

                    predict_height, predict_zcross = select_ground(predict_height_zcross_amplitude, mean, noise, NLCD)

                    # height_difference = abs((ALS_ground_zcross - predict_zcross) * 0.15)

                    ground_height_predict.append([ALS_ground_zcross*0.15, predict_zcross*0.15])

                #MAE_accuracy,Std_accuracy = np.nanmean(np.array(height_diff)), np.nanstd(np.array(height_diff))
                statistics = metric_cal(np.array(ground_height_predict))

                output_str = (f'site {label}, feature_num: {feature_num}: {R2:.2f},{RMSE:.2f},{test_R2:.2f},{test_RMSE:.2f},'
                              f'{statistics['R2']:.2f},{statistics['RMSE']:.2f},{statistics['MAE']:.2f},{statistics['BIAS']:.2f},{statistics['Bias (%)']:.2f}')

                optimal_txt.write(output_str + '\n')

                optimal_txt.flush()


def metric_cal(xy_array):
    obs = xy_array[:, 0]
    pred = xy_array[:, 1]

    r2 = r2_score(obs, pred)
    rmse = mean_squared_error(obs, pred, squared=False)
    mae = mean_absolute_error(obs, pred)
    bias = np.mean(pred - obs)
    bias_percent = 100 * bias / np.mean(obs)

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'BIAS': bias,
        'Bias (%)': bias_percent
    }

if __name__ == '__main__':

    gaussian_features_columns = ['normalized_amplitude', 'mode_duration', 'cumulative_ratio', 'mode_percentile', 'distance_to_max',
               'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6', 'mode_zcross',
               'mode_height']

    # derivative_features_columns = ['mode_zcross', 'distance_to_start', 'distance_to_end', 'distance_ratio',
    #                           'mode_amplitude', 'mode_duration', 'amplitude_percentile', 'cumulative_ratio',
    #                           'is_second_deri','ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6',
    #                           'per_5', 'per_25', 'per_45', 'per_65', 'per_85','cumu_5_index', 'cumu_25_index', 'cumu_45_index', 'cumu_65_index', 'cumu_85_index','stddev','mode_height']

    new_columns = ['Mode location', 'Front distance', 'Back distance', 'Mode amplitude', 'Mode width',
                      'Mode amplitude percentile', 'Cumulative integral',
                      'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6',
                      'p5','p25', 'p45', 'p65', 'p85', 'c5', 'c25', 'c45', 'c65', 'c85', 'stddev', 'mode_height']


    # feature_num_test(file_path.RF_derivative_features_excel,groupby = 'RF')

    # 应用于 DataFrame

    ### regression by gaussian decomposition results
    ## file_path.manually_derivative_samples

    # RF_regression_proportion_test(file_path.RF_derivative_features_excel, derivative_features_columns, groupby = 'RF')
    ### regression by manually selected results

    #amplitude_threshold_test()

    #RF_regression_proportion(file_path.RF_derivative_features_excel, derivative_features_columns)