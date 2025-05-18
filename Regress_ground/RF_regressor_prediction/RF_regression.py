import pandas as pd
from sympy.polys.matrices.normalforms import add_columns
from files_access import file_path
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import re
import os
import joblib
from sklearn.metrics import r2_score
import GEDI_waveform_processing as GEDI_process
from datetime import datetime

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

    R2 = rf_regression.score(x,y)

    return R2, RMSE, rf_regression

def RF_regression_optimization(feature_excel,feature_columns,resource = 'GEDI',groupby = 'site'):
    '''
    :param feature_excel:
    # feature_excel:
    # file_path.RF_modes_features_excel when using gaussian decomposition mode derived features
    # file_path.RF_derivative_features_excel when using derivative derived features
    :param feature_columns: features used in regression
    :param groupby: how to classify dataset when training; kmeans_labels/site
    :return:
    '''
    mode_feature_dataframe = pd.read_excel(feature_excel ,dtype = {'shot_number' :str} ,index_col = 0)

    group_classification = mode_feature_dataframe.groupby(groupby)

    # 定义参数网格
    param_grid = {
        'n_estimators': randint(50, 500),
        'max_depth': [5, 10, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    with open(file_path.RF_regression_parameters_output_txt, 'a') as optimal_txt:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        optimal_txt.write(current_time + '###################################' + '\n')
        for group in group_classification:

            label, dataframe = group[0], group[1]

            if resource in ['GEDI','manual']:
                samples_regression_array = dataframe[feature_columns].values
            else:
                randomly_selected_dataframe = dataframe.sample(n=int(0.1 * len(dataframe)))
                samples_regression_array = randomly_selected_dataframe[feature_columns].values

            samples_regression = samples_regression_array[~np.isnan(samples_regression_array).any(axis=1)]
            #stratified_dataframe = stratify_sample(mode_selected_features)
            # Random forest regression
            rf_regression = RandomForestRegressor(random_state=42)
            # 网格搜索
            grid_search = RandomizedSearchCV(rf_regression, param_distributions = param_grid, n_iter = 50, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

            X, y = samples_regression[:,:-1], samples_regression[:,-1]

            grid_search.fit(X, y)

            R2, RMSE, regression_model = RF_regression(samples_regression, grid_search.best_params_)

            saving_path = os.path.join(file_path.RF_regression_model_folder, f'{resource}_{label}_RF_regression.joblib')

            joblib.dump(regression_model, saving_path)

            print("waveform_class：", f'{label}')

            print("parameters：", grid_search.best_params_)

            print("best score：", grid_search.best_score_)

            output_str = f'waveform_class: {resource}_{label} number: {len(samples_regression)} \n R2: {R2} RMSE: {RMSE} \n {grid_search.best_params_} \n {grid_search.best_score_}'

            optimal_txt.write(output_str + '\n')

## RF run regression based on optimized parameters
def RF_regression_test(feature_excel,feature_columns, resource = '', groupby = 'site'):

    mode_feature_dataframe = pd.read_excel(feature_excel ,dtype = {'shot_number' :str} ,index_col = 0)

    mode_feature_dataframe.dropna(inplace=True)

    group_classification = mode_feature_dataframe.groupby(groupby)

    for group in group_classification:

        label, dataframe = group[0], group[1]

        samples_regression = dataframe[feature_columns].values

        model_path = os.path.join(file_path.RF_regression_model_folder, f'{resource}_{label}_RF_regression.joblib')

        loaded_model = joblib.load(model_path)

        # 使用加载的模型进行预测
        y_pred = loaded_model.predict(samples_regression[:, :-1])

        RMSE = np.sqrt(mean_squared_error(samples_regression[:,-1], y_pred))

        R2 = r2_score(samples_regression[:,-1], y_pred)

        print(label,RMSE,R2)

def RF_regression_predict(feature_excel,feature_columns,resource = 'GEDI' ,groupby = 'RF',model_label = 'site'):

    mode_feature_dataframe = pd.read_excel(feature_excel, dtype={'shot_number': str})

    group_classification = mode_feature_dataframe.groupby(groupby)

    for group in group_classification:

        label, dataframe = group[0], group[1]

        print(label)

        samples_regression = dataframe[feature_columns].values

        model_path = os.path.join(file_path.RF_regression_model_folder, f'{resource}_{model_label}_RF_regression.joblib')

        loaded_model = joblib.load(model_path)

        # predict based on loaded model
        y_pred = loaded_model.predict(samples_regression[:,:-1])

        mode_feature_dataframe.loc[dataframe.index.values,f'height_predict_{model_label}'] = y_pred

    mode_feature_dataframe.to_excel(feature_excel,index = False)

def RF_regression_selection(obj_excel,feature_excel,model_label = 'site'):

    mode_feature_dataframe = pd.read_excel(feature_excel, dtype={'shot_number': str},index_col = 0)

    RF_dataframe = pd.read_excel(obj_excel, dtype={'shot_number': str}, index_col = 0)

    i = 0

    for shot_number in RF_dataframe.index.values.tolist():

        rx_waveform_str, search_start, search_end, land_cover = RF_dataframe.loc[shot_number, ['rxwaveform','search_start', 'search_end','NLCD']].values

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_process.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

        last_10bins = np.concatenate([rx_waveform_value[0:200], rx_waveform_value[-200:]])

        mean, noise = np.mean(last_10bins), np.std(last_10bins)

        i = i + 1

        print(f'\r{i}', end='')

        prediction_result_array = mode_feature_dataframe.loc[shot_number,[f'Mode location',f'mode_height',f'height_predict_{model_label}']].values  # height_predict_{groupby}

        amplitude = smooth_waveform[prediction_result_array[:,0].astype(int)]

        if land_cover == 'Broadleaf forest':
            k = 1.5  #1.5 for ALS training; 4.5
        elif land_cover == 'Mixed forest':
            k = 2.5  # 2.5 for ALS training 6.5
        elif land_cover == 'Needleleaf forest':
            k = 3.5 # 3.5 for ALS training 7.5
        else:
            k = 3.5 # 3.5 for ALS training 7.5

        filter_result_array = prediction_result_array[amplitude > k * noise + mean]

        if  filter_result_array.shape[0] == 0:
            #RF_dataframe.loc[shot_number, [f'mode_zcross_{groupby}', f'mode_height_{groupby}']] = prediction_result_array[0:2]
            ground_row = filter_result_array[np.abs(filter_result_array[:, 2]).argmin()]
            RF_dataframe.loc[shot_number, [f'mode_zcross_{model_label}', f'mode_height_{model_label}']] = ground_row[0:2]
        else:
            ground_row = filter_result_array[np.abs(filter_result_array[:, 2]).argmin()]
            RF_dataframe.loc[shot_number,[f'mode_zcross_{model_label}', f'mode_height_{model_label}']] = ground_row[0:2]


    RF_dataframe.to_excel(obj_excel)

def RF_regression_read(txt_path):

    with open(txt_path, 'r') as file:
        txt_content = file.read()

    waveform_classes = re.findall(r'waveform_class:\s*(\d+)', txt_content)

    param_patterns = re.findall(r"\{(.*?)\}", txt_content)

    waveform_class_dict = {}

    for i, class_num in enumerate(waveform_classes):
        params = eval('{' + param_patterns[i] + '}')  # 使用 eval 将字符串转换为字典
        waveform_class_dict[int(class_num)] = params

    return waveform_class_dict

### select consistent number of dataset within each interval of mode height
def stratify_sample(dataframe,column = 'mode_height'):

    # divide by pgap
    bins = np.arange(0, 1, 0.1)

    labels = range(len(bins) - 1)  # 生成分层的标签

    # 使用 cut 函数将数据分层
    dataframe['bin'] = pd.cut(dataframe[column], bins = bins, labels = labels, include_lowest = True)

    # # 计算每个分层的样本数量，并找出最小的数量
    min_sample_size = dataframe['bin'].value_counts().min()

    # 按照最小的分层数量进行抽样
    sampled_df = dataframe.groupby('bin', group_keys=False,observed=False).apply(lambda x: x.sample(min_sample_size), include_groups=False)

    return sampled_df
################### regression based on features derived from derivative

def get_noise_waveform(rx_waveform_str,search_start,search_end):

    rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

    smooth_waveform = GEDI_process.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

    last_10bins = np.concatenate([smooth_waveform[0:200], smooth_waveform[-200:]])

    mean, noise = np.mean(last_10bins), np.std(last_10bins)

    return smooth_waveform, mean, noise

def varied_select_threshold():

    mode_feature_dataframe = pd.read_excel(file_path.RF_derivative_features_excel, dtype={'shot_number': str}, index_col=0)

    RF_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col=0)

    rf_dataframe_nlcd_group = RF_dataframe.groupby('NLCD')

    result_dataframe = pd.DataFrame()

    for group in rf_dataframe_nlcd_group:

        forest_type = group[0]

        forest_dataframe = group[1]

        i = 0

        for shot_number in forest_dataframe.index.values.tolist():

            rx_waveform_str, search_start, search_end = forest_dataframe.loc[shot_number, ['rxwaveform', 'search_start', 'search_end']].values

            prediction_result_array = mode_feature_dataframe.loc[shot_number, ['mode_zcross', 'mode_height', 'height_predict']].values

            smooth_waveform, mean, noise = get_noise_waveform(rx_waveform_str, search_start, search_end)

            amplitude = smooth_waveform[prediction_result_array[:, 0].astype(int)]

            i = i + 1

            print(f'\r{forest_type}_{i}', end='')

            result_dataframe.loc[shot_number,['site','NLCD','DEM_NEON_weighted', 'GEDI_lowestmode_height_NAVD','zcross']] = forest_dataframe.loc[shot_number,['site','NLCD','DEM_NEON_weighted', 'GEDI_lowestmode_height_NAVD','zcross']]

            for threshold in np.arange(1,7,0.5):

                selected_array = prediction_result_array[amplitude > threshold * noise + mean]

                column_zcross,column_height = f'mode_zcross_{threshold}',f'mode_height_{threshold}'

                if selected_array.ndim == 1:

                    result_dataframe.loc[shot_number, [column_zcross, column_height]] = selected_array[0:2]

                else:
                    ground_row = selected_array[np.abs(selected_array[:, 2]).argmin()]

                    result_dataframe.loc[shot_number, [column_zcross, column_height]] = ground_row[0:2]

    result_dataframe.index.name = 'shot_number'

    result_dataframe.to_excel(file_path.varied_threshold_test_save_excel)

def test_transferability(feature_excel, feature_columns):

    selected_sites = ['TALL', 'RMNP', 'UNDE', 'TREE', 'HARV', 'WREF']

    mode_feature_dataframe = pd.read_excel(feature_excel, dtype={'shot_number': str})

    group_classification = mode_feature_dataframe.groupby('site')

    add_columns = ['shot_number','mode_zcross', 'DEM_NEON_weighted','mode_height' ,'GEDI_lowestmode_height_NAVD', 'zcross', 'site']

    result_dataframe = pd.DataFrame()

    result_dataframe[add_columns] = mode_feature_dataframe[add_columns]

    for group in group_classification:

        label, dataframe = group[0], group[1]

        print(label)

        model_path = os.path.join(file_path.RF_regression_model_folder, f'{label}_RF_regression.joblib')

        loaded_model = joblib.load(model_path)
        # use a model built on dataset from one site to predict other sites' dataset
        for site in selected_sites:

            group_dataframe = group_classification.get_group(site)

            samples_regression = group_dataframe[feature_columns].values

            y_pred = loaded_model.predict(samples_regression[:, :-1])

            result_dataframe.loc[group_dataframe.index, f'{label}_height_predict'] = y_pred

    result_dataframe.to_excel(file_path.transferability_save_excel,index=False)

def transferability_selection():

    transfer_feature_dataframe = pd.read_excel(file_path.transferability_save_excel, dtype={'shot_number': str},index_col = 0)

    RF_dataframe = pd.read_excel(file_path.RF_excel, dtype={'shot_number': str}, index_col = 0)

    i = 0

    for shot_number in RF_dataframe.index.values.tolist():

        rx_waveform_str, search_start, search_end, land_cover = RF_dataframe.loc[shot_number, ['rxwaveform','search_start', 'search_end','NLCD']].values

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_process.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

        last_10bins = np.concatenate([rx_waveform_value[0:200], rx_waveform_value[-200:]])

        mean, noise = np.mean(last_10bins), np.std(last_10bins)

        i = i + 1
        print(f'\r{i}', end='')

        for site in ['HARV', 'RMNP', 'TALL', 'TREE', 'UNDE', 'WREF']:

            prediction_result_array = transfer_feature_dataframe.loc[shot_number,[f'mode_zcross',f'mode_height',f'{site}_height_predict']].values

            amplitude = smooth_waveform[prediction_result_array[:,0].astype(int)]

            if land_cover == 'Broadleaf forest':
                k = 1.5
            elif land_cover == 'Mixed forest':
                k = 2.5
            elif land_cover == 'Needleleaf forest':
                k = 3
            else:
                k = 3
            prediction_result_array = prediction_result_array[amplitude > k*noise + mean]

            if prediction_result_array.ndim == 1:
                RF_dataframe.loc[shot_number, [f'mode_zcross_{site}_model', f'mode_height_{site}_model']] = prediction_result_array[0:2]
            else:
                ground_row = prediction_result_array[np.abs(prediction_result_array[:, 2]).argmin()]
                RF_dataframe.loc[shot_number,[f'mode_zcross_{site}_mode', f'mode_height_{site}_model']] = ground_row[0:2]

    RF_dataframe.to_excel(file_path.RF_excel)

if __name__ == '__main__':

    gaussian_features_columns = ['normalized_amplitude', 'mode_duration', 'cumulative_ratio', 'mode_percentile', 'distance_to_max',
               'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6', 'mode_zcross',
               'mode_height']

    # derivative_features_columns = ['mode_zcross', 'distance_to_start', 'distance_to_end', 'distance_ratio',
    #                           'mode_amplitude', 'mode_duration', 'amplitude_percentile', 'cumulative_ratio',
    #                           'is_second_deri','ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6',
    #                           'per_5', 'per_25', 'per_45', 'per_65', 'per_85','cumu_5_index', 'cumu_25_index', 'cumu_45_index', 'cumu_65_index', 'cumu_85_index','stddev'
    #                           'Latitude','Longitude','beam','date','mode_height']

    # old_columns = ['mode_zcross', 'distance_to_start', 'distance_to_end','mode_amplitude', 'mode_duration', 'amplitude_percentile', 'cumulative_ratio',
    #                                'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6','loc1', 'loc2', 'loc3','loc4', 'loc5', 'loc6',
    #                                'per_5', 'per_25', 'per_45', 'per_65', 'per_85',
    #                                'cumu_5_index', 'cumu_25_index', 'cumu_45_index', 'cumu_65_index', 'cumu_85_index',
    #                                'stddev', 'mode_height']

    new_columns = ['Mode location', 'Front distance', 'Back distance', 'Mode amplitude', 'Mode width',
                   'Mode amplitude percentile', 'Cumulative integral',
                   'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6',
                   'p5', 'p25', 'p45', 'p65', 'p85', 'c5', 'c25', 'c45', 'c65', 'c85', 'stddev', 'mode_height']

    important_columns = ['Back distance','Cumulative integral','Mode amplitude','ma4','ma3','ma6',
                         'ma5','Front distance','Mode location','ma2','mode_height']

    #rename_dict = dict(zip(old_columns, new_columns))

    # 应用于 DataFrame
    # mode_feature_dataframe = pd.read_excel(file_path.all_validation_GEDI_feature_excel, dtype={'shot_number': str}, index_col=0)
    # mode_feature_dataframe.rename(columns = rename_dict, inplace=True)
    # mode_feature_dataframe.to_excel(file_path.all_validation_GEDI_feature_excel)

    ### regression by gaussian decomposition results
    ## file_path.manually_derivative_samples

    RF_regression_optimization(file_path.RF_derivative_features_excel, new_columns, resource ='GEDI', groupby ='site')
    #
    RF_regression_test(file_path.RF_derivative_features_excel,new_columns, resource = 'GEDI',groupby = 'site')
    #
    RF_regression_predict(file_path.RF_derivative_features_excel,new_columns, resource = 'GEDI', groupby = 'RF', model_label = 'site')
    #
    RF_regression_selection(file_path.RF_excel, file_path.RF_derivative_features_excel, model_label = 'site')

    #varied_select_threshold()

    ### regression by manually selected results

    #test_transferability(file_path.RF_derivative_features_excel, derivative_features_columns)
    #transferability_selection()