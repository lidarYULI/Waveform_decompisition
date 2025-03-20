import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
import derivative_samples
from pathlib import Path
from gaussian_samples import file_path

def train_by_elevation_samples():
    # using elevation derived samples
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[1]

    result_folder = os.path.join(project_root, 'result_output')

    sample_excel = os.path.join(result_folder, 'elevation_train', 'random_selection_samples.xlsx')

    save_folder = os.path.join(result_folder, 'elevation_train')

    train_by_different_site(sample_excel,save_folder)

def train_by_manual_samples():
    # using elevation derived samples
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[1]

    result_folder = os.path.join(project_root, 'result_output')

    sample_excel = os.path.join(result_folder, 'visual_train', 'manual_selection_samples.xlsx')

    save_folder = os.path.join(result_folder, 'visual_train')

    train_by_different_site(sample_excel, save_folder)

def train_by_different_site(sample_excel,save_folder):

    RF_Models_params = {
        # the default max_features is sqrt
        'HARV':{'max_depth': None, 'max_features': 'sqrt','max_leaf_nodes': None,'min_samples_leaf': 4,'min_samples_split': 2,'n_estimators': 200},

        'RMNP':{'max_depth': None, 'max_features': None, 'max_leaf_nodes': None,'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100},

        'TALL': {'max_depth': None, 'max_features': None, 'max_leaf_nodes': 20,'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 100},

        'TREE':{'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None,'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100},

        'UNDE':{'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 200},

        'WREF':{'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None,'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    }

    sample_dataframe = pd.read_excel(sample_excel,dtype={'shot_number': str})

    dataframe_group = sample_dataframe.groupby('site')

    for group in dataframe_group:
        site = group[0]

        dataframe = group[1]

        mode_path = os.path.join(save_folder, f'{site}.pkl')

        txt_path =  os.path.join(save_folder,f'{site}_model_accuracy.txt_para')

        RF_Models_param = RF_Models_params[site]

        print('\n')
        print(site+' #########################################################')

        X_train, X_test, y_train, y_test, columns = sample_split(dataframe)

        rf_model, accuracy = random_forest_train(X_train, X_test, y_train, y_test, RF_Models_param)

        model_info(rf_model,columns, mode_path, txt_path, accuracy)

def model_info(rf_model,feature_columns,model_save_path, output_textpath, accuracy):
    # 5. the importance of feature
    params = rf_model.get_params()
    importances = rf_model.feature_importances_

    output = []
    output.append("Random Forest Model Parameters:\n")
    for key, value in params.items():
        output.append(f"{key}: {value}\n")

    output.append("feature importance:\n")
    print("importance of feature:")
    i = 0
    for importance,column in zip(importances,feature_columns):
        print(f"{i + 1}_{column}: {importance:.4f}")
        i = i + 1
        output.append(f"{i + 1}_{column}: {importance:.4f}\n")

    output.append(f"\nModel Accuracy: {accuracy:.4f}\n")

    # 5. save the result to txt
    with open(output_textpath, "w") as f:
        f.writelines(output)

    joblib.dump(rf_model, model_save_path)

    print(f"save model to: {model_save_path}")

# train and test random forest and save the test result
def random_forest_train(X_train, X_test, y_train, y_test, RF_Params):

    # 1. creat a RF classifier based on the given parameters
    rf_model = RandomForestClassifier(
        n_estimators = RF_Params['n_estimators'],  # the number of decision tree
        max_depth = RF_Params['max_depth'],  # maximum depth of a tree
        max_leaf_nodes = RF_Params['max_leaf_nodes'],
        min_samples_leaf = RF_Params['min_samples_leaf'],
        min_samples_split = RF_Params['min_samples_split'],
        random_state=42, # random
        max_features = 'sqrt'
    )

    # 2. fit RF
    rf_model.fit(X_train, y_train)

    # 3. predict
    y_pred = rf_model.predict(X_test)

    # 4. evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    return rf_model, accuracy

def sample_split(sample_dataframe):

    cloumns = ['distance_to_start','distance_to_end','distance_ratio','mode_amplitude','mode_duration',
                                          'ma1', 'ma2', 'ma3', 'ma4', 'ma5','percentile_amp','cumulative_ratio',
                                          'stddev','per_25', 'per_50', 'per_75', 'cumu_25_index', 'cumu_50_index', 'cumu_75_index','is_ground']
    selected_featrues = sample_dataframe[cloumns].values

    non_return_sample = selected_featrues[selected_featrues[:, -1] == 0, :]

    return_sample = selected_featrues[selected_featrues[:, -1] == 1, :]

    if len(non_return_sample)> 3 * len(return_sample):
        selected_non_return_sample = non_return_sample[np.random.choice(len(non_return_sample), 3 * len(return_sample), replace=False), :]
    else:
        selected_non_return_sample = non_return_sample[np.random.choice(len(non_return_sample), 2 * len(return_sample), replace=False), :]

    select_data = np.concatenate([selected_non_return_sample, return_sample])

    print('positive', np.shape(return_sample))
    print('selected negative', np.shape(selected_non_return_sample))
    print('total negative', np.shape(non_return_sample))

    # first two columns are latitude and longitude
    X = select_data[:, :-1]
    y = select_data[:, -1].astype(int)
    # 2. 数据集划分：训练集和测试集
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, cloumns

######## prediction
def RF_predict_bysite(RF_excel,model_folder):

    RF_dataframe = pd.read_excel(RF_excel, dtype={'shot_number': str}, index_col=0)

    HARV_model = joblib.load(os.path.join(model_folder,'HARV.pkl'))
    RMNP_model = joblib.load(os.path.join(model_folder,'RMNP.pkl'))
    TALL_model = joblib.load(os.path.join(model_folder,'TALL.pkl'))
    UNDE_model = joblib.load(os.path.join(model_folder,'UNDE.pkl'))
    TREE_model = joblib.load(os.path.join(model_folder,'TREE.pkl'))
    WREF_model = joblib.load(os.path.join(model_folder,'WREF.pkl'))
    print("模型已加载！")

    for index, i in zip(RF_dataframe.index.values.tolist(), range(len(RF_dataframe))):
        print(i, index)
        site = RF_dataframe.loc[index, 'site']
        RF_model = None
        if site == 'HARV':
            RF_model = HARV_model
        elif site == 'RMNP':
            RF_model = RMNP_model
        elif site == 'UNDE':
            RF_model = UNDE_model
        elif site == 'TREE':
            RF_model = TREE_model
        elif site == 'WREF':
            RF_model = WREF_model
        elif site == 'TALL':
            RF_model = TALL_model
        #is_RF = Dataframe.loc[index, 'is_RF']

        predict_based_on_dataframe(RF_dataframe,index,RF_model)

    RF_dataframe.to_excel(RF_excel)

# this method is good and finally used in my dissertation
def predict_based_on_dataframe(Dataframe,index,RF_model):

    sample_fetures_dataframes = get_samples_RF_dataframe(Dataframe,index)
    ### ,'is_second_deri' (sixth): this feature is used in previous RF models (first version of RF model,D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\WRD\RF_Models_manual)
    sample_array = sample_fetures_dataframes.loc[:,
                   ['distance_to_start','distance_to_end','distance_ratio','mode_amplitude','mode_duration',
                    'ma1', 'ma2', 'ma3', 'ma4', 'ma5','percentile_amp','cumulative_ratio',
                    'stddev','per_25', 'per_50', 'per_75', 'cumu_25_index', 'cumu_50_index', 'cumu_75_index']].values

    y_pred = RF_model.predict(sample_array)

    sample_fetures_dataframes.loc[:, 'y_pred'] = y_pred

    predicted_ground_mode = sample_fetures_dataframes.loc[sample_fetures_dataframes.loc[:, 'y_pred'] == 1, :]

    if len(predicted_ground_mode) > 0:

        Dataframe.loc[index, 'detected_RF_mode_number'] = len(predicted_ground_mode) # number of modes
        Dataframe.loc[index, 'detected_RF_mode_loc_std'] = predicted_ground_mode['mode'].std() # std mode locations
        Dataframe.loc[index, 'detected_RF_mode_loc_mean'] = predicted_ground_mode['mode'].mean()

        if len(predicted_ground_mode) == 1:
            # if direct detection
            Dataframe.loc[index, 'randomforest_zcross'] = predicted_ground_mode['mode'].values[0]
            Dataframe.loc[index, 'is_RF'] = 0
        else:
            #detected_mode = mode_judgement(predicted_ground_mode)
            #detected_mode = np.percentile(predicted_ground_mode['mode'].values, 85)
            second_deri_mode = predicted_ground_mode.loc[predicted_ground_mode['is_second_deri']==1,:]
            if len(second_deri_mode) == 1:
                detected_mode = second_deri_mode['mode'].values[0]
                Dataframe.loc[index, 'is_RF'] = 1
                Dataframe.loc[index, 'randomforest_zcross'] = detected_mode
            else:
                #detected_mode = np.percentile(predicted_ground_mode['mode'].values, 50)  # for 75th percentile
                mode_value = predicted_ground_mode['mode'].values
                percentile_value = np.percentile(mode_value, 50)
                elements_above_percentile = mode_value[mode_value >= percentile_value]
                detected_mode = elements_above_percentile.min()
                Dataframe.loc[index, 'randomforest_zcross'] = detected_mode
                Dataframe.loc[index, 'is_RF'] = 2
    else:
        Dataframe.loc[index, 'is_RF'] = 3
        Dataframe.loc[index, 'randomforest_zcross'] = Dataframe.loc[index, 'zcross']

#get features of waveform for predicting
def get_samples_RF_dataframe(Dataframe,index):

    rx_waveform_str, search_start, toploc, search_end, mean, stddev = Dataframe.loc[index, ['rxwaveform', 'search_start', 'toploc', 'search_end', 'mean','stddev']].values

    rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

    waveform_info = [rx_waveform_value, search_start, search_end, mean, stddev]

    sample_fetures_dataframes = derivative_samples.generate_features(waveform_info)

    return sample_fetures_dataframes

if __name__ == '__main__':

    train_by_elevation_samples()

    # train_by_manual_samples()

    # RF_excel = file_path.RF_excel

    # model_folder = file_path.elevation_train

    # RF_predict_bysite(RF_excel, model_folder)
