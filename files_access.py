import os

import pandas as pd


class file_path:

    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file_path)

    # excel, all GEDI records used in this file; for RF prediction and getting reuslts
    RF_excel = os.path.join(project_root, '7_RF_comparison.xlsx')

    # excel, the same as below
    test_excel = os.path.join(project_root, 'test_data.xlsx')

    # excel,manually selected samples; to select ground return location,
    # the manually selected location is saved as column of "manually_zcross"
    manual_sample_excel = os.path.join(project_root, 'test_data.xlsx')

    # gaussian decomposition results saving path
    Gau_result_path = os.path.join(project_root, 'result_output', 'gaussian_decomposition')

    #txt; gaussian decomposition result;
    Gau_Decomposition_txt = os.path.join(Gau_result_path, 'result_output', 'gaussian_decomposition', 'Gau_decom.txt')


    ### RF work directory

    # RF training by the samples generated based on ALS elevation
    elevation_train = os.path.join(project_root, 'result_output', 'trainning_ALSsamples')

    # RF training by the samples generated based on manual selection of ground location
    manually_train = os.path.join(project_root, 'result_output', 'training_visualsamples')


    ### regression data
    RF_waveform_PCA_txt = os.path.join(project_root, 'Regress_ground', 'RF_waveform_PCA.txt')


def data_update(update_columns):
    test_excel = file_path().test_excel
    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    rf_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\7_RF_comparison.xlsx'
    storage_dataframe = pd.read_excel(rf_excel, dtype={'shot_number': str}, index_col=0)

    # Merge storage_dataframe with dataframe on index and update specified columns
    dataframe.update(storage_dataframe[update_columns])

    # Save updated dataframe back to Excel
    dataframe.to_excel(test_excel)


def add_newColumns(update_columns):
    test_excel = file_path().test_excel

    dataframe = pd.read_excel(test_excel, dtype={'shot_number': str}, index_col=0)

    rf_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\7_RF_comparison.xlsx'
    storage_dataframe = pd.read_excel(rf_excel, dtype={'shot_number': str}, index_col=0)

    dataframe = dataframe.merge(storage_dataframe[update_columns], left_index=True, right_index=True, how='left')

    dataframe.to_excel(test_excel)

if __name__ == '__main__':
    print('update data if needed')
    #add_newColumns(['DEM_NEON_average','DEM_NEON_weighted'])
