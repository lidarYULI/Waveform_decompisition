
import os
import pandas as pd
import GEDI_waveform_processing
import numpy as np
from files_access import file_path

def data_update(obj_excel,update_columns):

    dataframe = pd.read_excel(obj_excel, dtype={'shot_number': str}, index_col=0)

    rf_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\7_RF_comparison.xlsx'
    storage_dataframe = pd.read_excel(rf_excel, dtype={'shot_number': str}, index_col=0)

    # Merge storage_dataframe with dataframe on index and update specified columns
    dataframe.update(storage_dataframe[update_columns])

    # Save updated dataframe back to Excel
    dataframe.to_excel(obj_excel)

def add_newColumns(storage_excel,obj_excel,added_columns):

    existed_dataframe = pd.read_excel(storage_excel, dtype={'shot_number': str}, index_col=0)

    # basic_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\2_basic_data.xlsx'
    #
    # basic_dataframe = pd.read_excel(basic_excel, dtype={'shot_number': str}, index_col=0)

    waveform_df = pd.read_excel(obj_excel, dtype={'shot_number': str}, index_col=0)

    dataframe = existed_dataframe.merge(waveform_df[added_columns], left_index=True, right_index=True, how='left')

    dataframe.to_excel(obj_excel)

## extract samples based on a column
def stratify_sample(dataframe,column = 'pgap_theta'):

    # divide by pgap
    bins = np.arange(0, 1, 0.1)

    labels = range(len(bins) - 1)  # 生成分层的标签

    # use cut to stratify dataset
    dataframe['bin'] = pd.cut(dataframe[column], bins = bins, labels = labels, include_lowest = True)

    # # calculate the number of stratified samples
    min_sample_size = dataframe['bin'].value_counts().min()

    # # use the minimum stratified number
    sampled_df = dataframe.groupby('bin', group_keys=False,observed=False).apply(lambda x: x.sample(min_sample_size), include_groups=False)

    return sampled_df

### filter to get high accurate elevation products from all L2A products
def high_accurate_elevation_filter():

    elevation_df = pd.read_excel(file_path.all_GEDI_dataset_excel, dtype={'shot_number': str}, index_col = 0 )

    ### filter dataset based on Moudrý,2024
    groups_elevation = ['elev_lowestmode_a1','elev_lowestmode_a2','elev_lowestmode_a3','elev_lowestmode_a4','elev_lowestmode_a5','elev_lowestmode_a6']

    elevation_df['max_diff'] = elevation_df[groups_elevation].max(axis=1) - elevation_df[groups_elevation].min(axis=1)

    elevation_df['diff_to_Tan'] = abs(elevation_df['elev_lowestmode'] - elevation_df['DEM_TANX'])

    filter = (elevation_df['max_diff'] <= 2) & (elevation_df['diff_to_Tan'] <= 50) & (elevation_df['sensitivity'] > 0.95) # no filter on sensitivity

    filter_dataframe = elevation_df[filter]

    rows_to_delete = []

    # extra filter based on waveform mode amplitude
    for shot_number in filter_dataframe.index.values:

        rx_waveform_str, search_start, search_end, land_cover, zcross = filter_dataframe.loc[
            shot_number, ['rxwaveform', 'search_start', 'search_end', 'NLCD','zcross']].values

        rx_waveform_value = np.array(rx_waveform_str.split(',')).astype(np.float32)

        smooth_waveform = GEDI_waveform_processing.rx_waveform_denoise(rx_waveform_value, search_start, search_end, 4)

        last_10bins = np.concatenate([rx_waveform_value[0:200], rx_waveform_value[-200:]])

        mean, noise = np.mean(last_10bins), np.std(last_10bins)

        amplitude = smooth_waveform[int(zcross)]

        if amplitude < mean + 3.5 * noise:
            rows_to_delete.append(shot_number)

    # delete noise modes
    filter_dataframe = filter_dataframe.drop(rows_to_delete)

    ### stratify by pgap

    dataframe = pd.read_excel(r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\1_filter_dataset_near_months.xlsx', dtype={'shot_number': str},index_col = 0)

    filter_dataframe['is_power_beam'] = filter_dataframe['beam'].apply(GEDI_waveform_processing.is_powerbeams_byname)

    output_dataframe = filter_dataframe.merge(dataframe['pgap_theta'],left_index = True, right_index = True, how = 'left')

    stratified_dataframe = stratify_sample(output_dataframe, column='pgap_theta')

    stratified_dataframe.to_excel(file_path.stratified_filter_save_excel)


def self_elevation_filter():
    # self defined criteria
    elevation_df = pd.read_excel(file_path.ICCG_reference_excel, dtype={'shot_number': str}, index_col = 0 )

    groups_elevation = ['elev_lowestmode_a1','elev_lowestmode_a2','elev_lowestmode_a3','elev_lowestmode_a4','elev_lowestmode_a5','elev_lowestmode_a6']

    elevation_df['max_diff'] = elevation_df[groups_elevation].max(axis=1) - elevation_df[groups_elevation].min(axis=1)

    elevation_df['diff_to_Tan'] = abs(elevation_df['elev_lowestmode'] - elevation_df['DEM_TANX'])

    filter = (elevation_df['NLCD'] == 'Needleleaf forest') & (elevation_df['max_diff'] <= 2) & (elevation_df['diff_to_Tan'] <= 50) & (elevation_df['sensitivity'] > 0.95)

    filter_dataframe = elevation_df[filter]

    filter_dataframe.to_excel(file_path.ICCG_needleleaf_filter_excel)

if __name__ == '__main__':
    print('update data if needed')

    self_elevation_filter()
    # add_newColumns()
    #high_accurate_elevation_filter()
    #test_excel = file_path().test_excel
    #add_newColumns(['DEM_NEON_average','DEM_NEON_weighted'])
    #obj_excel = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\WRD\Manually_WRD.xlsx'

    #add_newColumns(obj_excel,['DEM_NEON_weighted','DEM_NEON_average','match_x','match_y'])



    #

    # saving_folder = r'D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\Elevation_conversion\L2A_output'
    #
    # navd_excel = os.path.join(saving_folder, "all_l2a_dataframe_other.xlsx")
    #
    # navd_dataframe = pd.read_excel(navd_excel, dtype={'shot_number': str}, index_col=0)
    #
    # l2a_dataframe = pd.read_excel(file_path.all_validation_GEDI_excel,dtype={'shot_number': str}, index_col=0)
    #
    # l2a_dataframe = l2a_dataframe.merge(navd_dataframe['elev_lowestmode_NAVD'],left_index=True, right_index=True, how='left')
    #
    # l2a_dataframe.to_excel(file_path.all_validation_GEDI_excel)
    #
    # filter_heigh_dataframe = pd.read_excel(file_path.filter_ele_excel, dtype={'shot_number': str}, index_col=0)
    #
    # filter_heigh_dataframe = filter_heigh_dataframe.merge(navd_dataframe['elev_lowestmode_NAVD'],left_index=True, right_index=True, how='left')
    #
    # filter_heigh_dataframe.to_excel(file_path.filter_ele_excel)