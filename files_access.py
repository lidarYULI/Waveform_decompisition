import os

class file_path:

    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file_path)

    dataset_folder = os.path.join(project_root,'dataset')

    result_folder = os.path.join(project_root,'result_output')

    repository_folder = os.path.join(dataset_folder,'dataset_in_all_sites')

    ### RF regressor saving directory
    RF_regressor_workspace = os.path.join(project_root, 'Regress_ground','RF_regressor_prediction')

    ### RF classifier saving directory
    RF_classifier_workspace = os.path.join(project_root, 'Regress_ground', 'RF_classifier_prediction')

    # <editor-fold desc="dataset repository">
    all_GEDI_dataset_excel = os.path.join(repository_folder, '1_L2A_product_all_datafilter.xlsx')

    all_predictive_variables_excel = os.path.join(repository_folder, '2-predictive_variables_all_waveform_mode.xlsx')

    ICCG_needleleaf_filter_excel = os.path.join(repository_folder, '3-filter_needleleaf_elevation.xlsx')

    ICCG_needleleaf_filter_predictive_variables_excel = os.path.join(repository_folder, '4-regress_variables_filter_needleleaf_elevation')

    ICCG_reference_excel = os.path.join(repository_folder, '5-reference_dataset_paper_used.xlsx')

    ### dataset repository
    # </editor-fold>

    # <editor-fold desc="high accurate filter output">
    stratified_filter_save_excel = os.path.join(repository_folder, '6-filter_high_accurate.xlsx')
    # </editor-fold>

    # <editor-fold desc="important excels in RF regression">
    # excel, all GEDI records used in this file; for RF prediction and getting reuslts
    RF_excel = os.path.join(dataset_folder, '1_RF_comparison.xlsx')

    ## RF prediction variabels for modes (derivatives) of the waveform
    RF_derivative_features_excel = os.path.join(dataset_folder,'2-RF_derivative_features_excel.xlsx')

    ### varied thresholds are used to select predicted mode
    varied_threshold_test_save_excel = os.path.join(dataset_folder, '3-varied_thresholds_results.xlsx')

    ### transferability test results
    transferability_save_excel = os.path.join(dataset_folder, '4-transferability_test.xlsx')

    variable_importance_save_excel = os.path.join(dataset_folder, '5-importance.xlsx')

    ### prediction variables for ground return mode
    RF_ground_features_excel = os.path.join(project_root, 'Regress_ground', '6-RF_ground_features.xlsx')

    ### RF classification result/excel
    RF_waveform_GMM_classification_excel = os.path.join(dataset_folder, '7_RF_comparison_GMM_classification.xlsx')

    # test excel,manually selected samples; to select ground return location,
    # the manually selected location is saved as column of "manually_zcross"
    manual_sample_excel = os.path.join(dataset_folder, '8-test_data.xlsx')

    manual_sample_prediction_variables_excel = os.path.join(dataset_folder, '9-test_derive_features_excel.xlsx')
    # </editor-fold>



    # <editor-fold desc="RF regression txt results">

    ### feature number test
    RF_regression_feature_num_output_txt = os.path.join(result_folder, 'RF_regression_feature_num.txt')

    ### RF regression parameters stored text
    RF_regression_parameters_output_txt = os.path.join(result_folder, 'RF_regression_parameters.txt')

    ### RF regression parameters stored text
    RF_regression_proportion_output_txt = os.path.join(result_folder, 'RF_regression_proportion.txt')

    ### RF regression parameters stored text
    RF_regression_model_record_txt = os.path.join(result_folder, 'RF_model_records.txt')

    ### RF_regression_model saving address
    amplitude_thresholds_metrics_output_txt = os.path.join(result_folder, 'filter_threshold.txt')

    # </editor-fold>

    # <editor-fold desc="regression by Gau decomposed results and manually selected samples">
    ### RF modes features
    RF_Gaussian_modes_features_excel = os.path.join(result_folder, 'gaussian_decomposition', 'Gaussian_modes_features_RF_excel.xlsx')

    ### mode features derived from test.xlsx
    Gau_decomposed_test_modes_features_excel = os.path.join(result_folder, 'gaussian_decomposition', 'Gaussian_modes_features_test_excel.xlsx')

    # gaussian decomposition results saving path
    Gau_result_path = os.path.join(result_folder, 'gaussian_decomposition')

    # txt; gaussian decomposition result;
    Gau_Decomposition_txt = os.path.join(Gau_result_path, 'Gau_decom.txt')

    # txt; iterative gaussian decomposition result;
    Iterative_GAU_txt = os.path.join(Gau_result_path, 'Gau_decom_iterative.txt')

    ########### RF training by the samples generated based on manual selection of ground location
    manual_samples_workspace = os.path.join(result_folder, 'training_visualsamples')

    ### it is generated from 8-test_data.xlsx
    manually_derivative_predictive_samples = os.path.join(manual_samples_workspace, 'manual_selection_samples.xlsx')

    # </editor-fold>

    # <editor-fold desc="GMM classification related data">
    ### ground features PCA
    RF_ground_features_PCA_txt = os.path.join(result_folder, 'RF_ground_features_PCA.txt')

    ### RF waveform PCA result
    RF_waveform_PCA_txt = os.path.join(result_folder, 'RF_waveform_PCA.txt')

    ### fitted waveform txt file
    RF_fitted_waveform_txt = os.path.join(result_folder, 'RF_fitted_waveform.txt')

    ### RF fitted_waveform_PCA_txt
    RF_fitted_waveform_PCA_txt = os.path.join(result_folder, 'RF_fitted_waveform_PCA.txt')

    ### RF classification result/txt
    RF_waveform_classification_txt = os.path.join(project_root, 'Regress_ground', 'GMM_classification_result.txt')

    ########### GMM classification related results
    # </editor-fold>


    ### RF_regression_model saving address
    RF_regression_model_folder = os.path.join(RF_regressor_workspace, 'RF_regression_model')

    ### RF classifiers saving address
    RF_classifier_save_dir = os.path.join(RF_classifier_workspace, 'RF_classifier_save_dir')

