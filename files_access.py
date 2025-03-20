import os

class file_path:

    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file_path)

    # excel, all GEDI records used in this file; for results derivation and RF prediction
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
    elevation_train = os.path.join(project_root, 'result_output', 'elevation_train')

    # RF training by the samples generated based on manual selection of ground location
    manually_train = os.path.join(project_root, 'result_output', 'visual_train')