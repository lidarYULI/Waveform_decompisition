Here is the introduction of ground return mode identification process based on GEDI waveform

The ground return mode location is important because it corresponds to elevation and impact return energy estimation

I developed a method based on Random Forest (RF) regressor to identify ground return mode, then I re-estimated canopy cover based on it. This method has been tested in six NEON forest sites

Mode is the first derivative of the waveform; 30 variable of modes from three aspects are calculated and used in RF;

![result](https://github.com/lidarYULI/Waveform_decompisition/blob/master/result_output/Features_definition.png)

![result](https://github.com/lidarYULI/Waveform_decompisition/blob/master/result_output/Waveform_RF.png)

Three steps in my work:
(1) prediction variables/features generation; and dependent variable generation

the dependent variable is the mode height relative to reference elevation;

The reference elevation can be determined from three ways: 

the first is visual selection from waveform based on reference ALS elevation and canopy cover;
"in 8-test_data.xlsx, zcross_manually indicates the manually selected ground return mode"

the second is to directly determine the ground position by ALS elevation;
The "DEM_NEON_weighted" in many of excels means the reference elevation derived from ALS point cloud

the third is to take the GEDI high accurate products as the reference elevation 
The "zcross" in many of excels means the GEDI elevation

generate predictive variables:
run Samples_generation/derivative_samples/generate_predicted_samples(): take the first derivative of waveform as the mode


(2) RF training and prediction

to train and optimize RF regressor based on training sample from last steps:

Run: Regress_ground/RF_regressor_prediction/RF_regression_py/RF_regression_optimization()

test: Regress_ground/RF_regressor_prediction/RF_regression_py/RF_regression_test()

predict: Regress_ground/RF_regressor_prediction/RF_regression_py/RF_regression_predict()

select ground return mode with absolute minimum predicted height: RF_regression_selection()


(3) canopy cover recalculation

use product_derive.py to calculate canopy cover

RF_excel = file_path.RF_excel

decompose_based_selected_zcross(RF_excel,'randomforest_zcross')

in this process, ground return mode is fitted by gaussian function or extended gaussian function to separate the canopy waveform and ground waveform

the identified ground return mode location and corresponding amplitude are used in fitting;

the boundary sets for gaussian fitting are important in subsequent product estimation

Three columns will be added if using gaussian fit

'RV_RF_z': cumulative canopy return energy at different heights

'CC_RF_z': cumulative canopy cover at different heights;

'Fitted_parameters_Rg_RF_GAU': ground mode fitting parameters: amplitude, center, and sigma 


