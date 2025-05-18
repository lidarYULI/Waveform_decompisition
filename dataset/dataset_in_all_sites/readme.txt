
each excel saves combinations of different GEDI products, estimation values from ALS data (elevation, adjusted footprint coordination; reference canopy cover), and main results derived from every part in ground return mode identification.

key columns' meanings in each excel:

######## "shot_number": unique identity; "beam": beam types; "date": GEDI acquisition date; "time": precise acquistion date
######## "GEDI_L1B_File"/"GEDI_L2B_File"/"GEDI_L2A_File": the file path of each data product
######## "rx_energy"/"tx_egamplitude"/"geolocation_solar_elevation"/"rg"/"rv"/: to see GEDI product dictionary.
######## "elev_lowestmode": elevation of lowest waveform mode; "rh100": relative height when energy is cumulated to 100% from lowest mode;
######## "cover_z": vertical profile of CC product; "pag_theta": total gap fraction; "sensitivity": measure how well a beam is able to penetrate the canopy
######## "cloud_mask": 1: this footprint is covered by cloud; otherwise 0. "t_x","t_y”: projected coordinates of GEDI footprint latitude and longitude
######## "month", "site"; which month when the product is acquired, and where the GEDI footprint is located. "site" is the NEON site;
######## "SCI_str_vertical"; SCI cover indices estimated based on unadjusted coordinates; "SCI_str_c": SCI cover indices estimated based on adjusted coordinates;
######## "DHM_98_c": 98th percentile of Digital height model derived from ALS point cloud; 
######## "DEM_NEON_average" GEDI footprint's reference elevation derived from ALS point cloud using direct avergae of ground elevation.
######## "DEM_NEON_weighted" GEDI footprint's reference elevation derived from ALS point cloud using Gaussian weights.
######## "DEM_NEON_min_z" the lowest elevation of ALS point cloud within the GEDI footprint area
######## "DHM_98_amp_joint": 98th percentile of height at adjusted GEDI footprint using maximum amplitude matched method;
######## "SCI_str_amp_joint": SCI at adjusted GEDI footprint using maximum amplitude matched method;
######## "match_x_amplitude_joint"/"match_y_amplitude_joint": adjusted projected coordinates.
######## "elev_lowestmode_a1/-a6" six different elevations of lowest mode based on threshold method


# in D:\Pycharm_Projects\Waveform_decompisition\dataset\dataset_in_all_sites

# 1-L2A_product_all_datafilter.xlsx: from D:\2-forest_reflectance_result\1-NEON_Six_Sites_GEDI_Product\3_beam_only.xlsx, the null data records were deleted because some of GEDI L2A product are not downloaded.

# 2-predictive_variables_all_waveform_mode.xlsx: include 30 predictive variables for all dataset

# 3-filter_needleleaf_elevation.xlsx: filtered high quality data from the reference_dataset_paper_used.xlsx;

# 4-regress_variables_filter_needleleaf_elevation: predictive variables of 3-filter_needleleaf_elevation.xlsx

# 5-reference_dataset_paper_used.xlsx: key dataset existed in 7_RF_comparison.xlsx to meet need by research on this excel as it saves the date-matched GEDI and ALS data.

It is used to select high quality data

# filter_needleleaf_elevation.xlsx: all needleleaf sampled data and they are filtered using method of files_access.py/high_accurate_elevation_filter()




