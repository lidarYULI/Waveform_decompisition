This repository is to implement gaussian decomposition for lidar waveform; 
The code is mostly based on (Hofton et al., 2000); Hofton M A, Minster J B, & Blair J B. Decomposition of laser altimeter waveforms[J]. Ieee Transactions on Geoscience and Remote Sensing. 2000, 38, 1989-1996. https://doi.org/10.1109/36.851780 
The code has been partially tested on GEDI (Global Ecosystem Dynamics Investigation) received waveforms.

Two kinds of models were developed to fit the GEDI waveform. 
The first fits all component waveforms using a Gaussian function (Gaussian decomposition). 
The second additionally uses an extended Gaussian function specifically for ground return fitting, as in the GEDI L2B product. However, the second model may not fully follow GEDI's practices because I do not know whether GEDI directly fits the ground return mode (located at the lowest peak of the waveform) using an extended Gaussian function or if they first perform Gaussian decomposition and then further fit the lowest component result using an extended Gaussian function. In my code, I replaced the Gaussian function with an extended Gaussian function to fit the lowest mode. Apart from this, all details are the same as in the Gaussian decomposition.

This is a preliminary version.

Here is the GEDI waveform labeled by some key parameters derived from GEDI preprocessing algorithm 2

![result](https://github.com/lidarYULI/Waveform_decompisition/blob/master/result_output/waveform_info.png)

"search_start" and "search_end" define the waveform range where reflected signal is searched by algorithm

"toploc": the highest detectable return

"botloc": the lowest detectable signal return

Refer to "Hofton M & Blair J B (2019). Algorithm Theoretical Basis Document (ATBD) for GEDI Transmit and Receive Waveform Processing for L1 and L2 Products. In"


Usage for Gaussian decomposition:

I provide a test_data.xlsx, it includes waveforms and some fields derived from GEDI products;
The description of some columns may be helpful for my collaborators
"rxwaveform": GEDI received waveform
"txwaveform": GEDI transimitted waveform
"zcross": ground return mode location (bins) provided by GEDI product
"zcross_manually": visually selected ground return mode location (bins) in our manuscript
"GEDI_lowestmode_height_NAVD": GEDI elevation reported in the North American Vertical Datum 1988 (NAVD88) used in NEON ALS points cloud.
"DEM_NEON": the average of NEON ALS elevation within the GEDI footprint area
"DHM_98_c": 98th percentile canopy height derived from ALS points cloud
"t_x" and "t_y": coordinates under UTM projection
"ALS_total_CC" and "GEDI_total_CC": canopy cover derived from ALS and GEDI
"selected_l2a_algorithm": the preprocessing algorithm ID used for each waveform

download this repository, run test_gaussia_decomposition() in main.py.
you will see a fig of decomposition result; below is the test function

![result](https://github.com/lidarYULI/Waveform_decompisition/blob/master/result_output/gau_decompistion.png)









