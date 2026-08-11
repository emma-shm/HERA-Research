# General approach: Calibration vs Analysis Runs
# For calibration runs, you've let detectors continuously collect data and need to manually fit Moyal distribution to get calibration cosntants (MPVs)
# For analysis runs, you're not manually fitting the Moyal distribution, but rather, using the calibration constants from the calibration run as arguments for the moyal fitting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import linregress
import os
import calibration_methods
from calibration_methods import *
import inspect

DATA_DIR = '/Users/emmamartignoni/Desktop/Desktop - Emma’s MacBook Pro (3)/HERA-Research/Data/'
RESULTS_DIR = '/Users/emmamartignoni/HERA-Research/CW-analysis/Results/Cs137'

# ============================ Cs 137 background run =============================
cs137background_datalogger_fp = f'{DATA_DIR}/Cs137_Background/AHD013.csv' # @param {type:"string"}
cs137background_top_scint_fp = f'{DATA_DIR}/Cs137_Background/left_AxLab_M_040.txt' # @param {type:"string"}
cs137background_mid_scint_fp = f'{DATA_DIR}/Cs137_Background/middle_AxLab_M_039.txt' # @param {type:"string"}
cs137background_bot_scint_fp = f'{DATA_DIR}/Cs137_Background/right_AxLab_M_040.txt'

# cs_df = Datalogger_Processing(cs137background_datalogger_fp).process()
# three_scintillator_cs = Scintillators_Processing([cs137background_top_scint_fp, cs137background_mid_scint_fp, cs137background_bot_scint_fp], cs_df)
# analysis_cs = Detector_Analysis(three_scintillator_cs, cs_df, results_dir=os.path.join(RESULTS_DIR, 'Background'))
# analysis_cs.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])


cesium137_background = process_run(cs137background_datalogger_fp,
                                   [cs137background_top_scint_fp, cs137background_mid_scint_fp, cs137background_bot_scint_fp],
                                   results_dir=os.path.join(RESULTS_DIR, 'Background'),
                                   MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
                                   twodim_hist_args={'col': 'MIP', 'cbar_max': 1000}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable


# ============================ Cs 137: ANALYSIS RUN =============================
cs137_datalogger_fp = f'{DATA_DIR}/Cs 137/data logger/AHD001.csv' # @param {type:"string"}
cs137_top_scint_fp = f'{DATA_DIR}/Cs 137/left/leftAxLab_M_029.txt' # @param {type:"string"}
cs137_mid_scint_fp = f'{DATA_DIR}/Cs 137/middle/middleAxLab_M_028.txt' # @param {type:"string"}
cs137_bot_scint_fp = f'{DATA_DIR}/Cs 137/right/rightAxLab_M_029.txt'

# # Processing pipeline for any analysis data where you DON'T need to fit the Moyal distribution -- you ALREADY DID MOYAL FIT WITH OTHER RUNS OF THE SAME SCINTILLATORS, AND NEED TO APPLY RESULTS OF THOSE CALIBRATIONS CONSTANTS TO GROUND DATA
# # Datalogger_Processing --> Scintillators_Processing --> Detector_Analysis again, only difference is that you now run analyze_calibrated_data_with_fixed_MPVs() and give it the MPVs output from the calibation run you're using 
# cs137_dir = os.path.join(RESULTS_DIR, 'Source')

# cs137_df = Datalogger_Processing(cs137_datalogger_fp, results_dir=cs137_dir).process()
# three_scintillator_cs137 = Scintillators_Processing([cs137_top_scint_fp, cs137_mid_scint_fp, cs137_bot_scint_fp], cs137_df, results_dir=cs137_dir)
# analysis_cs137 = Detector_Analysis(three_scintillator_cs137, cs137_df, results_dir=cs137_dir)
# analysis_cs137.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040], twodim_hist_args={'col': 'MIP', 'cbar_max': 0.0005})

# # # Plotting the 2D histograms of the calibrated data for the Cs 137 analysis run to see if the amplitude calibrations make a huge difference to the 2D histograms
# # analysis_cs137.two_dimensional_histograms(col='MIP_ampcal')

cesium137_source = process_run(cs137_datalogger_fp,
                                [cs137_top_scint_fp, cs137_mid_scint_fp, cs137_bot_scint_fp],
                                    results_dir=os.path.join(RESULTS_DIR, 'Source'),
                                    MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
                                    twodim_hist_args={'col': 'MIP', 'cbar_max': 1000}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable