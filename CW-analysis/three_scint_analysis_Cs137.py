# General approach: Calibration vs Analysis Runs
# For calibration runs, you've let detectors continuously colelct data and need to manually fit Moyal distribution to get calibration cosntants (MPVs)
# For analysis runs, you're not manually fitting the Moyal distribution, but rather, using the calibration constants from the calibration run as arguments for the moyal fitting
# For both, the initial steps of the pipeline are Datalogger_Processing --> Scintillators_Processing --> Detector_Analysis, the only difference is for a CALIBRATION RUN, you then
# do calibrate_and_analyze_grounddata(), starting with guesses of the x-axis ranges for fitting, and then look at where they fall on the plot and adjust to the MIP bump by re-running the script
# For an ANALYSIS RUN, you do analyze_calibrated_data_with_fixed_MPVs() and use MPVs calculated from the calibation run as an input.

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
RESULTS_DIR = '/Users/emmamartignoni/Desktop/Desktop - Emma’s MacBook Pro (3)/HERA-Research/Results'

# ============================ Cs 137: ANALYSIS RUN =============================
cs137_datalogger_fp = f'{DATA_DIR}/Cs 137/data logger/AHD001.csv' # @param {type:"string"}
cs137_top_scint_fp = f'{DATA_DIR}/Cs 137/left/leftAxLab_M_029.txt' # @param {type:"string"}
cs137_mid_scint_fp = f'{DATA_DIR}/Cs 137/middle/middleAxLab_M_028.txt' # @param {type:"string"}
cs137_bot_scint_fp = f'{DATA_DIR}/Cs 137/right/rightAxLab_M_029.txt'

# Processing pipeline for any analysis data where you DON'T need to fit the Moyal distribution -- you ALREADY DID MOYAL FIT WITH OTHER RUNS OF THE SAME SCINTILLATORS, AND NEED TO APPLY RESULTS OF THOSE CALIBRATIONS CONSTANTS TO GROUND DATA
# Datalogger_Processing --> Scintillators_Processing --> Detector_Analysis again, only difference is that you now run analyze_calibrated_data_with_fixed_MPVs() and give it the MPVs output from the calibation run you're using 
cs_df = Datalogger_Processing(cs137_datalogger_fp).process()
three_scintillator_cs = Scintillators_Processing([cs137_top_scint_fp, cs137_mid_scint_fp, cs137_bot_scint_fp], cs_df)
analysis_cs = Detector_Analysis(three_scintillator_cs, cs_df, results_dir=os.path.join(RESULTS_DIR, 'cs137'))
analysis_cs.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])

plot_density_heatmap_ampcal(analysis_cs, normalize_by_livetime=True)



# ============================ Cs 137 background run =============================
cs137background_datalogger_fp = f'{DATA_DIR}/Cs137_Background/AHD013.csv' # @param {type:"string"}
cs137background_top_scint_fp = f'{DATA_DIR}/Cs137_Background/left_AxLab_M_040.txt' # @param {type:"string"}
cs137background_mid_scint_fp = f'{DATA_DIR}/Cs137_Background/middle_AxLab_M_039.txt' # @param {type:"string"}
cs137background_bot_scint_fp = f'{DATA_DIR}/Cs137_Background/right_AxLab_M_040.txt'

cs_df = Datalogger_Processing(cs137background_datalogger_fp).process()

three_scintillator_cs = Scintillators_Processing([cs137background_top_scint_fp, cs137background_mid_scint_fp, cs137background_bot_scint_fp], cs_df)
analysis_cs = Detector_Analysis(three_scintillator_cs, cs_df, results_dir=os.path.join(RESULTS_DIR, 'cs137'))
analysis_cs.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])

plot_density_heatmap_ampcal(analysis_cs, normalize_by_livetime=True)