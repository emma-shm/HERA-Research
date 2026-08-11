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
RESULTS_DIR = '/Users/emmamartignoni/HERA-Research/CW-analysis/Results/Europium'



# ============================ Eu background run =============================
eu_background_datalogger_fp = f'{DATA_DIR}/Eu Background Vert/AHD016.csv' # @param {type:"string"}
eu_background_top_scint_fp = f'{DATA_DIR}/Eu Background Vert/Top.txt' # @param {type:"string"}
eu_background_mid_scint_fp = f'{DATA_DIR}/Eu Background Vert/Mid.txt' # @param {type:"string"}
eu_background_bot_scint_fp = f'{DATA_DIR}/Eu Background Vert/Bot.txt'

# eu_df_background = Datalogger_Processing(eu_background_datalogger_fp, show_plots=False).process()

# three_scintillator_eu_background = Scintillators_Processing([eu_background_top_scint_fp, eu_background_mid_scint_fp, eu_background_bot_scint_fp], eu_df_background, show_plots=False)
# analysis_eu_background = Detector_Analysis(three_scintillator_eu_background, eu_df_background, results_dir=os.path.join(RESULTS_DIR, 'Background'))
# # analysis_eu.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])
# analysis_eu_background.calibrate_and_analyze_grounddata(moyal_fit_ranges=[(46-1, 80-3), (46-2, 82-2), (44-1, 76-1)], twodim_hist_args={'cbar_max': 300})
# # print(analysis_eu.mpv_per_scint[1])

europium_background = process_run(eu_background_datalogger_fp,
                                    [eu_background_top_scint_fp, eu_background_mid_scint_fp, eu_background_bot_scint_fp],
                                    results_dir=os.path.join(RESULTS_DIR, 'Background'),
                                    moyal_fit_ranges=[(46-1, 80-3), (46-2, 82-2), (44-1, 76-1)],
                                    twodim_hist_args={'col': 'MIP', 'cbar_max': 300}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable

analysis_eu_background = europium_background[2] # the Detector_Analysis object is the 3rd element of the tuple returned by process_run()

# ============================ Eu: ANALYSIS RUN =============================
eu_datalogger_fp = f'{DATA_DIR}/Eu Vert/Datalogger.csv' # @param {type:"string"}
eu_top_scint_fp = f'{DATA_DIR}/Eu Vert/top.txt' # @param {type:"string"}
eu_mid_scint_fp = f'{DATA_DIR}/Eu Vert/middle.txt' # @param {type:"string"}
eu_bot_scint_fp = f'{DATA_DIR}/Eu Vert/bottom.txt'

# # Processing pipeline for any analysis data where you DON'T need to fit the Moyal distribution -- you ALREADY DID MOYAL FIT WITH OTHER RUNS OF THE SAME SCINTILLATORS, AND NEED TO APPLY RESULTS OF THOSE CALIBRATIONS CONSTANTS TO GROUND DATA
# # Datalogger_Processing --> Scintillators_Processing --> Detector_Analysis again, only difference is that you now run analyze_calibrated_data_with_fixed_MPVs() and give it the MPVs output from the calibation run you're using 
# eu_df = Datalogger_Processing(eu_datalogger_fp, show_plots=False).process()
# three_scintillator_eu = Scintillators_Processing([eu_top_scint_fp, eu_mid_scint_fp, eu_bot_scint_fp], eu_df, show_plots=False)
# analysis_eu = Detector_Analysis(three_scintillator_eu, eu_df, results_dir=os.path.join(RESULTS_DIR, 'Source'))
# analysis_eu.analyze_calibrated_data_with_fixed_MPVs(MPVs=[analysis_eu_background.mpv_per_scint[1], analysis_eu_background.mpv_per_scint[2], analysis_eu_background.mpv_per_scint[3]], twodim_hist_args={'col': 'MIP', 'cbar_max': 300})
# # # Plotting the 2D histograms of the calibrated data for the Eu analysis run to see if the amplitude calibrations make a huge difference to the 2D histograms
# # analysis_eu_.two_dimensional_histograms(col='MIP_ampcal')

europium_source = process_run(eu_datalogger_fp,
                                [eu_top_scint_fp, eu_mid_scint_fp, eu_bot_scint_fp],
                                    results_dir=os.path.join(RESULTS_DIR, 'Source'),
                                    MPVs=[analysis_eu_background.mpv_per_scint[1], analysis_eu_background.mpv_per_scint[2], analysis_eu_background.mpv_per_scint[3]],
                                    twodim_hist_args={'col': 'MIP', 'cbar_max': 300}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable
