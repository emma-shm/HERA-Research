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

DATA_DIR = '/Users/emmamartignoni/Desktop/Desktop - Emma’s MacBook Pro (3)/HERA-Research/Data'
RESULTS_DIR = '/Users/emmamartignoni/HERA-Research/CW-analysis/Results/Na-22'

TRIM_S = 57600   # shortest time is livetime on Scint 1, which is 16 hours

# ============================ Na-22 background run =============================
print("\n PART 1: Analyze Na-22 background run \n")
bg = split_by_time_marks(f'{DATA_DIR}/Background_C_Na/AHD039.csv',
                        [f'{DATA_DIR}/Background_C_Na/top.txt', f'{DATA_DIR}/Background_C_Na/mid.txt', f'{DATA_DIR}/Background_C_Na/bot.txt'],
                        [0, TRIM_S],
                        labels=["trim"])[0] # function returns list of dicts, one per section, in time order so getting the 0th item in that list is the first

print(f"Running full analysis on datalogger and scintillator files trimmed to {TRIM_S} s livetime")
na_background = process_run(bg['datalogger'], bg['scints'], # bg is the dict returned by split_by_time_marks() for the trimmed section; the key 'datalogger' gives filepath to trimed datalogger, same with 'scints' key
                            results_dir=os.path.join(RESULTS_DIR, 'Background'),
                            moyal_fit_ranges=[(18, 31), (18, 31), (18, 28)],
                            twodim_hist_args={'col': 'MIP', 'cbar_max': 250, 'range_x': [0, 8], 'range_y': [0, 8]}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable

analysis_na_background = na_background[2] # the Detector_Analysis object is the 3rd element of the tuple returned by process_run()

# ============================ Na-22 source run =============================
print("\n PART 2: Analyze Na-22 source run \n")
src = split_by_time_marks(f'{DATA_DIR}/Na-22/AHD038.csv',
                        [f'{DATA_DIR}/Na-22/top.txt', f'{DATA_DIR}/Na-22/mid.txt', f'{DATA_DIR}/Na-22/bot.txt'],
                        [0, TRIM_S],
                        labels=["trim"])[0]

na_source = process_run(src['datalogger'], src['scints'],
                        results_dir=os.path.join(RESULTS_DIR, 'Source'),
                        MPVs=[analysis_na_background.mpv_per_scint[1], analysis_na_background.mpv_per_scint[2], analysis_na_background.mpv_per_scint[3]],
                        twodim_hist_args={'col': 'MIP', 'cbar_max': 250, 'range_x': [0, 8], 'range_y': [0, 8]}) # col can be 'MIP' or 'MIP_ampcal', cbar_max is the max value of the colorbar for the 2D histograms, which you can adjust to make the plots more readable



# ============================ Source − Background subtraction map =============================
print("\n PART 3: Subtract background from source run to get net source counts \n")
subtraction_map(na_source[2], na_background[2], cbar_max=75, results_dir=RESULTS_DIR)



print(f"MPVs derived from background run (to be used for other runs): {analysis_na_background.mpv_per_scint}")