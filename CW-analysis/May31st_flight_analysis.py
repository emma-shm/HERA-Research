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
from flightanalysis_methods import *
import inspect

DATA_DIR = '/Users/emmamartignoni/Desktop/Desktop - Emma’s MacBook Pro (3)/HERA-Research/Data/'
RESULTS_DIR = '/Users/emmamartignoni/HERA-Research/CW-analysis/Results/May31st_Flight'


# ========================= Isolate flight in full files =========================

datalogger_csv_fp = f'{DATA_DIR}/May_31st_Flight/AHD011 copy.csv' # @param {type:"string"}
top_scint_fp = f'{DATA_DIR}/May_31st_Flight/left_AxLab_M_038.txt' # @param {type:"string"}
mid_scint_fp = f'{DATA_DIR}/May_31st_Flight/middle_AxLab_M_037 copy.txt' # @param {type:"string"}
bot_scint_fp = f'{DATA_DIR}/May_31st_Flight/right_AxLab_M_038 copy.txt' # @param {type:"string"}

# Original / full flight data
og_flight_df = Datalogger_Processing(datalogger_csv_fp, show_plots=True).process()

# Trimming datalogger and scintillator files based on timestamp of altitude where the balloon returns to ground
dl_fp, scint_fps_trimmed = trim_to_flight(datalogger_csv_fp, [top_scint_fp, mid_scint_fp, bot_scint_fp])

# Re-run datalogger to make new, trimmed version of flight
trimmed_flight_df = Datalogger_Processing(dl_fp, show_plots=True).process()


# plot Altitude vs time to check my work
fig_alt = plt.figure(figsize=(10, 6))
# plt.plot(og_flight_df['Absolute Timer (S)'], og_flight_df['Altitude[m]']*3.281, label='CW1&2&3', color='blue')
# plt.xlabel('Timer')
# plt.ylabel('Altitude [ft]')
plt.plot(trimmed_flight_df['Absolute Timer (S)'], trimmed_flight_df['Altitude[m]']*3.281, label='Flight', color='blue')
plt.xlabel('Timer')
plt.ylabel('Altitude [ft]')
plt.legend()
finish_mpl(fig_alt, "altitude_check")
print(f"Max altitude of original launch data: {og_flight_df['Altitude[m]'].max()*3.281} ft")
print(f"Max altitude (trimmed), should be same as max altitude of original launch data: {trimmed_flight_df['Altitude[m]'].max()*3.281} ft")
print(f"The first few rows of the new datalogger dataframe:\n{trimmed_flight_df.head()}")



# ========================= Split flight and background =========================

segments = analyze_flight_in_segments(
    dl_fp,                          # the trimmed datalogger CSV from trim_to_flight
    scint_fps_trimmed,
    MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
    flight_df=trimmed_flight_df,    # used only to locate the segment boundaries
    n_segments=4,
    split_by="time",                # or "events"
    run_name="may31flight",
    results_dir=os.path.join(RESULTS_DIR),
    cbar_max=0.035
)

# grab any segment's analysis object for further work (Tcal columns, etc.)
analysis_seg2 = segments["seg2"]["analysis"]