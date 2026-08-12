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


# ========================= Isolate flight and analyze in segments =========================

print("\n PART 1: Isolate flight data from the full datalogger and scintillator files, then analyze in segments \n")

datalogger_csv_fp = f'{DATA_DIR}/May_31st_Flight/AHD011 copy.csv' # @param {type:"string"}
top_scint_fp = f'{DATA_DIR}/May_31st_Flight/left_AxLab_M_038.txt' # @param {type:"string"}
mid_scint_fp = f'{DATA_DIR}/May_31st_Flight/middle_AxLab_M_037 copy.txt' # @param {type:"string"}
bot_scint_fp = f'{DATA_DIR}/May_31st_Flight/right_AxLab_M_038 copy.txt' # @param {type:"string"}

og_flight_df      = Datalogger_Processing(datalogger_csv_fp, show_plots=False, results_dir=RESULTS_DIR).process(name="original_flight_datalogger") # Original / full flight data


dl_fp, scint_fps_trimmed = trim_to_flight(datalogger_csv_fp, [top_scint_fp, mid_scint_fp, bot_scint_fp]) # Trimming datalogger and scintillator files based on timestamp of altitude where the balloon returns to ground


trimmed_flight_df = Datalogger_Processing(dl_fp, show_plots=False, results_dir=RESULTS_DIR).process(name="trimmed_flight_datalogger") # Re-run datalogger to make new, trimmed version of flight -- just to see


# plot Altitude vs time to check my work
fig_alt = plt.figure(figsize=(10, 6))
# plt.plot(og_flight_df['Absolute Timer (S)'], og_flight_df['Altitude[m]']*3.281, label='CW1&2&3', color='blue')
# plt.xlabel('Timer')
# plt.ylabel('Altitude [ft]')
plt.plot(trimmed_flight_df['Absolute Timer (S)'], trimmed_flight_df['Altitude[m]']*3.281, label='Flight', color='blue')
plt.xlabel('Timer')
plt.ylabel('Altitude [ft]')
plt.title('Altitude vs Time for Trimmed Flight Data')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "altitude_check_trimmed_flight.png"))
print(f"Max altitude of original launch data: {og_flight_df['Altitude[m]'].max()*3.281} ft")
print(f"Max altitude (trimmed), should be same as max altitude of original launch data: {trimmed_flight_df['Altitude[m]'].max()*3.281} ft")
print(f"The first few rows of the new datalogger dataframe:\n{trimmed_flight_df.head()}")


segments = analyze_flight_in_segments(
    dl_fp,
    scint_fps_trimmed,
    MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
    flight_df=trimmed_flight_df,
    n_segments=32, # THIS LINE is where you can change the number of bins / segments you split the flight into
    split_by="time",
    run_name="may31flight_fine",
)

# # grab any segment's analysis object for further work (Tcal columns, etc.)
# analysis_seg2 = segments["seg2"]

print("\n PART 2: Count events in the negative slope region of the MIP vs MIP_std heatmap, per segment \n")

# ================== Total counts in the negative slope region, per segment ==================
BOX_X = (1.7, 3.7)   # Mean MIP range
BOX_Y = (1.7, 4.0)   # Std MIP range

segment_numbers = []
counts = []
t_mid = []

for i, (lbl, analysis) in enumerate(segments.items(), start=1):

    # md is the master dataframe of the analysis object for the given segment
    md = analysis.master_df

    # Select events inside the specified window
    box = md[md["SiPM_scints_avg_MIP"].between(*BOX_X) & md["SiPM_scints_std_MIP"].between(*BOX_Y)]

    count = len(box)
    print(f"Segment {i} ({lbl}): {count} counts in the window {BOX_X} x {BOX_Y}")

    segment_numbers.append(i)
    counts.append(count)

    t_mid.append(md["Absolute Timer (S)"].mean())   # segment mean time [min]

    print(f"{lbl}: {count} counts")

    # Heatmap of only the selected window
    fig = px.density_heatmap(
        box,
        x="SiPM_scints_avg_MIP",
        y="SiPM_scints_std_MIP",
        nbinsx=25,
        nbinsy=25,
        range_color=(0, 200),
        color_continuous_scale="Inferno",
        title=f"{lbl}: {count} Counts")
    # fig.write_image(os.path.join(RESULTS_DIR, f"heatmap_windowed_{lbl}.png"))
    # fig.show()


# Plot counts by segment
plt.figure(figsize=(10, 6))
plt.plot(t_mid, counts, 'o-')
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.title(f"Total Counts in Selected Window ({BOX_X[0]} - {BOX_X[1]} mean MIP, {BOX_Y[0]} - {BOX_Y[1]} std MIP) by Flight Segment")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "counts_by_segment.png"))
plt.show()


# ========================= Isolate ground before flight in full files =========================

print("\n PART 3: Isolate ground data before flight in the full datalogger and scintillator files, then analyze \n")

# Trimming datalogger and scintillator files based on timestamp of altitude where the balloon returns to ground
sections_flight = split_by_time_marks(datalogger_csv_fp, [top_scint_fp, mid_scint_fp, bot_scint_fp],
                                      time_marks=[0, 3350, 9000, 42750],
                                      labels=['pre-flight', 'flight', 'post-flight'],
                                      )


# isolating the pre-flight section
pre_flight = next(sec for sec in sections_flight if sec['label'] == 'pre-flight')

print(f"\n=== {pre_flight['label']} ===")
df, processor, analysis = process_run(
    pre_flight['datalogger'], pre_flight['scints'],
    MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
    results_dir=os.path.join(RESULTS_DIR, pre_flight['label']),
    twodim_hist_args={'cbar_max': 1},
)

fig_alt = plt.figure(figsize=(10, 6))
plt.plot(df['Absolute Timer (S)'], df['Altitude[m]']*3.281, label='Ground', color='blue')
plt.xlabel('Timer')
plt.ylabel('Altitude [ft]')
plt.title(f'Altitude vs Time for Section {pre_flight["label"]}')
plt.savefig(os.path.join(RESULTS_DIR, f"altitude_check_{pre_flight['label']}.png"))
plt.close(fig_alt)


# # looping through each of the sections created by split_by_time_marks
# flight_runs = {}

# for sec in sections_flight:
#     print(f"\n=== {sec['label']} ===")
#     df, processor, analysis = process_run(
#         sec['datalogger'], sec['scints'],
#         MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040],
#         results_dir=os.path.join(RESULTS_DIR, sec['label']),
#         twodim_hist_args={'cbar_max': 1},
#     )

#     fig_alt = plt.figure(figsize=(10, 6))
#     plt.plot(df['Absolute Timer (S)'], df['Altitude[m]']*3.281, label='Flight', color='blue')
#     plt.xlabel('Timer')
#     plt.ylabel('Altitude [ft]')
#     plt.title(f'Altitude vs Time for Section {sec["label"]}')
#     plt.savefig(os.path.join(RESULTS_DIR, f"altitude_check_{sec['label']}.png"))
#     plt.close(fig_alt)

#     flight_runs[sec['label']] = {'df': df, 'processor': processor, 'analysis': analysis}