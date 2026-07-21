# HASP-Drexel
Scripts for processing and analyzing HASP data. Data structure is different than that of the normal HERA CosmicWatch data, and requires different analysis methods.


## HASP_data_processing.py

Main orchestration script. Loads the raw teensy CSVs for one run, merges them into a single per-event DataFrame, and hands that off to + Scintillator_Processing for analysis. No analysis logic in this script — just I/O, pre-processing, and dispatch.
+ Raw data per run: two event-driven SiPM-teensy CSVs (sipm_teensy_1, sipm_teensy_2), one row per detection event. Each file has 16 trigger_NN_binary columns (1 = scint NN fired) and 16 trigger_NN_signal_time columns (timer value when it fired). Both teensies log all 16 trigger columns — that redundancy is what lets us sync them. Teensy 1 carries real ADC values for layers 1 and 3; teensy 2 for layers 2 and 4.
+ Pre-processing: load both teensy CSVs, then merge into one unified per-event DataFrame by matching events between teensies on trigger pattern + signal time within a tolerance. Handle orphan events on either side (e.g. from a teensy reboot) rather than assuming 1:1 row correspondence.
+ Drift diagnostic: for each matched event and each fired trigger, compute the signal-time difference between teensies. Plot the distribution and the trend over time to check whether the offset is constant or drifting.
+ Handoff: pass the unified DataFrame plus a config dict (trigger → physical-scint mapping, coincidence groups) to Scintillator_Processing. Coincidence is then a row-wise AND over the relevant trigger binaries — no external coincidence counter needed.

## sipm_teensy_1_template_eray.csv and sipm_teensy_2_template_eray.csv
Sample data files provided by Eray that are meant to exemplify what the data will look like. The data in these CSVs was generated randomly, but patterns do not match for the event detections on each scintillator at corresponding time stamps since the data was generated randomly, so I generated my own data files, **sipm_teensy_1_test.csv and sipm_teensy_2_test.csv**


## HASP_calibration_methods.py
NOT YET COMPLETE. Script that should have the same calibration and analysis functionality as the calibration_methods.py script for the three-scintillator, standard HERA setup but adapted to the sixteen scintillator data.
