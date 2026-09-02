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

# ========================= Helper functions =========================

def trim_to_flight(datalogger_fp, scint_fps, ground_band=50.0, lead_minutes=55.0):
    # Build Absolute Timer with the existing class.
    dl = Datalogger_Processing(datalogger_fp, show_plots=False)
    dl.process()
    df = dl.df.copy()
    df['Timer[S]'] = df['Absolute Timer (S)']   # flatten resets so the trimmed file reprocesses cleanly

    # Pre-launch warmup: drop everything before launch.
    t_start = lead_minutes * 60.0

    # Cut where altitude returns to ground after apogee.
    alt = df['Altitude[m]']
    ground   = alt.iloc[:50].median()
    peak_idx = alt.idxmax()
    landed   = alt.loc[peak_idx:][alt.loc[peak_idx:] <= ground + ground_band]
    t_cut    = df['Absolute Timer (S)'].loc[landed.index[0]]
    print(f"Keeping {t_start:.0f}-{t_cut:.0f} s "
          f"(trimmed first {lead_minutes:.0f} min, cut at t = {t_cut:.0f} s)")

    out_dir = os.path.dirname(datalogger_fp)

    # Datalogger: keep rows in [t_start, t_cut], drop helper cols so the class rebuilds them.
    df = df.drop(columns=[c for c in ['Absolute Timer (S)', 'Timer_rel'] if c in df.columns])
    base = os.path.splitext(os.path.basename(datalogger_fp))[0]
    dl_out = os.path.join(out_dir, f"{base}_flight.csv")
    dl_mask = (df['Timer[S]'] >= t_start) & (df['Timer[S]'] <= t_cut)
    df[dl_mask].to_csv(dl_out, index=False)

    # Scintillators: same window on Time[s], keep the 3-line header + tab format.
    cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]','SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
    scint_out = []
    for fp in scint_fps:
        with open(fp) as f:
            header = [next(f) for _ in range(3)]
        sdf = pd.read_csv(fp, sep='\t', comment='#', header=None, skiprows=3, names=cols, engine='python')
        sbase = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(out_dir, f"{sbase}_flight.txt")
        s_mask = (sdf['Time[s]'] >= t_start) & (sdf['Time[s]'] <= t_cut)
        with open(out, 'w') as f:
            f.writelines(header)
            sdf[s_mask].to_csv(f, sep='\t', header=False, index=False)
        scint_out.append(out)

    return dl_out, scint_out


def analyze_flight_in_segments(
    datalogger_fp,
    scint_fps,
    MPVs,
    flight_df=None,
    n_segments=4,
    split_by="time",
    time_marks=None,
    labels=None,
    noise_threshold=0.1,
    mip_window=(0.8, 1.2),
    twodim_hist_args=None,
    run_name="flight",
    results_dir=None,
):
    """
    Split a flight into segments and run the fixed-MPV pipeline on each one,
    writing every segment's figures to its own folder.

    Cut points: time_marks if given, otherwise n_segments+1 points spaced evenly
    in time ("time") or in event quantiles ("events").

    Each segment is handed to process_run with MPVs and save_all=True, so no
    Moyal fit is performed and the supplied MPVs are used directly.

    Args:
        datalogger_fp : path to the datalogger CSV
        scint_fps     : list of scintillator TXT paths
        MPVs          : list of 3 floats, one per scintillator
        flight_df     : optional pre-processed datalogger dataframe
        n_segments    : number of segments to split into (ignored if time_marks)
        split_by      : "time" or "events" (ignored if time_marks)
        time_marks    : optional list of Absolute Timer cut points in seconds
        labels        : optional list of section names (len == len(marks)-1)
        noise_threshold: threshold for noise filtering in process_run
        mip_window     : tuple of (low, high) MIP range for process_run
        twodim_hist_args: dict of kwargs for 2D histogram plotting in process_run
        run_name      : base name for segment folders, default is "flight"; folder for each of the N segments will be created under this name, as [run_name]_seg1, ..., [run_name]_segN
        results_dir   : optional path to save segment folders; if None, data isn't saved
    """
    # 1) cut points on the Absolute Timer
    if time_marks is None:
        if flight_df is None:
            flight_df = Datalogger_Processing(datalogger_fp, show_plots=False,
                                              results_dir=results_dir).process()
        t = np.asarray(flight_df["Absolute Timer (S)"], dtype=float)
        t_lo, t_hi = float(np.nanmin(t)), float(np.nanmax(t))
        if split_by == "events":
            # quantiles of t -> equal event counts per segment
            marks = list(np.quantile(t, np.linspace(0.0, 1.0, n_segments + 1)))
        else:
            # evenly spaced seconds -> equal duration per segment
            marks = list(np.linspace(t_lo, t_hi, n_segments + 1))
        marks[-1] = t_hi + 1e-6      # split_by_time_marks uses [t0, t1); keep the final row
    else:
        marks = sorted(float(x) for x in time_marks)

    if labels is None:
        labels = [f"seg{k + 1}" for k in range(len(marks) - 1)]

    # 2) write the windowed files, then analyze each window
    segments = {}
    for sec in split_by_time_marks(datalogger_fp, scint_fps, marks, labels=labels):
        lbl = sec["label"]
        print(f"\n=== SEGMENT {lbl}: {sec['t_start']:.0f}-{sec['t_end']:.0f} s ===")

        seg_dir = os.path.join(results_dir, f"{run_name}_{lbl}") if results_dir is not None else None

        _, _, analysis = process_run(
            sec["datalogger"], sec["scints"],
            MPVs=MPVs,
            results_dir=seg_dir,
            save_all=True,
            noise_threshold=noise_threshold,
            mip_window=mip_window,
            twodim_hist_args=twodim_hist_args,
        )
        segments[lbl] = analysis

    return segments