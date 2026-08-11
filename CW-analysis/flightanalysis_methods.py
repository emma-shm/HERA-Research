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

def split_by_time_marks(datalogger_fp, scint_fps, time_marks, labels=None):
    """
    Split a datalogger + scintillator set into the sections BETWEEN a list of
    Absolute Timer marks (seconds). N marks → N-1 sections: section k spans
    [time_marks[k], time_marks[k+1]). To keep the run start/end as their own
    sections, include 0 and the max Absolute Timer in time_marks.

    Args:
        datalogger_fp : path to the datalogger CSV
        scint_fps     : list of scintillator TXT paths
        time_marks    : list of Absolute Timer cut points in seconds
        labels        : optional list of section names (len == len(marks)-1);
                        defaults to seg1, seg2, ...

    Returns:
        list of dicts, one per section:
            {'label', 't_start', 't_end', 'datalogger', 'scints': [...]}
    """
    marks = sorted(float(t) for t in time_marks)
    if len(marks) < 2:
        raise ValueError("Need at least 2 time marks to define a section.")
    if labels is not None and len(labels) != len(marks) - 1:
        raise ValueError(f"Got {len(labels)} labels for {len(marks) - 1} sections.")

    # Build Absolute Timer with the existing class.
    dl = Datalogger_Processing(datalogger_fp, show_plots=False)
    dl.process()
    df = dl.df.copy()
    df['Timer[S]'] = df['Absolute Timer (S)']        
    df = df.drop(columns=[c for c in ['Absolute Timer (S)', 'Timer_rel'] if c in df.columns]) # drop columns

    out_dir = os.path.dirname(datalogger_fp)
    base    = os.path.splitext(os.path.basename(datalogger_fp))[0]

    # Pre-read each scintillator once (3-line header + body), reuse across sections.
    cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]','SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
    scint_data = []
    for fp in scint_fps: # loop over scintillator files, read them in, and store the header and dataframe for later slicing
        with open(fp) as f:
            header = [next(f) for _ in range(3)] # next(f) reads the next line from the file object f; this reads the first 3 lines of the file and stores them in a list called header
        sdf = pd.read_csv(fp, sep='\t', comment='#', header=None, skiprows=3, names=cols, engine='python')
        sbase = os.path.splitext(os.path.basename(fp))[0]
        scint_data.append((sbase, header, sdf))

    sections = []
    for k in range(len(marks) - 1): # loop over the time marks to define the sections
        t0, t1 = marks[k], marks[k + 1] # take the k-th and (k+1)-th time marks to define the start and end of the section
        tag = labels[k] if labels is not None else f"seg{k + 1}"
        print(f"Section '{tag}': {t0:.0f}-{t1:.0f} s")

        # Datalogger slice [t0, t1).
        dl_out  = os.path.join(out_dir, f"{base}_{tag}.csv")
        dl_mask = (df['Timer[S]'] >= t0) & (df['Timer[S]'] < t1)
        df[dl_mask].to_csv(dl_out, index=False)

        # Scintillator slices on the same window, keeping the 3-line header + tab format.
        scint_out = []
        for sbase, header, sdf in scint_data:
            out    = os.path.join(out_dir, f"{sbase}_{tag}.txt")
            s_mask = (sdf['Time[s]'] >= t0) & (sdf['Time[s]'] < t1)
            with open(out, 'w') as f:
                f.writelines(header)
                sdf[s_mask].to_csv(f, sep='\t', header=False, index=False)
            scint_out.append(out)

        sections.append({'label': tag, 't_start': t0, 't_end': t1,
                         'datalogger': dl_out, 'scints': scint_out})

    return sections

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

    Returns:
        {label: Detector_Analysis}. Each analysis still carries .master_df,
        .processor, .datalogger_df and .mpv_per_scint.
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