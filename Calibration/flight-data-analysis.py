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
    df['Timer[S]'] = df['Absolute Timer (S)']        # flatten resets so each section reprocesses cleanly
    df = df.drop(columns=[c for c in ['Absolute Timer (S)', 'Timer_rel'] if c in df.columns])

    out_dir = os.path.dirname(datalogger_fp)
    base    = os.path.splitext(os.path.basename(datalogger_fp))[0]

    # Pre-read each scintillator once (3-line header + body), reuse across sections.
    cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]','SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
    scint_data = []
    for fp in scint_fps:
        with open(fp) as f:
            header = [next(f) for _ in range(3)]
        sdf = pd.read_csv(fp, sep='\t', comment='#', header=None, skiprows=3, names=cols, engine='python')
        sbase = os.path.splitext(os.path.basename(fp))[0]
        scint_data.append((sbase, header, sdf))

    sections = []
    for k in range(len(marks) - 1):
        t0, t1 = marks[k], marks[k + 1]
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
    normalize_by_livetime=True,
    noise_threshold=0.1,
    mip_window=(0.8, 1.2),
    show_heatmap=True,
    run_name="flight",
):
    # 1) cut points on the Absolute Timer
    if time_marks is None:
        if flight_df is None:
            flight_df = Datalogger_Processing(datalogger_fp, show_plots=False).process()
        t = np.asarray(flight_df["Absolute Timer (S)"], dtype=float)
        t_lo, t_hi = float(np.nanmin(t)), float(np.nanmax(t))
        if split_by == "events":
            marks = list(np.quantile(t, np.linspace(0.0, 1.0, n_segments + 1)))
        else:  # "time"
            marks = list(np.linspace(t_lo, t_hi, n_segments + 1))
        marks[-1] = t_hi + 1e-6
    else:
        marks = sorted(float(x) for x in time_marks)

    n_seg = len(marks) - 1
    if labels is None:
        labels = [f"seg{k + 1}" for k in range(n_seg)]

    # DEBUG: marks + datalogger row count per window
    print(f"[DEBUG] split_by={split_by!r}  n_seg={n_seg}")
    print(f"[DEBUG] marks = {[round(m, 1) for m in marks]}")
    _tt = np.asarray(flight_df["Absolute Timer (S)"], dtype=float) if flight_df is not None else None
    if _tt is not None:
        for k in range(n_seg):
            _c = int(((_tt >= marks[k]) & (_tt < marks[k + 1])).sum())
            print(f"[DEBUG]   {labels[k]}: {marks[k]:.0f}-{marks[k+1]:.0f} s  datalogger rows = {_c}")

    # 2) write windowed datalogger + scint files for each segment
    sections = split_by_time_marks(datalogger_fp, scint_fps, marks, labels=labels)

    # DEBUG: confirm split_by_time_marks actually produced n_seg sections
    print(f"[DEBUG] split_by_time_marks returned {len(sections)} section(s): "
          f"{[s['label'] for s in sections]}")

    hm_params = inspect.signature(plot_density_heatmap_ampcal).parameters

    # 3) run the fixed-MPV pipeline + ampcal heatmap per segment
    results = {}
    for sec in sections:
        lbl = sec["label"]
        print("\n" + "=" * 78)
        print(f"SEGMENT {lbl}:  {sec['t_start']:.0f}–{sec['t_end']:.0f} s")
        print("=" * 78)

        set_results_dir(f"{run_name}_{lbl}")

        # DEBUG: scint coverage in this window (rows actually written to each seg file)
        _cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]',
                 'SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
        for _fp in sec["scints"]:
            _sdf = pd.read_csv(_fp, sep='\t', comment='#', header=None,
                               skiprows=3, names=_cols, engine='python')
            _ts = _sdf['Time[s]']
            print(f"[DEBUG]   scint {os.path.basename(_fp):40s} "
                  f"rows={len(_sdf):>6}  Time[s] "
                  f"{(_ts.min() if len(_sdf) else float('nan')):.0f}-"
                  f"{(_ts.max() if len(_sdf) else float('nan')):.0f}")

        seg_df = Datalogger_Processing(sec["datalogger"], show_plots=True).process()
        print(f"[DEBUG]   seg_df rows = {len(seg_df)}")  # DEBUG

        proc   = CW_Processing(sec["scints"], seg_df)
        ana    = CW_Analysis(proc, seg_df)
        ana.rate_spectra_with_fixed_MPVs(
            MPVs=MPVs, noise_threshold=noise_threshold, mip_window=mip_window
        )

        # DEBUG: how many points the heatmap will actually plot
        _md = ana.master_df
        _avg = _md["SiPM_scints_avg_MIP_ampcal"]
        print(f"[DEBUG]   master_df rows={len(_md)}  "
              f"ampcal-avg finite={int(_avg.notna().sum())}  "
              f">= noise_thr({ana.noise_threshold})={int((_avg >= ana.noise_threshold).sum())}")
        for i in range(1, len(proc.fps) + 1):
            print(f"[DEBUG]   total_livetime_scint{i}_s = "
                  f"{getattr(proc, f'total_livetime_scint{i}_s'):.2f}")

        if show_heatmap:
            kw = {"normalize_by_livetime": normalize_by_livetime}
            for cand in ("label", "title_suffix"):
                if cand in hm_params:
                    kw[cand] = lbl
                    break
            plot_density_heatmap_ampcal(ana, **kw)

        results[lbl] = {
            "t_start": sec["t_start"], "t_end": sec["t_end"],
            "datalogger_df": seg_df, "processor": proc, "analysis": ana,
        }

    return results


def analyze_flight_window(datalogger_fp, scint_fps, MPVs, t_start, t_end,
                          label="window", normalize_by_livetime=True,
                          noise_threshold=0.1, mip_window=(0.8, 1.2),
                          show_heatmap=True):
    """
    Run the fixed-MPV pipeline + ampcal heatmap on a SINGLE Absolute-Timer
    window [t_start, t_end] (seconds). Returns that segment's analysis object.
    """
    results = analyze_flight_in_segments(
        datalogger_fp, scint_fps, MPVs,
        time_marks=[float(t_start), float(t_end) + 1e-6],   # [t0, t1) is half-open; keep the last row
        labels=[label],
        normalize_by_livetime=normalize_by_livetime,
        noise_threshold=noise_threshold,
        mip_window=mip_window,
        show_heatmap=show_heatmap,
    )
    return results[label]["analysis"]




# ========================= Isolate flight in full files =========================

set_results_dir("may31flight_overview")

datalogger_csv_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/May_31st_Flight/AHD011 copy.csv' # @param {type:"string"}
top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/May_31st_Flight/left_AxLab_M_038.txt' # @param {type:"string"}
mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/May_31st_Flight/middle_AxLab_M_037 copy.txt' # @param {type:"string"}
bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/May_31st_Flight/right_AxLab_M_038 copy.txt' # @param {type:"string"}

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
)

# grab any segment's analysis object for further work (Tcal columns, etc.)
analysis_seg2 = segments["seg2"]["analysis"]






