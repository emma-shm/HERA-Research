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
    noise_threshold=0.1,
    mip_window=(0.8, 1.2),
    show_heatmap=True,
    cbar_max=None,
    run_name="flight",
    results_dir=None,
):
    """
    Split a flight datalogger + scintillator set into segments and run the
    fixed-MPV analysis pipeline (analyze_calibrated_data_with_fixed_MPVs) on
    each segment independently, saving every segment's outputs to its own
    results_dir instead of showing them interactively.

    Segmentation:
        If time_marks is given, those are used directly as the Absolute
        Timer cut points (n_seg = len(time_marks) - 1). Otherwise, marks are
        auto-generated: n_segments+1 evenly-spaced points either in time
        ("time") or in event quantiles ("events").

    Per-segment analysis:
        Each segment gets its own Datalogger_Processing → Scintillators_Processing
        → Detector_Analysis chain, run with the externally supplied MPVs (no
        Moyal fit per segment — see analyze_calibrated_data_with_fixed_MPVs).
        two_dimensional_histograms always runs as part of that pipeline; when
        show_heatmap=True, it's additionally pointed at the amplitude-calibrated
        MIP columns (col='MIP_ampcal') so the ampcal heatmap is produced for
        each segment, with cbar_max forwarded if given.

    Args:
        datalogger_fp : path to the full-flight datalogger CSV
        scint_fps     : list of full-flight scintillator TXT paths
        MPVs          : list of per-scint MPVs in mV (from a prior ground/temp
                        calibration), ordered scint1..scintN — passed straight
                        through to analyze_calibrated_data_with_fixed_MPVs
        flight_df     : optional pre-processed datalogger dataframe (with
                        Absolute Timer (S) already built); if None and
                        time_marks is also None, it's built here so the
                        auto-segmentation has something to slice on
        n_segments    : number of segments to auto-generate when time_marks
                        is not given
        split_by      : "time" (evenly spaced in Absolute Timer) or "events"
                        (evenly spaced in event-counts, meaning ); ignored if
                        time_marks is given
        time_marks    : explicit list of Absolute Timer cut points in seconds;
                        overrides n_segments/split_by entirely
        labels        : optional list of segment names (len == n_seg);
                        defaults to seg1, seg2, ...
        noise_threshold : MIP threshold separating noise from signal; forwarded
                        to analyze_calibrated_data_with_fixed_MPVs
        mip_window    : (lo, hi) MIP zoom window for the std heatmaps, since
                        there's no Moyal fit here to auto-derive one from
        show_heatmap  : if True, run two_dimensional_histograms on the
                        amplitude-calibrated MIP columns (col='MIP_ampcal') for
                        each segment; if False, two_dimensional_histograms still
                        runs (it's always part of the pipeline) but on the
                        default uncalibrated MIP columns instead
        cbar_max      : optional fixed colorbar upper bound, forwarded to
                        two_dimensional_histograms; only used when
                        show_heatmap=True
        run_name      : prefix used to build each segment's results_dir,
                        e.g. "flight_seg1", "flight_seg2",
        results_dir   : optional base directory under which each segment's
                            subfolder is created, as os.path.join(results_dir,
                            f"{run_name}_{label}"). If None, each segment's folder
                            is created relative to the current working directory
                            instead (old behavior: just f"{run_name}_{label}").

    Returns:
        dict keyed by segment label, each value:
            {'t_start', 't_end', 'datalogger_df', 'processor', 'analysis'}
        where 'analysis' is that segment's Detector_Analysis instance (so
        master_df, mpv_per_scint, etc. are all still accessible afterward).
    """
    # 1) cut points on the Absolute Timer
    if time_marks is None:
        # if there are no time cuts/segments, and if the Datalogger_Processing class hasn't been run yet, run the class to build the absolute timer
        if flight_df is None:
            flight_df = Datalogger_Processing(datalogger_fp, show_plots=False).process()
        t = np.asarray(flight_df["Absolute Timer (S)"], dtype=float)
        t_lo, t_hi = float(np.nanmin(t)), float(np.nanmax(t))
        # can either split by time (evenly spaced in seconds) or by events (evenly spaced in event count quantiles, meaning each segment has roughly the same number of events)
        if split_by == "events": # if splitting by events, use np.quantile to get the cut points for the segments
            marks = list(np.quantile(t, np.linspace(0.0, 1.0, n_segments + 1))) # np.linspace(0.0, 1.0, n_segments + 1) returns an array of evenly spaced
                                                                                    # fractions/percentiles (e.g. 0, 0.25, 0.5, 0.75, 1.0 for n_segments=4);
                                                                                    # np.quantile(t, ...) then returns the actual TIMER VALUES
                                                                                    # at each of those percentiles — i.e. the cut points that divide t's
                                                                                    # EVENTS into equal-count segments, rather than equal-duration ones
        else:  # "time"
            marks = list(np.linspace(t_lo, t_hi, n_segments + 1)) # if splitting by time, use np.linspace to get the cut points for the segments
        marks[-1] = t_hi + 1e-6        # split_by_time_marks uses [t0, t1); keep the final row
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

    # 2) write windowed datalogger + scint files for each segment (tested helper)
    sections = split_by_time_marks(datalogger_fp, scint_fps, marks, labels=labels)

    # DEBUG: confirm split_by_time_marks actually produced n_seg sections
    print(f"[DEBUG] split_by_time_marks returned {len(sections)} section(s): "
          f"{[s['label'] for s in sections]}")

    # 3) run the fixed-MPV pipeline per segment, saving to its own results_dir
    results = {}
    for sec in sections:
        lbl = sec["label"]
        print("\n" + "=" * 78)
        print(f"SEGMENT {lbl}:  {sec['t_start']:.0f}–{sec['t_end']:.0f} s")
        print("=" * 78)

        # instance-level results_dir (passed to Detector_Analysis below) replaces
        # the old global set_results_dir() call — each segment gets its own folder
        seg_results_dir = (os.path.join(results_dir, f"{run_name}_{lbl}")
                            if results_dir is not None
                            else f"{run_name}_{lbl}")

        # DEBUG: scint coverage in this window (rows actually written to each seg file)
        _cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]', 'SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
        for _fp in sec["scints"]:
            _sdf = pd.read_csv(_fp, sep='\t', comment='#', header=None,
                               skiprows=3, names=_cols, engine='python')
            _ts = _sdf['Time[s]']
            print(f"[DEBUG]   scint {os.path.basename(_fp):40s} "
                  f"rows={len(_sdf):>6}  Time[s] "
                  f"{(_ts.min() if len(_sdf) else float('nan')):.0f}-"
                  f"{(_ts.max() if len(_sdf) else float('nan')):.0f}")

        seg_df = Datalogger_Processing(sec["datalogger"], show_plots=True).process()
        print(f"[DEBUG]   seg_df rows = {len(seg_df)}")

        proc = Scintillators_Processing(sec["scints"], seg_df, results_dir=seg_results_dir)
        ana  = Detector_Analysis(proc, seg_df, results_dir=seg_results_dir)

        # Build the two_dimensional_histograms args: ampcal columns when
        # show_heatmap=True (mirrors what the old standalone
        # plot_density_heatmap_ampcal call used to do), plain defaults otherwise.
        # two_dimensional_histograms always runs as part of the pipeline either way.
        twodim_hist_args = {'col': 'MIP_ampcal'} if show_heatmap else None
        if show_heatmap and cbar_max is not None: # add cbar_max to two dimensional plotting dictionary
            twodim_hist_args['cbar_max'] = cbar_max

        ana.analyze_calibrated_data_with_fixed_MPVs(
            MPVs=MPVs,
            noise_threshold=noise_threshold,
            mip_window=mip_window,
            twodim_hist_args=twodim_hist_args,
        )

        # DEBUG: how many points the ampcal heatmap actually plotted
        _md = ana.master_df
        _avg = _md["SiPM_scints_avg_MIP_ampcal"]
        print(f"[DEBUG]   master_df rows={len(_md)}  "
              f"ampcal-avg finite={int(_avg.notna().sum())}  "
              f">= noise_thr({ana.noise_threshold})={int((_avg >= ana.noise_threshold).sum())}")
        for i in range(1, len(proc.fps) + 1):
            print(f"[DEBUG]   total_livetime_scint{i}_s = "
                  f"{getattr(proc, f'total_livetime_scint{i}_s'):.2f}")

        results[lbl] = {
            "t_start": sec["t_start"], "t_end": sec["t_end"],
            "datalogger_df": seg_df, "processor": proc, "analysis": ana,
        }

    return results