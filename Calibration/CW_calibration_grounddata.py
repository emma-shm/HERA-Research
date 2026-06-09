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



# ============================ 14 hour room temp run =============================
roomtemp_datalogger_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/DataLogger/AHD002_roomtemp?.csv' # @param {type:"string"}
roomtemp_top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDleft/SDleft_AxLab_M_024_roomtemp.txt' # @param {type:"string"}
roomtemp_mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDmiddle/SDmiddle_AxLab_M_024_roomtemp.txt' # @param {type:"string"}
roomtemp_bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDright/SDright_AxLab_M_024_roomtemp.txt' # @param {type:"string"}

roomtemp_df = Datalogger_Processing(roomtemp_datalogger_fp, show_plots=False).process()

three_scintillator_roomtemp = CW_Processing([roomtemp_top_scint_fp, roomtemp_mid_scint_fp, roomtemp_bot_scint_fp], roomtemp_df)
analysis_roomtemp = CW_Analysis(three_scintillator_roomtemp, roomtemp_df)
analysis_roomtemp.rate_spectra_with_moyal(moyal_fit_ranges=[(46, 80), (46, 82), (44, 76)])



# ============================ Cs 137 run =============================
cs137_datalogger_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs 137/data logger/AHD001.csv' # @param {type:"string"}
cs137_top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs 137/left/leftAxLab_M_029.txt' # @param {type:"string"}
cs137_mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs 137/middle/middleAxLab_M_028.txt' # @param {type:"string"}
cs137_bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs 137/right/rightAxLab_M_029.txt'

cs_df = Datalogger_Processing(cs137_datalogger_fp).process()

three_scintillator_cs = CW_Processing([cs137_top_scint_fp, cs137_mid_scint_fp, cs137_bot_scint_fp], cs_df)
analysis_cs = CW_Analysis(three_scintillator_cs, cs_df)
analysis_cs.rate_spectra_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])

plot_density_heatmap_ampcal(analysis_cs, normalize_by_livetime=True)




# ============================ Cs 137 background run =============================
cs137background_datalogger_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs137_Background/AHD013.csv' # @param {type:"string"}
cs137background_top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs137_Background/left_AxLab_M_040.txt' # @param {type:"string"}
cs137background_mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs137_Background/middle_AxLab_M_039.txt' # @param {type:"string"}
cs137background_bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/Cs137_Background/right_AxLab_M_040.txt'

cs_df = Datalogger_Processing(cs137background_datalogger_fp).process()

three_scintillator_cs = CW_Processing([cs137background_top_scint_fp, cs137background_mid_scint_fp, cs137background_bot_scint_fp], cs_df)
analysis_cs = CW_Analysis(three_scintillator_cs, cs_df)
analysis_cs.rate_spectra_with_fixed_MPVs(MPVs=[56.78344621184912, 57.002885606912805, 54.6370444867040])

plot_density_heatmap_ampcal(analysis_cs, normalize_by_livetime=True)




# TEMPERATURE-VARYING RUNS

# ============================ Fridge =============================
fridge_datalogger_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/DataLogger/AHD003_fridge.csv' # @param {type:"string"}
fridge_top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDleft/SDleft_AxLab_M_024_fridge.txt' # @param {type:"string"}
fridge_mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDmiddle/SDmiddle_AxLab_M_024_fridge.txt' # @param {type:"string"}
fridge_bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDright/SDright_AxLab_M_025_fridge.txt' # @param {type:"string"}

fridge_df = Datalogger_Processing(fridge_datalogger_fp, show_plots=False).process()

sections_fridge = split_by_time_marks(
    fridge_datalogger_fp,
    [fridge_top_scint_fp, fridge_mid_scint_fp, fridge_bot_scint_fp],
    time_marks=[4900, 11000, 13000, 29397.77],
    labels=['fridge_warmup', 'fridge_mid', 'fridge_cold'],
)

section_fit_ranges = {
    # 'fridge_cold':   [(42, 72), (48, 76), (44, 72)],
    'fridge_warmup': [(41, 75), (48, 78), (42, 71)],
}

fridge_runs = {}
skip = {'fridge_mid', 'fridge_cold'}   # sections_fridge to leave out of the analysis

for sec in sections_fridge:
    if sec['label'] in skip:
        print(f"skipping '{sec['label']}'")
        continue
    print(f"\n=== {sec['label']} ===")
    df        = Datalogger_Processing(sec['datalogger'], show_plots=False).process()
    processor = CW_Processing(sec['scints'], df)
    analysis  = CW_Analysis(processor, df)
    analysis.rate_spectra_with_moyal(moyal_fit_ranges=section_fit_ranges[sec['label']])
    fridge_runs[sec['label']] = {'df': df, 'processor': processor, 'analysis': analysis}




# ============================ Freezer =============================
freezer_datalogger_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/DataLogger/AHD004_freezer.csv'
freezer_top_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDleft/SDleft_AxLab_M_025_freezer.txt'
freezer_mid_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDmiddle/SDmiddle_AxLab_M_025_freezer.txt'
freezer_bot_scint_fp = '/Users/emmamartignoni/Desktop/HERA-Research/Data/05292026/SDright/SDright_AxLab_M_026_freezer.txt'

freezer_df = Datalogger_Processing(freezer_datalogger_fp, show_plots=False).process()

sections_freezer = split_by_time_marks(
    freezer_datalogger_fp,
    [freezer_top_scint_fp, freezer_mid_scint_fp, freezer_bot_scint_fp],
    time_marks=[0, 2745, 10000],          # ← check that 1000; sorted → [1000, 4900, 10000]
    labels=['freezer_warmup', 'freezer_cold'],
)

section_fit_ranges = {
    # 'freezer_cold': [(41, 75), (48, 78), (42, 71)],
    'freezer_warmup': [(46, 76), (48, 78), (48, 76)],
}

freezer_runs = {}
skip = {'freezer_cold'}   # sections to leave out of the analysis

for sec in sections_freezer:                 # ← was `sections` (the fridge one)
    if sec['label'] in skip:
        print(f"skipping '{sec['label']}'")
        continue
    print(f"\n=== {sec['label']} ===")
    df        = Datalogger_Processing(sec['datalogger'], show_plots=False).process()
    processor = CW_Processing(sec['scints'], df)
    analysis  = CW_Analysis(processor, df)
    analysis.rate_spectra_with_moyal(moyal_fit_ranges=section_fit_ranges[sec['label']])
    freezer_runs[sec['label']] = {'df': df, 'processor': processor, 'analysis': analysis}




# ============================ Room temp (14-hour run, again) =============================

# split — fill time_marks from the overlap window above
sections_roomtemp = split_by_time_marks(
    roomtemp_datalogger_fp,
    [roomtemp_top_scint_fp, roomtemp_mid_scint_fp, roomtemp_bot_scint_fp],
    time_marks=[0, 5000, 8000, 12000, 48555],                          # ← fill from overlap
    labels=['roomtemp_warmup', 'roomtemp_cooldown1', 'roomtemp_cooldown2', 'roomtemp_settled'],   # ← match len(marks)-1
)

section_fit_ranges = {
    'roomtemp_warmup': [(44, 78), (46, 82), (44, 76)],
    'roomtemp_cooldown1': [(44, 78), (45, 80), (43, 72)], 
    'roomtemp_cooldown2': [(45, 79), (43, 80), (44, 77)], 
    'roomtemp_settled': [(46, 80), (46, 82), (44, 76)], 
}

roomtemp_runs = {}
skip = {'roomtemp_warmup'}                     # sections to leave out

for sec in sections_roomtemp:
    if sec['label'] in skip:
        print(f"skipping '{sec['label']}'")
        continue
    print(f"\n=== {sec['label']} ===")
    df        = Datalogger_Processing(sec['datalogger'], show_plots=False).process()
    processor = CW_Processing(sec['scints'], df)
    analysis  = CW_Analysis(processor, df)
    analysis.rate_spectra_with_moyal(moyal_fit_ranges=section_fit_ranges[sec['label']])
    roomtemp_runs[sec['label']] = {'df': df, 'processor': processor, 'analysis': analysis}







# ============================ Fitting =============================

# ──────────────────────────────────────────────────────────────────────────
# Time-binned global-mean-MPV vs temperature calibration
#
# Replaces the old 3-point fit (one global_mean_mpv per run). Each run's
# master_df is split into ~2-hour time bins; in each bin every scintillator's
# COINCIDENT SiPM peak (SiPM_mV_<top>_scint i) is re-fit with the same Moyal
# form used run-level, then averaged into a per-chunk MPV — identical estimator
# to global_mean_mpv, just on a slice. Each run reuses its OWN per-scint
# moyal_fit_ranges (already tuned), so no per-chunk range tuning is needed.
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt


def _moyal(x, mpv, eta, A):
    return A * np.exp(-0.5 * (((x - mpv) / eta) + np.exp(-(x - mpv) / eta)))


def _fit_peak_mpv(values_mV, fit_range, nb=30, min_n=60):
    """
    Fit one scintillator's coincident SiPM peak inside [lo, hi] mV.
    Returns (mpv, mpv_err, n_used) or (nan, nan, n_used) if the fit can't run.
    Counts-based linear-bin fit (peak location is scale-invariant, so this
    matches the run-level fitted MPV to within the comparison your flight
    `fit_bin_mpv` cell already prints).
    """
    lo, hi = fit_range
    x = np.asarray(values_mV, float)
    x = x[np.isfinite(x) & (x >= lo) & (x <= hi)]
    if x.size < min_n:
        return np.nan, np.nan, x.size

    counts, edges = np.histogram(x, bins=nb)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p0 = [centers[counts.argmax()], 15.0, counts.max()]
    bounds = ([lo, 1.0, 0.0], [hi, 60.0, np.inf])   # mpv pinned to the window
    try:
        popt, pcov = curve_fit(_moyal, centers, counts, p0=p0,
                               bounds=bounds, maxfev=20000)
        return popt[0], np.sqrt(np.diag(pcov))[0], x.size
    except (RuntimeError, ValueError):
        return np.nan, np.nan, x.size


def timebinned_mpv_points(analysis, bin_hours=2.0, nb=30, min_n=60,
                          pooled=False, verbose=True):
    """
    Build per-time-bin (T, global-mean MPV) points from one processed
    CW_Analysis run.

    Args:
        analysis  : a CW_Analysis instance that has already run
                    rate_spectra_with_moyal (so master_df + moyal_fit_ranges exist)
        bin_hours : width of each time bin in hours
        nb        : histogram bins per scint-fit
        min_n     : minimum coincident events in a (scint, bin) to attempt a fit
        pooled    : if True, pool all scints into ONE histogram per bin and fit
                    once (more events/fit, more robust on thin bins, but a
                    slightly different estimator than run-level global_mean_mpv).
                    If False (default), fit each scint and average → matches
                    how global_mean_mpv is defined.

    Returns:
        DataFrame with one row per usable time bin:
        [t_center, T_mean, T_std, MPV, MPV_err, N_events, n_scints_fit]
    """
    m = analysis.master_df
    top = analysis.processor.top_tag                       # e.g. 'CW123'
    n_scint = len(analysis.processor.fps)
    mv_cols = [f'SiPM_mV_{top}_scint{i}' for i in range(1, n_scint + 1)]

    # per-scint fit windows: reuse the run's tuned ranges; fall back to a
    # ±35% window around each scint's run-level MPV if ranges aren't set.
    if getattr(analysis, 'moyal_fit_ranges', None) is not None:
        ranges = list(analysis.moyal_fit_ranges)
    else:
        ranges = [(0.65 * analysis.mpv_per_scint[i], 1.35 * analysis.mpv_per_scint[i])
                  for i in range(1, n_scint + 1)]

    t = m['Absolute Timer (S)'].to_numpy(float)
    t0 = np.nanmin(t)
    width = bin_hours * 3600.0
    bin_idx = np.floor((t - t0) / width).astype(int)
    m = m.assign(_tbin=bin_idx)

    rows = []
    for b, g in m.groupby('_tbin', observed=True):
        T_mean = g['Temperature[C]'].mean()
        T_std = g['Temperature[C]'].std()
        t_center = g['Absolute Timer (S)'].mean()

        if pooled:
            pooled_vals = pd.concat([g[c] for c in mv_cols]).to_numpy(float)
            lo = float(np.mean([r[0] for r in ranges]))
            hi = float(np.mean([r[1] for r in ranges]))
            mpv, err, nused = _fit_peak_mpv(pooled_vals, (lo, hi), nb=nb, min_n=min_n)
            if not np.isfinite(mpv):
                continue
            rows.append((t_center, T_mean, T_std, mpv, err, nused, 1))
        else:
            scint_mpvs, scint_errs, n_used_total = [], [], 0
            for i in range(1, n_scint + 1):
                vals = g[f'SiPM_mV_{top}_scint{i}'].to_numpy(float)
                mpv_i, err_i, nused = _fit_peak_mpv(vals, ranges[i - 1], nb=nb, min_n=min_n)
                n_used_total += nused
                if np.isfinite(mpv_i):
                    scint_mpvs.append(mpv_i)
                    scint_errs.append(err_i)
            if len(scint_mpvs) == 0:
                continue
            mpv = float(np.mean(scint_mpvs))
            # mirror the existing y-error definition: spread across scints / sqrt(N)
            if len(scint_mpvs) >= 2:
                err = float(np.std(scint_mpvs, ddof=1) / np.sqrt(len(scint_mpvs)))
            else:
                err = float(scint_errs[0])
            rows.append((t_center, T_mean, T_std, mpv, err, n_used_total, len(scint_mpvs)))

    out = pd.DataFrame(rows, columns=['t_center', 'T_mean', 'T_std', 'MPV',
                                      'MPV_err', 'N_events', 'n_scints_fit'])
    if verbose:
        print(f"  {len(out)} usable {bin_hours:g}-hr bins "
              f"(events/bin: {out['N_events'].min()}–{out['N_events'].max()}, "
              f"T range {out['T_mean'].min():.1f} to {out['T_mean'].max():.1f} °C)")
    return out


# ── Build the dense point set across all three runs ──────────────────────────
runs   = [analysis_freezer, analysis_fridge, analysis_roomtemp]
names  = ['freezer', 'fridge', 'roomtemp']
colors = ['steelblue', 'mediumseagreen', 'indianred']

BIN_HOURS = 2.0   # ← tweak this; smaller = more points but fewer events/fit

pts = []
for a, nm in zip(runs, names):
    print(f"{nm}:")
    p = timebinned_mpv_points(a, bin_hours=BIN_HOURS, pooled=False)
    p['run'] = nm
    pts.append(p)
pts = pd.concat(pts, ignore_index=True)
print(f"\nTotal calibration points: {len(pts)}  (was 3)")

# ── Linear fit on the dense points ───────────────────────────────────────────
fit = linregress(pts['T_mean'], pts['MPV'])
print(f"global_mean_MPV(T) = {fit.slope:.4f}·T + {fit.intercept:.4f}   "
      f"(R² = {fit.rvalue**2:.3f}, n = {len(pts)})")

global_mean_mpv_at = lambda T: fit.slope * T + fit.intercept

def calibrate(scint_mVs, scint_mpv, T):
    gmpv = global_mean_mpv_at(T)
    return (scint_mVs - abs(gmpv - scint_mpv)) / gmpv

# ── Build one (T, MPV) point per SURVIVING section ───────────────────────────
# Skipped sections were never added to the *_runs dicts, so every entry here is
# a kept section. Labels are prefixed (fridge_/freezer_/roomtemp_) so merging is safe.
runs_dict = {**freezer_runs, **fridge_runs, **roomtemp_runs}

names = list(runs_dict.keys())                          # e.g. ['freezer_cold', 'fridge_warmup', 'roomtemp_settled']
runs  = [runs_dict[k]['analysis'] for k in names]

temps = [a.datalogger_df['Temperature[C]'].mean() for a in runs]
mpvs  = [a.global_mean_mpv for a in runs]

# color each point by which run-family it came from
family_colors = {'freezer': 'steelblue', 'fridge': 'mediumseagreen', 'roomtemp': 'indianred'}
colors = [family_colors[k.split('_')[0]] for k in names]

# ── Linear fit ───────────────────────────────────────────────────────────────
fit = linregress(temps, mpvs)
print(f"global_mean_MPV(T) = {fit.slope:.4f}·T + {fit.intercept:.4f}   "
      f"(R² = {fit.rvalue**2:.3f}, n = {len(runs)})")

global_mean_mpv_at = lambda T: fit.slope * T + fit.intercept

def calibrate(scint_mVs, scint_mpv, T):
    gmpv = global_mean_mpv_at(T)
    return (scint_mVs - abs(gmpv - scint_mpv)) / gmpv

# per-row predicted global mean MPV on each section's master_df (downstream use)
for a in runs:
    a.master_df['global_mean_MPV_at_T'] = global_mean_mpv_at(a.master_df['Temperature[C]'])

# ── Error bars ───────────────────────────────────────────────────────────────
# y-error: spread of the per-scint MPVs about their mean (standard error)
mpv_err  = [np.std(list(a.mpv_per_scint.values()), ddof=1) / np.sqrt(len(a.mpv_per_scint)) for a in runs]
# x-error: temperature drift over the section
temp_err = [a.datalogger_df['Temperature[C]'].std() for a in runs]

# ── Plot ─────────────────────────────────────────────────────────────────────
T_line = np.linspace(min(temps), max(temps), 100)
plt.figure(figsize=(7, 5))
for T, m, xe, ye, name, c in zip(temps, mpvs, temp_err, mpv_err, names, colors):
    plt.errorbar(T, m, yerr=ye, xerr=xe, fmt='o', color=c, capsize=4, zorder=3, label=name)
# plt.plot(T_line, global_mean_mpv_at(T_line), 'k--',
#          label=f'Fit: {fit.slope:.3f}·T + {fit.intercept:.2f}  (R² = {fit.rvalue**2:.3f})')
plt.xlabel('Temperature [°C]  (x-error = temperature std over section)', fontsize=11)
plt.ylabel('Global Mean MPV [mV]  (y-error = std error across scints)', fontsize=11)
plt.title('Global Mean MPV vs. Temperature  (per-section points)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

