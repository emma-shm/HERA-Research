# `CW-analysis/`
Scripts for calibrating and analyzing HERA's typical three-scintillator CW detectors, using calibration for particle identification

## Usage

Every run is the same three steps — process the datalogger, align the
scintillators, analyze — and differs only in whether the MPVs get fit from this
dataset or supplied from a previous one. Update the paths to your own data; the
examples assume three scintillators (top, middle, bottom) plus a datalogger.

```python
datalogger_fp = f"{DATA_DIR}/<date>/DataLogger/<datalogger_file>.csv"
top_scint_fp  = f"{DATA_DIR}/<date>/<top_detector_directory>/<top_detector_file>.txt"
mid_scint_fp  = f"{DATA_DIR}/<date>/<middle_detector_directory>/<middle_detector_file>.txt"
bot_scint_fp  = f"{DATA_DIR}/<date>/<bottom_detector_directory>/<bottom_detector_file>.txt"
```

### Option A — the three classes, step by step

Use this when you want a handle on the intermediate objects, or when a run needs
something non-standard (a custom `name=` on the datalogger figure, a different
`results_dir` per class, inspecting `proc.aligned_scint1` before analyzing).

```python
from calibration_methods import (
    Datalogger_Processing, Scintillators_Processing, Detector_Analysis
)

# 1) datalogger → continuous Absolute Timer (handles timer resets)
df   = Datalogger_Processing(datalogger_fp, show_plots=False).process()

# 2) align each scintillator to the datalogger (deadtime-corrected livetime)
proc = Scintillators_Processing([top_scint_fp, mid_scint_fp, bot_scint_fp], df, show_plots=False)

# 3) calibrate + plot — pick ONE mode:
analysis = Detector_Analysis(proc, df, results_dir=os.path.join(RESULTS_DIR, "<output_directory>"))
analysis.calibrate_and_analyze_grounddata(moyal_fit_ranges=[(46, 80), (46, 82), (44, 76)])  # CALIBRATION RUN
# analysis.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78, 57.00, 54.64])              # ANALYSIS RUN
```

### Option B — one call

Same pipeline, same outputs. Which terminal method runs is decided by which
argument you pass — exactly one of `moyal_fit_ranges` or `MPVs`.

```python
from calibration_methods import process_run

# CALIBRATION RUN — fit a Moyal per scintillator to extract the MPVs
df, proc, analysis = process_run(
    datalogger_fp, [top_scint_fp, mid_scint_fp, bot_scint_fp],
    moyal_fit_ranges=[(46, 80), (46, 82), (44, 76)],
    results_dir=os.path.join(RESULTS_DIR, "<output_directory>"),
)

# ANALYSIS RUN — apply MPVs from a previous calibration, no fitting
df, proc, analysis = process_run(
    datalogger_fp, [top_scint_fp, mid_scint_fp, bot_scint_fp],
    MPVs=[56.78, 57.00, 54.64],
    results_dir=os.path.join(RESULTS_DIR, "<output_directory>"),
    twodim_hist_args={'col': 'MIP', 'cbar_max': 0.0005},
)
```

### Calibration vs. analysis runs

- **Calibration run** — the MPVs aren't known yet, so a Moyal is fit to each
  scintillator's coincident spectrum. Requires `moyal_fit_ranges`.
- **Analysis run** — the MPVs are already known from a prior calibration of the
  *same* scintillators, so no fitting happens. Requires `MPVs`.

### Choosing `moyal_fit_ranges`

Determined by guess-and-check:

1. Start with reasonable lower/upper mV bounds for each scintillator.
2. Run the pipeline.
3. Inspect the generated rate spectrum, `mip_normalized.png`.
4. Adjust the ranges so they isolate the MIP peak more accurately.
5. Update `moyal_fit_ranges` and rerun.

Each tuple is `(lower_bound, upper_bound)` for the top, middle, and bottom
scintillators respectively. Detector response varies between datasets, so there
is no universal set of ranges — pick them separately for each calibration run.

### Outputs

Results live on the returned `analysis`: `analysis.master_df` (the calibrated
event dataframe) and `analysis.mpv_per_scint` (the MPVs used). Plots and
`calibration_summary.csv` are written to `results_dir`; with no `results_dir`
they're shown interactively instead. In Option B, `results_dir` reaches
`Detector_Analysis` only — pass `save_all=True` to also save the datalogger and
per-scintillator SiPM figures.

**The scripts in this directory show both patterns in context:**

- **`three_scint_calibration_grounddata.py`** — bench datasets (room-temp plus
  temperature-varying fridge/freezer runs), extracting reference MPVs via Moyal
  fits. Start here for the fitting workflow.
- **`three_scint_analysis_Eu.py`** — a background calibration run whose fitted
  MPVs are chained straight into the source run's `MPVs=` argument.
- **`three_scint_analysis_Cs137.py`** — source and background runs, both fixed-MPV.
- **`May31st_flight_analysis.py`** — ground-calibration MPVs applied to the
  balloon run, split into sections with the `flightanalysis_methods.py` helpers.


Either way, results live on the returned `analysis`: `analysis.master_df`
(calibrated event dataframe) and `analysis.mpv_per_scint` (MPVs used). Plots and
`calibration_summary.csv` are written to `results_dir`; with no `results_dir`
they're shown interactively instead.


## `calibration_methods.py`

The core processing and calibration module for the CosmicWatch (CW) detectors.
It ingests the raw datalogger CSV and the per-scintillator TXT files, aligns
them on a common timeline, calibrates the SiPM amplitude response, and produces
the rate spectra and density heatmaps used in the analysis. The pipeline is
built around three classes plus a top-level helper, process_run(), which runs
the methods in order.

**Classes**

- **`Datalogger_Processing`** — reads the datalogger CSV and plots the raw
  coincidence-event counts, pressure, and temperature vs. time. Detects timer
  resets (backward jumps in the timer column) and stitches the individual runs
  into a single continuous `Absolute Timer (S)` column so multi-run files
  reprocess cleanly.

- **`Scintillators_Processing`** — reads each scintillator file, applies a deadtime
  correction to compute per-event livetime, and aligns every scintillator to
  the datalogger independently via nearest-timestamp `merge_asof`. Handles
  arbitrary N-fold coincidence naming (CW12, CW123, …), tags coincidence events
  per scintillator, tracks per-scint and total deadtime-corrected livetime, and
  plots the SiPM voltage distributions (per-scint and combined).

- **`Detector_Analysis`** — assembles a master dataframe anchored on the datalogger
  timeline. Normalizes each scintillator's SiPM amplitude to MIP units using
  per-scintillator MPVs (either from Moyal fits or supplied as fixed values),
  applies an amplitude calibration that shifts each channel to a global-mean
  MPV, and derives cross-scintillator mean/std columns. Produces the calibrated
  rate spectra and the 2D density heatmaps, with all rates livetime-normalized
  so runs of different duration are directly comparable. Two top-level entry
  points, depending on whether MPVs need to be fit or are already known:
  `calibrate_and_analyze_grounddata(moyal_fit_ranges)` fits a Moyal per
  scintillator to extract the MPV, while
  `analyze_calibrated_data_with_fixed_MPVs(MPVs, ...)` uses externally supplied
  MPVs and skips fitting entirely.

**Helper functions, that CAN be imported and used in other scripts if desired**

- **`process_run(datalogger_fp, scint_fps, moyal_fit_ranges=None, MPVs=None, ...)`**
  — one-call entry point that chains `Datalogger_Processing` → `Scintillators_Processing`
  → `Detector_Analysis` and dispatches to the right terminal method. Pass either
  `moyal_fit_ranges` (calibration run) or `MPVs` (analysis run) — exactly one, not both.
  `results_dir` goes to `Detector_Analysis`; add `save_all=True` to also save the
  datalogger and per-scintillator SiPM figures. Extra keywords (`noise_threshold`,
  `twodim_hist_args`, and `mip_window` on the fixed-MPV path only) are forwarded to
  whichever method is dispatched. Returns `(datalogger_df, processor, analysis)`.
- **`PlotOutput`** — mixin inherited by all three classes above, holding the
  save/show policy in one place. `_init_output()` reads `results_dir` and
  `show_plots` into the `save_plots` / `show_plots` flags (see the table in
  [Outputs](#outputs)); `_finish_mpl()` / `_finish_plotly()` end every figure
  by saving, showing, and closing according to those flags; `_save_table()`
  and `_save_summary()` write the CSVs.

**Usage:**

One call runs the whole thing — it chains the three classes together and
produces all the calibration plots.

**Inputs:** a datalogger CSV and a list of per-scintillator TXT files
(ordered `scint1 … scintN`).


## `three_scint_calibration_grounddata.py`

Bench/ground calibration example. Runs the lab datasets through the pipeline to
establish the reference MPVs and the temperature dependence of the detector gain
on background calibration runs. Includes:

- **Room-temperature run** — a 14-hour run fit with Moyal (`calibrate_and_analyze_grounddata`)
  to extract the baseline per-scintillator MPVs.
- **Temperature-varying runs** (fridge, freezer, room temp) — each long run is
  cut into temperature sections with `split_by_time_marks`, and each kept
  section is Moyal-fit with its own tuned fit ranges (a `skip` set leaves out
  sections that aren't used).

It then builds the **gain-vs-temperature calibration**. `timebinned_mpv_points`
splits each run's `master_df` into ~2-hour bins and re-fits every scintillator's
coincident MIP peak per bin (`_fit_peak_mpv` / `_moyal`), averaging to a per-bin
MPV. Those points across all three runs are pooled and a linear fit gives
`global_mean_MPV(T) = slope·T + intercept`, exposed as `global_mean_mpv_at(T)`
and a `calibrate(scint_mVs, scint_mpv, T)` helper, plus an MPV-vs-temperature
plot.

**Output used downstream:** the per-scint MPVs and the `MPV(T)` fit, which feed
the flight analysis.

## three_scint_analysis_Cs137.py

Applying the calibration methods to radioactive source testing.

- **Cs-137 source + Cs-137 background runs** — processed with fixed MPVs
  (`analyze_calibrated_data_with_fixed_MPVs`)


## `May31st_flight_analysis.py`

Flight data driver (May 31st flight). Applies the ground-calibration MPVs to the
balloon flight and analyzes the run in segments. Uses helpers defined in `flightanalysis_methods.py`

The main block loads the flight datalogger and three scintillator files,
processes the full run, trims it to the flight window, plots altitude vs. time
as a sanity check (and prints max altitude before/after trimming to confirm the
trim didn't clip the peak), then splits the flight into 4 time segments and runs
the fixed-MPV analysis using the MPVs from `three_scint_calibration_grounddata.py`.

## flightanalysis_methods.py
Defines a few flight-specific helpers on top of `calibration_methods`:

- **`trim_to_flight(...)`** — drops the pre-launch warmup (`lead_minutes`) and
  cuts the run where altitude returns to ground after apogee, writing trimmed
  datalogger + scintillator files.
- **`split_by_time_marks(...)`** — slices a run into sections between Absolute
  Timer marks (seconds).
- **`analyze_flight_in_segments(...)`** — splits the flight into N segments (by
  time or by event count), then runs the fixed-MPV pipeline plus an
  amplitude-calibrated heatmap on each, returning a dict of per-segment analysis
  objects.
- **`analyze_flight_window(...)`** — convenience wrapper for a single
  `[t_start, t_end]` window.

## `CW_calibration_grounddata.ipynb`
NOTE: NOTEBOOK OUT OF DATE
All the analysis had originally mean done in one Jupyter notebook, but as it ended up being too large, I split the analsysis into separate scripts: calibration_methods.py has classes and helpers, that are imported into CW_calibration_grounddata.py to do calibration with the ground data, and then results are used in/applied to flight data in flight-data-analysis.pyThere are two equivalent ways to run a dataset: instantiate the three classes
yourself, or hand everything to `process_run` in one call. They do the same work

---

## Analysis notes

A few of the core techniques used across the notebooks:

- **MPV extraction via Moyal/Landau fit** — the energy deposited by a minimum-ionizing particle in a thin scintillator follows a Landau-like distribution; fitting it gives a stable most-probable value for gain calibration rather than relying on the mean.
- **Deadtime & livetime correction** — the measured rate is corrected for the time the detector spends processing each event so that flux estimates reflect true livetime.
- **Coincidence rate** — requiring simultaneous hits in stacked detectors suppresses noise and singles, isolating the through-going cosmic-ray/muon component.
- **Temperature-dependent calibration** — SiPM gain drifts with temperature, which matters over a balloon ascent; the calibration accounts for this so in-flight ADC values map correctly to energy.
- **Altitude profile** — plotting corrected coincidence rate vs. altitude recovers the Regener–Pfotzer curve.

---

## Citation

If you use this code or reference this work, please cite the associated thesis:

> Martignoni, E. *High-Altitude Engineering and Analysis for Cosmic Ray Tomography.* M.S. Thesis, Department of Mechanical Engineering, Drexel University, 2026.

---

## Contact

Maintained by [@emma-shm](https://github.com/emma-shm). Questions, issues, or contributions are welcome via the repo's [Issues](https://github.com/emma-shm/HERA-Research/issues) page.