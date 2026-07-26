# `CW-analysis/`
Scripts for calibrating and analyzing HERA's typical three-scintillator CW detectors, using calibration for particle identification

## `CW_calibration_grounddata.ipynb`
NOTE: NOTEBOOK OUT OF DATE
All the analysis had originally mean done in one Jupyter notebook, but as it ended up being too large, I split the analsysis into separate scripts: calibration_methods.py has classes and helpers, that are imported into CW_calibration_grounddata.py to do calibration with the ground data, and then results are used in/applied to flight data in flight-data-analysis.py


## `calibration_methods.py`

The core processing and calibration module for the CosmicWatch (CW) detectors.
It ingests the raw datalogger CSV and the per-scintillator TXT files, aligns
them on a common timeline, calibrates the SiPM amplitude response, and produces
the rate spectra and density heatmaps used in the analysis. The pipeline is
built around three classes plus a set of top-level helpers.

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

- **`processing_pipeline(datalogger, scintillators, Moyal_fit_ranges=None, MPVs=None, ...)`**
  — one-call entry point that chains `Datalogger_Processing` → `Scintillators_Processing`
  → `Detector_Analysis`. Pass either Moyal fit ranges (to fit each spectrum) or fixed
  MPVs (to skip fitting); returns the datalogger processor, scintillator
  processor, and analysis instances.
- **`split_flight_and_background(...)`** — splits a run into flight vs.
  background using altitude, cutting where the payload returns to ground after
  apogee.
- **`split_by_time_marks(...)`** — splits a datalogger + scintillator set into
  sections between a list of Absolute Timer marks (e.g. ascent / float /
  descent).
- **`plot_density_heatmap_ampcal(...)`** — 2D density heatmap of the
  amplitude-calibrated cross-scintillator MIP (mean vs. spread across
  scintillators).
- **Saving helpers** (`set_results_dir`, `finish_mpl`, `finish_plotly`,
  `save_table`, `save_summary`) — by default every plot is shown inline; calling
  `set_results_dir("may31flight")` switches the module into save mode, writing
  PNGs and CSVs (including the per-scintillator fit-constant summary) into a
  `<name>_results/` folder.

**Key techniques:** deadtime-corrected livetime, nearest-timestamp
datalogger/scintillator alignment, Moyal fits for MPV extraction, MIP
normalization, amplitude calibration to a global-mean MPV, and
livetime-normalized rate heatmaps.

**Usage:**

One call runs the whole thing — it chains the three classes together and
produces all the calibration plots.

**Inputs:** a datalogger CSV and a list of per-scintillator TXT files
(ordered `scint1 … scintN`).


## `CW_calibration_grounddata.py`

Bench/ground calibration driver. Runs the lab datasets through the pipeline to
establish the reference MPVs and the temperature dependence of the detector
gain. In order, it:

- **Room-temperature run** — a 14-hour run fit with Moyal (`calibrate_and_analyze_grounddata`)
  to extract the baseline per-scintillator MPVs.
- **Cs-137 source + Cs-137 background runs** — processed with those fixed MPVs
  (`analyze_calibrated_data_with_fixed_MPVs`) and an amplitude-calibrated density heatmap
  (`plot_density_heatmap_ampcal`).
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

## `flight-data-analysis.py`

Flight data driver (May 31st flight). Applies the ground-calibration MPVs to the
balloon flight and analyzes the run in segments. Defines a few flight-specific
helpers on top of `calibration_methods`:

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

The main block loads the flight datalogger and three scintillator files,
processes the full run, trims it to the flight window, plots altitude vs. time
as a sanity check (and prints max altitude before/after trimming to confirm the
trim didn't clip the peak), then splits the flight into 4 time segments and runs
the fixed-MPV analysis using the MPVs from `CW_calibration_grounddata.py`.

---

### Usage

`calibration_methods.py` is the library — the classes and helpers that do the
work. `CW_calibration_grounddata.py` and `flight-data-analysis.py` are the two
worked examples that use it.

Every run follows the same three-step pattern:

```python
from calibration_methods import (
    Datalogger_Processing, Scintillators_Processing, Detector_Analysis, set_results_dir
)

set_results_dir("roomtemp")   # optional: save plots + CSVs to ./roomtemp_results/

# 1) datalogger → continuous Absolute Timer (handles timer resets)
df = Datalogger_Processing(datalogger_fp, show_plots=False).process()

# 2) align each scintillator to the datalogger (deadtime-corrected livetime)
proc = Scintillators_Processing([scint1_fp, scint2_fp, scint3_fp], df)

# 3) calibrate + plot — pick ONE mode:
analysis = Detector_Analysis(proc, df)
analysis.calibrate_and_analyze_grounddata(moyal_fit_ranges=[(46, 80), (46, 82), (44, 76)])  # fit each MPV
# analysis.analyze_calibrated_data_with_fixed_MPVs(MPVs=[56.78, 57.00, 54.64])                # or supply MPVs
```

Results live on the returned `analysis`: `analysis.master_df` (calibrated event
dataframe) and `analysis.mpv_per_scint` (MPVs used).

**The two example scripts show this pattern in context:**

- **`CW_calibration_grounddata.py`** — runs the bench datasets (room-temp,
  Cs-137, and temperature-varying fridge/freezer runs), extracts the reference
  MPVs via Moyal fits, and builds the gain-vs-temperature calibration. Start
  here to see the fitting workflow and how MPVs are established.
- **`flight-data-analysis.py`** — takes the MPVs from the ground calibration,
  trims the balloon run to the flight window, and analyzes it in time segments
  with the fixed-MPV mode. Start here to see the flight workflow and the
  segment-splitting helpers.

Pass **either** `Moyal_fit_ranges` **or** `MPVs`.

**Outputs:** the returned `analysis` holds `analysis.master_df` (the calibrated
event dataframe) and `analysis.mpv_per_scint` (the MPVs used). With
`set_results_dir(...)` set, plots and a `calibration_summary.csv` are written to
`./<name>_results/`.

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