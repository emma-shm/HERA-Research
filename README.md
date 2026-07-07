# HERA-Research
Repository for CW calibrations and data analysis scripts for the HERA collaboration and the HASP flight


### `Calibration/`
Scripts for calibrating HERA's CW detectors + using calibration for particle identification

#### `calibration_methods.py`

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

- **`CW_Processing`** — reads each scintillator file, applies a deadtime
  correction to compute per-event livetime, and aligns every scintillator to
  the datalogger independently via nearest-timestamp `merge_asof`. Handles
  arbitrary N-fold coincidence naming (CW12, CW123, …), tags coincidence events
  per scintillator, tracks per-scint and total deadtime-corrected livetime, and
  plots the SiPM voltage distributions (per-scint and combined).

- **`CW_Analysis`** — assembles a master dataframe anchored on the datalogger
  timeline. Normalizes each scintillator's SiPM amplitude to MIP units using
  per-scintillator MPVs (either from Moyal fits or supplied as fixed values),
  applies an amplitude calibration that shifts each channel to a global-mean
  MPV, and derives cross-scintillator mean/std columns. Produces the calibrated
  rate spectra and the 2D density heatmaps, with all rates livetime-normalized
  so runs of different duration are directly comparable.

**Top-level functions**

- **`processing_pipeline(datalogger, scintillators, Moyal_fit_ranges=None, MPVs=None, ...)`**
  — one-call entry point that chains `Datalogger_Processing` → `CW_Processing`
  → `CW_Analysis`. Pass either Moyal fit ranges (to fit each spectrum) or fixed
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

```python
from calibration_methods import processing_pipeline, set_results_dir

datalogger    = "data/may31_datalogger.csv"
scintillators = ["data/scint1.txt", "data/scint2.txt", "data/scint3.txt"]

# Optional: save PNGs + CSVs to ./may31flight_results/ instead of showing inline
set_results_dir("may31flight")

# Mode A — fit a Moyal to each scint to get its MPV
# (pass per-scint (low, high) mV windows around each MIP peak)
dl, scints, analysis = processing_pipeline(
    datalogger, scintillators,
    Moyal_fit_ranges=[(40, 120), (45, 130), (40, 115)],
    Show_plots=True,
)

# Mode B — supply MPVs from a prior calibration (skips fitting)
dl, scints, analysis = processing_pipeline(
    datalogger, scintillators,
    MPVs=[78.3, 81.0, 76.5],
    Show_plots=True,
)
```

Pass **either** `Moyal_fit_ranges` **or** `MPVs`.

**Outputs:** the returned `analysis` holds `analysis.master_df` (the calibrated
event dataframe) and `analysis.mpv_per_scint` (the MPVs used). With
`set_results_dir(...)` set, plots and a `calibration_summary.csv` are written to
`./<name>_results/`.

#### `CW_calibration_grounddata.py`

Bench/ground calibration driver. Runs the lab datasets through the pipeline to
establish the reference MPVs and the temperature dependence of the detector
gain. In order, it:

- **Room-temperature run** — a 14-hour run fit with Moyal (`rate_spectra_with_moyal`)
  to extract the baseline per-scintillator MPVs.
- **Cs-137 source + Cs-137 background runs** — processed with those fixed MPVs
  (`rate_spectra_with_fixed_MPVs`) and an amplitude-calibrated density heatmap
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

#### `flight-data-analysis.py`

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


### `HASP-Drexel/`
Scripts for applying that code/analysis to HASP flight data
- Ingesting and cleaning the flight datalogger files.
- Applying the calibration from `Calibration/`.
- Coincidence-rate and flux computation vs. altitude and time.
- 2D density heatmaps and summary plots (Plotly / Matplotlib).

---

## Data

The functions and scripts in the Calibration folder expect raw output from the CosmicWatch datalogger (SD-card / serial logs). Each event row typically includes an event number, timestamp/millis, the measured ADC value (SiPM peak), a computed deadtime, and — depending on configuration — coincidence and temperature fields. These scripts are not yet fully calibrated for the HASP data structure.

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


