'''
Test driver: reproduce the notebook's rate plots using alignment_methods.py.

Folder layout (run from inside this folder):
    alignment_methods.py
    test_alignment_plot.py
    Top_battery_filtered.txt
    Middle_battery_filtered.txt
    Bottom_battery_filtered.txt
    AHD011_timer_corrected_battery_filtered.csv
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from alignment_methods import *    # load_sources, binned_rate, moving_average, align_files

# ================= Cell 2 parameters, verbatim (imports and %matplotlib inline removed) =================
DATA_DIR = Path(".")            # folder holding the four data files
WATCH_FILES = {
    "Top":    "Top_battery_filtered.txt",
    "Middle": "Middle_battery_filtered.txt",
    "Bottom": "Bottom_battery_filtered.txt",
}
LOGGER_FILE = "AHD011_timer_corrected_battery_filtered.csv"
LOGGER_COLUMNS = ["Events CW1&2", "Events CW1&2&3"]

BIN_SECONDS = 60.0              # bin width; 30 is noisier, 120 is smoother
SMOOTH_BINS = 1                 # centered moving average over N bins; 1 disables
MIN_COVERAGE = 0.5              # drop bins covering less than this fraction of BIN_SECONDS
DEADTIME_CORRECTION = False     # divide watch rates by live time using Deadtime[s]

T_MIN, T_MAX = 2500.0, 9500.0   # time cutoff for the windowed figure, raw timer seconds

XUNIT = "s"                     # "s", "min" or "h"
ERRORBAND = True                # shade the +/-1 sigma Poisson band
LOGY = False                    # log y axis
FIGSIZE = (13.0, 7.0)
DPI = 150
SAVE_DIR = Path("figures")
COLORS = {
    "Top CosmicWatch":        "#1f77b4",
    "Middle CosmicWatch":     "#2ca02c",
    "Bottom CosmicWatch":     "#d62728",
    "Logger Events CW1&2":    "#9467bd",
    "Logger Events CW1&2&3":  "#ff7f0e",
}
XUNITS = {"s": (1.0, "Timer [s]"), "min": (60.0, "Timer [min]"), "h": (3600.0, "Timer [h]")}

SAVE_DIR.mkdir(exist_ok=True)
print("Reading from:", DATA_DIR.resolve())
print("Figures to:  ", SAVE_DIR.resolve())


# ================= Cell 10 plot_rates, verbatim =================
def plot_rates(sources=None, tmin=None, tmax=None, bin_seconds=None, smooth=None,
               deadtime=None, errorband=None, logy=None, xunit=None, title=None,
               save_as=None, figsize=None):
    """Draw events/second vs timer. Every argument falls back to the parameters cell."""
    sources = SOURCES if sources is None else sources
    bin_seconds = BIN_SECONDS if bin_seconds is None else bin_seconds
    smooth = SMOOTH_BINS if smooth is None else smooth
    deadtime = DEADTIME_CORRECTION if deadtime is None else deadtime
    errorband = ERRORBAND if errorband is None else errorband
    logy = LOGY if logy is None else logy
    xunit = XUNIT if xunit is None else xunit
    figsize = FIGSIZE if figsize is None else figsize
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive.")

    low = min(s["window"][0] for s in sources) if tmin is None else float(tmin)
    high = max(s["window"][1] for s in sources) if tmax is None else float(tmax)
    if not high > low:
        raise ValueError(f"Empty timer range: tmin={low}, tmax={high}.")
    edges = np.arange(np.floor(low / bin_seconds) * bin_seconds, high + bin_seconds, bin_seconds)
    centers = 0.5 * (edges[:-1] + edges[1:])
    scale, xlabel = XUNITS[xunit]

    figure, axis = plt.subplots(figsize=figsize)
    rows = []
    for source in sources:
        # Confine the exposure window to the plotted range, or the reported span stays
        # at the full file length and the mean rate is meaningless.
        window = (max(source["window"][0], low), min(source["window"][1], high))
        dead = (source["times"], source["deadtime"]) if (deadtime and source["deadtime"] is not None) else (None, None)
        rate, error, counts, exposure = binned_rate(source["times"], source["weights"],
                                                    edges, window, deadtime=dead)
        curve = moving_average(rate, smooth)
        color = COLORS.get(source["label"])
        style = "-" if source["kind"] == "watch" else "--"
        total, span = float(counts.sum()), window[1] - window[0]
        mean = total / span if span > 0 else np.nan
        axis.plot(centers / scale, curve, style, color=color, linewidth=1.0, alpha=0.9,
                  label=f"{source['label']}  (mean {mean:.2f}/s)")
        if errorband:
            band = moving_average(error, smooth)
            axis.fill_between(centers / scale, curve - band, curve + band,
                              color=color, alpha=0.18, linewidth=0)
        rows.append((source["label"], total, span, mean, int(np.sum(np.isfinite(rate)))))

    axis.grid(alpha=0.3, linestyle=":")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Events / second")
    axis.set_xlim(low / scale, high / scale)
    if logy:
        axis.set_yscale("log")
    axis.legend(loc="best", fontsize=9, framealpha=0.9)
    if title is None:
        title = (f"Event rate vs timer  |  {bin_seconds:g} s bins"
                 + (f", {smooth}-bin moving average" if smooth > 1 else "")
                 + (", deadtime corrected" if deadtime else ""))
    axis.set_title(title)
    figure.tight_layout()

    print(f"{'series':<26}{'events':>12}{'span [s]':>13}{'mean [/s]':>12}{'bins':>7}")
    for label, total, span, mean, bins in rows:
        print(f"{label:<26}{total:>12,.0f}{span:>13.2f}{mean:>12.3f}{bins:>7}")
    if save_as:
        path = SAVE_DIR / save_as
        figure.savefig(path, dpi=DPI)
        print(f"\nSaved {path.resolve()}")
    plt.show()
    return figure, axis


# ================= Raw plots (cell 8 second half + cells 12, 14) =================
SOURCES = load_sources(DATA_DIR, WATCH_FILES, LOGGER_FILE, LOGGER_COLUMNS)
for source in SOURCES:
    print(f"{source['label']:<26} {int(source['weights'].sum()):>9,} events   "
          f"from {source['file']}")

plot_rates(save_as="event_rate_full.png")
plot_rates(tmin=T_MIN, tmax=T_MAX, save_as="event_rate_window.png")


# ================= Aligned plots (cell 25), read back from the WRITTEN files =================
# align_files() writes *_aligned.txt / *_aligned.csv. Re-reading those files (instead of
# using the in-memory ALIGNED list) tests exactly what calibration_methods will receive.
aligned = align_files(DATA_DIR, WATCH_FILES, LOGGER_FILE, LOGGER_COLUMNS)

aligned_watch_files = {}
for label, path in zip(WATCH_FILES.keys(), aligned["scints"]):   # same order as WATCH_FILES
    aligned_watch_files[label] = Path(path).name

ALIGNED_FROM_FILES = load_sources(DATA_DIR, aligned_watch_files,
                                  Path(aligned["datalogger"]).name, LOGGER_COLUMNS)

plot_rates(sources=ALIGNED_FROM_FILES, tmin=T_MIN, tmax=T_MAX, save_as="event_rate_aligned.png",
           title="Event rate vs timer, time-aligned  |  "
                 f"{BIN_SECONDS:g} s bins, estimator={ESTIMATOR}")

# The descent edge is the steepest feature, so it is where any misalignment shows up.
plot_rates(sources=ALIGNED_FROM_FILES, tmin=8000, tmax=8700, bin_seconds=10, smooth=3,
           save_as="event_rate_aligned_edge.png",
           title="Descent edge after alignment  |  10 s bins")
