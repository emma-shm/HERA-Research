'''
Methods developed by Annafy and converted to a python script for correcting drift offset in CW data.
'''

from pathlib import Path
import csv
import numpy as np
import pandas as pd          # new: used only to rewrite the datalogger CSV

MIN_COVERAGE = 0.5           # binned_rate() uses this as a default argument
BIN_SECONDS = 60.0
SMOOTH_BINS = 1
DEADTIME_CORRECTION = False
T_MIN, T_MAX = 2500.0, 9500.0
XUNIT = "s"
ERRORBAND = True
LOGY = False
FIGSIZE = (13.0, 7.0)
DPI = 150
SAVE_DIR = Path("figures")      # unused: align_files passes absolute paths (see step 4)
COLORS = {
    "Top CosmicWatch":        "#1f77b4",
    "Middle CosmicWatch":     "#2ca02c",
    "Bottom CosmicWatch":     "#d62728",
    "Logger Events CW1&2":    "#9467bd",
    "Logger Events CW1&2&3":  "#ff7f0e",
}
XUNITS = {"s": (1.0, "Timer [s]"), "min": (60.0, "Timer [min]"), "h": (3600.0, "Timer [h]")}

# alignment setup
ALIGN_SEARCH = (3500.0, 8800.0)   # where to look for the two maxima
PEAK_BIN = 30.0                   # bin width for peak finding (finer than the plot)
PEAK_SMOOTH = 11                  # moving-average bins; 11 x 30 s = 330 s window
XCORR_WINDOW = (7300.0, 8700.0)   # steep descent edge, used by the xcorr estimator
REFERENCE = "Logger Events CW1&2" # clock everything else is mapped onto

ESTIMATOR = "xcorr"       # "peaks" = two-maxima affine fit; "xcorr" = full-curve offset
SCALE_CORRECTION = False  # "peaks" only: fit scale as well as offset (read section 8 first)
PER_CLOCK = True          # one correction per physical clock, not per series
APPLY = True              # build the ALIGNED source list


def align_files(data_dir, watch_files, logger_file,
                logger_columns=("Events CW1&2", "Events CW1&2&3"), out_suffix="aligned",
                results_dir=None):
    data_dir = Path(data_dir)
    SOURCES = load_sources(data_dir, watch_files, logger_file, list(logger_columns))

    # >>> cell 19 from "# ---- run detection" to the end, indented one level, unchanged
    PEAKS = {}
    for source in SOURCES:
        centers, curve = rate_curve(source, *ALIGN_SEARCH, PEAK_BIN, PEAK_SMOOTH)
        (first, second), drop = find_two_maxima(centers, curve, PEAK_SMOOTH // 2 + 1)
        PEAKS[source["label"]] = {"peak1": first, "peak2": second,
                                "separation": second - first, "prominence": drop}

    print(f"Two maxima per series   |   {PEAK_BIN:g} s bins, {PEAK_SMOOTH}-bin smoothing, "
        f"search {ALIGN_SEARCH[0]:g}-{ALIGN_SEARCH[1]:g} s\n")
    print(f"{'series':<26}{'ascent [s]':>12}{'descent [s]':>13}{'separation [s]':>16}{'prominence':>12}")
    for label, peak in PEAKS.items():
        print(f"{label:<26}{peak['peak1']:>12.2f}{peak['peak2']:>13.2f}"
            f"{peak['separation']:>16.2f}{peak['prominence']:>12.2f}")

    control = (PEAKS["Logger Events CW1&2"]["peak1"] - PEAKS["Logger Events CW1&2&3"]["peak1"],
            PEAKS["Logger Events CW1&2"]["peak2"] - PEAKS["Logger Events CW1&2&3"]["peak2"])
    print(f"\nSAME-CLOCK CONTROL  (CW1&2 vs CW1&2&3, same Timer[S], true difference = 0.00 s)")
    print(f"  ascent  {control[0]:+8.2f} s      descent {control[1]:+8.2f} s"
        f"      <-- this is the two-maxima method's noise floor")
    
    # >>> All of cell 23, indented one level, unchanged
    reference_source = next(s for s in SOURCES if s["label"] == REFERENCE)
    reference_peaks = PEAKS[REFERENCE]

    mapping = {}
    print(f"{'series':<26}{'peaks: scale':>14}{'offset [s]':>13}"
        f"{'xcorr: offset [s]':>20}{'corr':>9}")
    for source in SOURCES:
        peak = PEAKS[source["label"]]
        span = peak["peak2"] - peak["peak1"]
        peak_scale = (reference_peaks["peak2"] - reference_peaks["peak1"]) / span if span else 1.0
        if not SCALE_CORRECTION:
            peak_scale = 1.0
        peak_offset = reference_peaks["peak1"] - peak_scale * peak["peak1"]

        if source["label"] == REFERENCE:
            lag, corr = 0.0, 1.0
        else:
            lag, corr = xcorr_offset(reference_source, source, XCORR_WINDOW)
        mapping[source["label"]] = {"peak_scale": peak_scale, "peak_offset": peak_offset,
                                    "xcorr_offset": -lag, "corr": corr}
        print(f"{source['label']:<26}{peak_scale:>14.6f}{peak_offset:>13.2f}"
            f"{-lag:>20.2f}{corr:>9.5f}")

    if PER_CLOCK:   # the two logger columns share one clock -> one shared correction
        logger_labels = [s["label"] for s in SOURCES if s["kind"] == "logger"]
        for key in ("peak_scale", "peak_offset", "xcorr_offset"):
            shared = float(np.mean([mapping[label][key] for label in logger_labels]))
            for label in logger_labels:
                mapping[label][key] = shared

    ALIGNED = []
    for source in SOURCES:
        fit = mapping[source["label"]]
        scale = fit["peak_scale"] if ESTIMATOR == "peaks" else 1.0
        offset = fit["peak_offset"] if ESTIMATOR == "peaks" else fit["xcorr_offset"]
        if not APPLY:
            scale, offset = 1.0, 0.0
        aligned = dict(source)
        aligned["times"] = scale * source["times"] + offset
        aligned["window"] = (scale * source["window"][0] + offset,
                            scale * source["window"][1] + offset)
        if source["deadtime"] is not None:
            aligned["deadtime"] = source["deadtime"] * scale
        aligned["applied"] = (scale, offset)
        ALIGNED.append(aligned)

    print(f"\nApplied estimator: {ESTIMATOR!r}   scale correction: {'on' if (ESTIMATOR == 'peaks' and SCALE_CORRECTION) else 'off'}")
    for source in ALIGNED:
        scale, offset = source["applied"]
        print(f"  {source['label']:<26} t -> {scale:.6f} * t {offset:+.2f} s")

    # ---------------- NEW: write aligned files in the formats calibration_methods reads
    applied = {}
    for s in ALIGNED:
        applied[s["label"]] = s["applied"]              # label -> (scale, offset)

    scint_out = []
    for label, filename in watch_files.items():         # dict order = scint1..scintN
        scale, offset = applied[f"{label} CosmicWatch"]
        src = data_dir / filename
        dst = data_dir / f"{src.stem}_{out_suffix}.txt"
        with src.open("r", encoding="utf-8", errors="replace") as fin, \
             dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    fout.write(line)                     # header/blank lines copied as-is
                    continue
                cells = stripped.split("\t")             # calibration reader uses sep='\t'
                cells[1] = f"{scale * float(cells[1]) + offset:.6f}"   # Time[s]
                cells[5] = f"{scale * float(cells[5]):.6f}"            # Deadtime[s], same as notebook
                fout.write("\t".join(cells) + "\n")
        scint_out.append(str(dst))

    scale, offset = applied[REFERENCE]                   # PER_CLOCK: both logger columns share this
    src = data_dir / logger_file
    dst = data_dir / f"{src.stem}_{out_suffix}.csv"
    dl = pd.read_csv(src, encoding="utf-8-sig")
    dl["Timer[S]"] = scale * dl["Timer[S]"] + offset
    dl.to_csv(dst, index=False)


        # ---------------- NEW: save the notebook figures to alignment_results
    if results_dir is None:
        results_dir = data_dir / "alignment_results"      # default: subfolder next to the data
    results_dir = Path(results_dir).resolve()               # absolute, so SAVE_DIR / save_as ignores SAVE_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_rates(sources=SOURCES, save_as=results_dir / "event_rate_full.png")
    plot_rates(sources=SOURCES, tmin=T_MIN, tmax=T_MAX, save_as=results_dir / "event_rate_window.png")
    plot_rates(sources=ALIGNED, tmin=T_MIN, tmax=T_MAX, save_as=results_dir / "event_rate_aligned.png",
               title="Event rate vs timer, time-aligned  |  "
                     f"{BIN_SECONDS:g} s bins, estimator={ESTIMATOR}")
    plot_rates(sources=ALIGNED, tmin=8000, tmax=8700, bin_seconds=10, smooth=3,
               save_as=results_dir / "event_rate_aligned_edge.png",
               title="Descent edge after alignment  |  10 s bins")
    
    return {"datalogger": str(dst), "scints": scint_out}


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def read_watch(path):
    """Return (event_time_s, event_weight, cumulative_deadtime_s) for a CosmicWatch TXT."""
    times, events, dead = [], [], []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cells = stripped.split()
            if len(cells) < 6:
                raise ValueError(f"{Path(path).name} line {line_number}: expected >=6 fields.")
            events.append(int(cells[0]))
            times.append(float(cells[1]))
            dead.append(float(cells[5]))

    if not times:
        raise ValueError(f"{Path(path).name}: no data rows found.")
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=np.int64)
    dead = np.asarray(dead, dtype=float)

    if np.any(np.diff(times) < 0):
        raise ValueError(f"{Path(path).name}: Time[s] is not monotonic -- unwrap the timer first.")
    if np.any(np.diff(events) <= 0):
        raise ValueError(f"{Path(path).name}: Event column is not strictly increasing.")

    # First row counts as one event; later rows count their increase over the previous row.
    weights = np.diff(events, prepend=events[0] - 1).astype(float)
    if np.any(np.diff(dead) < 0):
        dead = np.maximum.accumulate(dead)
    return times, weights, dead


def read_logger(path, columns):
    """Return (sample_time_s, {column: per_row_counter_increase}) for the logger CSV."""
    times = []
    counts = {name: [] for name in columns}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{Path(path).name}: empty file.")
        header = [cell.strip() for cell in header]
        for name in ["Timer[S]", *columns]:
            if header.count(name) != 1:
                raise ValueError(f"{Path(path).name}: missing or repeated column {name!r}.")
        time_index = header.index("Timer[S]")
        column_index = {name: header.index(name) for name in columns}

        for line_number, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"{Path(path).name} line {line_number}: "
                                 f"{len(row)} fields, expected {len(header)}.")
            times.append(float(row[time_index]))
            for name, index in column_index.items():
                counts[name].append(int(row[index]))

    if len(times) < 2:
        raise ValueError(f"{Path(path).name}: need >=2 rows to form counter differences.")
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{Path(path).name}: Timer[S] decreases -- use the corrected CSV.")

    increases = {}
    for name in columns:
        series = np.asarray(counts[name], dtype=np.int64)
        delta = np.diff(series)
        if np.any(delta < 0):
            raise ValueError(f"{Path(path).name}: counter {name!r} is not cumulative.")
        increases[name] = delta.astype(float)   # first row dropped: snapshot, no event time
    return times, increases




def binned_rate(event_times, weights, edges, window, deadtime=(None, None),
                min_coverage=MIN_COVERAGE):
    """Events per SECOND per bin, divided by the live time actually covered."""
    counts, _ = np.histogram(event_times, bins=edges, weights=weights)
    start, stop = window
    low = np.clip(edges[:-1], start, stop)
    high = np.clip(edges[1:], start, stop)
    exposure = np.maximum(high - low, 0.0)

    dead_times, dead_cumulative = deadtime
    if dead_cumulative is not None and len(dead_times) >= 2:
        clipped = np.clip(edges, dead_times[0], dead_times[-1])
        exposure = np.maximum(exposure - np.diff(np.interp(clipped, dead_times, dead_cumulative)), 0.0)

    width = float(edges[1] - edges[0])
    good = exposure >= min_coverage * width
    rate = np.full(counts.shape, np.nan)
    error = np.full(counts.shape, np.nan)
    rate[good] = counts[good] / exposure[good]                      # events / second
    error[good] = np.sqrt(np.maximum(counts[good], 0.0)) / exposure[good]
    return rate, error, counts, exposure


def moving_average(values, window):
    """Centered moving average that skips NaN bins instead of poisoning them."""
    if window <= 1:
        return values
    finite = np.isfinite(values)
    kernel = np.ones(int(window), dtype=float)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    smoothed = np.divide(numerator, denominator,
                         out=np.full_like(values, np.nan), where=denominator > 0)
    smoothed[~finite] = np.nan
    return smoothed



def load_sources(DATA_DIR, WATCH_FILES, LOGGER_FILE, LOGGER_COLUMNS):
    sources = []
    for label, filename in WATCH_FILES.items():
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path.resolve()}")
        times, weights, dead = read_watch(path)
        sources.append({"label": f"{label} CosmicWatch", "kind": "watch", "file": path.name,
                        "times": times, "weights": weights, "deadtime": dead,
                        "window": (0.0, float(times[-1]))})

    path = DATA_DIR / LOGGER_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path.resolve()}")
    logger_times, increases = read_logger(path, LOGGER_COLUMNS)
    for name in LOGGER_COLUMNS:
        sources.append({"label": f"Logger {name}", "kind": "logger", "file": path.name,
                        "times": logger_times[1:], "weights": increases[name], "deadtime": None,
                        "window": (float(logger_times[0]), float(logger_times[-1]))})
    return sources



def rate_curve(source, low, high, bin_seconds, smooth):
    """Smoothed events/second on a fixed grid, for peak finding and correlation."""
    count = int(np.floor((high - low) / bin_seconds))   # whole bins only: a short trailing
    if count < 3:                                       # bin would fail the coverage test
        raise ValueError("Search window is too narrow for this bin width.")
    edges = low + bin_seconds * np.arange(count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    window = (max(source["window"][0], low), min(source["window"][1], high))
    rate, _, _, _ = binned_rate(source["times"], source["weights"], edges, window,
                                min_coverage=0.9)
    curve = moving_average(rate, smooth)
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"{source['label']}: gaps inside {low}-{high} s; narrow ALIGN_SEARCH.")
    return centers, curve


def find_two_maxima(centers, curve, margin):
    """Two Pfotzer crossings via dR/dt sign changes, refined to the zero of the derivative."""
    derivative = np.gradient(curve, centers)
    crossings = [i for i in np.where((derivative[:-1] > 0) & (derivative[1:] <= 0))[0]
                 if margin <= i < len(curve) - margin - 1]

    peaks = set()                       # snap each crossing onto a true local maximum
    for index in crossings:
        walk = index
        while 0 < walk < len(curve) - 1 and (curve[walk - 1] > curve[walk] or curve[walk + 1] > curve[walk]):
            walk = walk - 1 if curve[walk - 1] > curve[walk] else walk + 1
        peaks.add(walk)

    interior = slice(margin, len(curve) - margin)
    main = int(np.argmax(curve[interior])) + margin
    best = None
    for candidate in sorted(peaks):     # second peak = largest drop between it and the main one
        if candidate == main:
            continue
        low, high = sorted((candidate, main))
        drop = curve[candidate] - curve[low:high + 1].min()
        if best is None or drop > best[0]:
            best = (drop, candidate)
    if best is None:
        raise ValueError("Only one maximum found; widen ALIGN_SEARCH or lower PEAK_SMOOTH.")

    refined = []
    for index in sorted((main, best[1])):
        i = index - 1 if derivative[index] <= 0 else index
        i = max(0, min(i, len(derivative) - 2))
        if derivative[i] == derivative[i + 1]:
            refined.append(float(centers[index]))
        else:
            step = centers[i + 1] - centers[i]
            zero = centers[i] - derivative[i] * step / (derivative[i + 1] - derivative[i])
            refined.append(float(np.clip(zero, centers[index] - step, centers[index] + step)))
    return refined, float(best[0])


def xcorr_offset(reference, source, window, bin_seconds=10.0, smooth=5, max_lag=400.0):
    """Lag of `source` relative to `reference` from the whole curve shape, not two points."""
    centers, a = rate_curve(reference, window[0], window[1], bin_seconds, smooth)
    _, b = rate_curve(source, window[0], window[1], bin_seconds, smooth)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    lags = np.arange(-int(max_lag / bin_seconds), int(max_lag / bin_seconds) + 1)
    score = np.array([np.corrcoef(a, np.roll(b, int(k)))[0, 1] for k in lags])
    i = int(np.argmax(score))
    shift = 0.0
    if 0 < i < len(score) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        if (y0 - 2 * y1 + y2) != 0:
            shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float((lags[i] + shift) * bin_seconds), float(score[i])


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






