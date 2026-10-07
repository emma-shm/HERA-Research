"""
Combined flight analysis: 1 or 2 flights at once, 2- or 3-scintillator, time alignment,
shared max-altitude cut, altitude-based segmentation starting at ascent (default 50 m).

Pipeline per flight
-------------------
  0. (optional) battery cut + time alignment of datalogger and scintillator files
  1. trim to flight, cut off everything below `ascent_alt_m` (default 50 m),
     split into N altitude segments up to the shared max altitude
  2. count events in a (mean MIP) x (spread MIP) window per segment
       - 3 scints: spread = SiPM_scints_std_MIP
       - 2 scints: spread = |MIP_scint1 - MIP_scint2|  (std of 2 points is not meaningful)
  3. pre-flight ground-data analysis + spread-metric sanity check

Running one flight vs. two
--------------------------
  Give run_flight_analysis() a single FlightConfig (or a list of one) and it runs that flight.
  Give it a list of two (or more) and it runs every flight, and, with shared_max_alt="auto",
  finds the highest altitude BOTH flights reached and applies that cut to both.
  A comparison plot of counts per segment is saved when more than one flight is run.

Shared max altitude (`shared_max_alt`)
-------------------------------------
  None     -> no common cut
  16532.4  -> fixed value in meters
  "auto"   -> lowest max altitude among all flights in this run
              (with only one flight there is nothing to share, so no cut is applied)

SWITCHING TO TIME SPLITS
------------------------
  Search for "TIME SPLIT" below: uncomment `run_segments_by_time` and the one
  commented call line inside `run_flight_analysis` (and comment out the altitude call).
"""

import os
import shutil
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import calibration_methods
from calibration_methods import *
from flightanalysis_methods import *
from alignment_methods import *   # align_files


#configuration stuff
@dataclass
class FlightConfig:
    name: str                              # used for folder / file names
    datalogger: str                        # datalogger CSV
    scints: dict                           # ordered {label: filepath}; 2 or 3 entries
    MPVs: Sequence[float]                  # one per scint, same order as `scints`
    results_dir: str

    # time alignment
    align: bool = False
    battery_cut_s: Optional[float] = None  # cut all files here before aligning; None = keep everything

    # trimming / segmentation
    ground_band: float = 100.0
    ascent_alt_m: Optional[float] = 50.0   # ascent "starts" when altitude first exceeds this; None = no cut. change as needed
    n_segments: int = 4

    # event windows. none is default but can change as needed
    box_x: Optional[tuple] = None
    box_y: Optional[tuple] = None

    # pre-flight section override [start, end]; None gives [0, time altitude first exceeds ascent_alt_m]
    preflight_marks: Optional[Sequence[float]] = None

    # set True if the MPVs are borrowed from other hardware (prints a warning each run)
    mpvs_are_placeholder: bool = False

    n_scints: int = field(init=False)

    def __post_init__(self):
        self.n_scints = len(self.scints)
        if self.n_scints not in (2, 3):
            raise ValueError(f"{self.name}: expected 2 or 3 scintillators, got {self.n_scints}")
        if len(self.MPVs) != self.n_scints:
            raise ValueError(f"{self.name}: {len(self.MPVs)} MPVs for {self.n_scints} scintillators")
        if self.box_x is None:
            self.box_x = (1.7, 3.7)
        if self.box_y is None:
            # 3 scints: std window; 2 scints: |diff| window
            self.box_y = (1.7, 4.0) if self.n_scints == 3 else (1.0, 3.0)


#defining functions
def get_spread_metric(md: pd.DataFrame, n_scints: int) -> pd.Series:
    """Spread between detectors: std for 3 scints, absolute difference for 2."""
    if n_scints == 3:
        return md["SiPM_scints_std_MIP"]
    if "SiPM_diff_scint1_minus_scint2" in md.columns:
        return md["SiPM_diff_scint1_minus_scint2"].abs()
    return (md["SiPM_MIP_CW12_scint1"] - md["SiPM_MIP_CW12_scint2"]).abs()


def spread_label(n_scints: int) -> str:
    return "MIP std" if n_scints == 3 else "MIP diff"


def _isolate_files(cfg: FlightConfig, tag: str, dl_fp: str, scint_fps: Sequence[str]):
    """Copy a datalogger file + scint files into <results_dir>/flight_files/<tag>/ and return the
    new paths. File basenames are kept; each scint goes in its own subfolder named by its label."""
    base = os.path.join(cfg.results_dir, "flight_files", tag)
    os.makedirs(base, exist_ok=True)
    new_dl = os.path.join(base, os.path.basename(dl_fp))
    shutil.copy2(dl_fp, new_dl)
    new_scints = []
    for label, fp in zip(cfg.scints.keys(), scint_fps):
        d = os.path.join(base, label)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, os.path.basename(fp))
        shutil.copy2(fp, dest)
        new_scints.append(dest)
    return new_dl, new_scints


def prepare_flight(cfg: FlightConfig) -> dict:
    """Steps 0-1a: optional alignment, trim to flight, cut below ascent altitude."""
    print(f"\n[{cfg.name}] PART 1: isolate flight ({cfg.n_scints} scints)")
    if cfg.mpvs_are_placeholder:
        warnings.warn(f"[{cfg.name}] MPVs are PLACEHOLDERS borrowed from other hardware. "
                      f"Results are not meaningful until this hardware is calibrated.")

    os.makedirs(cfg.results_dir, exist_ok=True)
    dl_fp = cfg.datalogger
    scint_fps = list(cfg.scints.values())

    # ---- optional battery cut + time alignment ----
    if cfg.align:
        end_t = cfg.battery_cut_s
        if end_t is None:
            raw_df = Datalogger_Processing(dl_fp, show_plots=False,
                                           results_dir=cfg.results_dir).process(name="raw_datalogger")
            end_t = raw_df["Absolute Timer (S)"].max() + 1

        # writes Timer[S] = Absolute Timer (rollovers corrected), cuts all files at end_t
        filtered = split_by_time_marks(dl_fp, scint_fps, time_marks=[0, end_t],
                                       labels=["battery_filtered"])[0]

        aligned = align_files(
            os.path.dirname(filtered["datalogger"]),
            {label: fp for label, fp in zip(cfg.scints.keys(), filtered["scints"])},
            filtered["datalogger"],
            results_dir=os.path.join(cfg.results_dir, "alignment_results"),
        )
        dl_fp = aligned["datalogger"]
        scint_fps = list(aligned["scints"])

    og_df = Datalogger_Processing(dl_fp, show_plots=False,
                                  results_dir=cfg.results_dir).process(name="original_flight_datalogger")

    trimmed_dl_fp, trimmed_scint_fps = trim_to_flight(dl_fp, scint_fps, ground_band=cfg.ground_band)
    trimmed_df = Datalogger_Processing(trimmed_dl_fp, show_plots=False,
                                       results_dir=cfg.results_dir).process(name="trimmed_flight_datalogger")

    max_orig = og_df["Altitude[m]"].max()
    max_trim = trimmed_df["Altitude[m]"].max()
    print(f"  Max altitude, original: {max_orig * 3.281:.0f} ft | trimmed: {max_trim * 3.281:.0f} ft "
          f"(should match)")

    # ---- start the flight at ascent (first time altitude > ascent_alt_m) ----
    if cfg.ascent_alt_m is not None:
        above = trimmed_df[trimmed_df["Altitude[m]"] > cfg.ascent_alt_m]
        ascent_t = above["Absolute Timer (S)"].min()
        end_t = trimmed_df["Absolute Timer (S)"].max() + 1
        print(f"  Ascent starts at t = {ascent_t:.0f} s (altitude > {cfg.ascent_alt_m:g} m); "
              f"cutting everything before it")
        sec = split_by_time_marks(trimmed_dl_fp, trimmed_scint_fps,
                                  time_marks=[ascent_t, end_t], labels=["ascent"])[0]
        trimmed_dl_fp, trimmed_scint_fps = sec["datalogger"], sec["scints"]
        trimmed_df = Datalogger_Processing(trimmed_dl_fp, show_plots=False,
                                           results_dir=cfg.results_dir).process(name="ascent_flight_datalogger")

    # ---- private copies of generated files, inside THIS flight's results folder ----
    # (helper functions may write intermediates to shared/fixed locations; copying guarantees one
    #  flight can never overwrite or reuse another flight's trimmed/aligned data)
    files_dir = os.path.join(cfg.results_dir, "flight_files")
    if os.path.exists(files_dir):
        shutil.rmtree(files_dir)
    trimmed_dl_fp, trimmed_scint_fps = _isolate_files(cfg, "trimmed", trimmed_dl_fp, trimmed_scint_fps)
    if cfg.align:
        dl_fp, scint_fps = _isolate_files(cfg, "aligned", dl_fp, scint_fps)

    # altitude plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(trimmed_df["Absolute Timer (S)"], trimmed_df["Altitude[m]"] * 3.281, color="blue", label="Flight")
    plt.xlabel("Timer"); plt.ylabel("Altitude [ft]")
    plt.title(f"{cfg.name}: Altitude vs Time (trimmed, from ascent)")
    plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, "altitude_check_trimmed_flight.png"))
    plt.close(fig)

    return dict(cfg=cfg, dl_fp=dl_fp, scint_fps=scint_fps, og_df=og_df,
                trimmed_dl_fp=trimmed_dl_fp, trimmed_scint_fps=trimmed_scint_fps,
                trimmed_df=trimmed_df, max_alt_m=trimmed_df["Altitude[m]"].max())


def resolve_shared_max_alt(prepared: list, shared_max_alt) -> Optional[float]:
    if shared_max_alt is None:
        return None
    if isinstance(shared_max_alt, str):
        if shared_max_alt.lower() != "auto":
            raise ValueError("shared_max_alt must be None, a number (m), or 'auto'")
        print("\nMax altitude reached by each flight:")
        for p in prepared:
            m = p["max_alt_m"]
            print(f"  {p['cfg'].name}: {m:.2f} m = {m * 3.281:.0f} ft")
        if len(prepared) < 2:
            print("Only one flight given, so there is nothing to share: no common max altitude cut applied.")
            return None
        val = min(p["max_alt_m"] for p in prepared)
        print(f"Shared max altitude (auto, lowest of all flights): {val:.2f} m = {val * 3.281:.0f} ft")
        return float(val)
    print(f"\nShared max altitude (given): {shared_max_alt:.2f} m = {shared_max_alt * 3.281:.0f} ft")
    return float(shared_max_alt)


def _clean_dir(path: str):
    """Clear partial results from an interrupted run."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"  cleared partial results: {path}")
    os.makedirs(path, exist_ok=True)


# ---------------------------- split by altitude (not time as previously) ----------------------------
def run_segments_by_altitude(prep: dict, common_max_alt_m: Optional[float],
                             twodim_hist_args: Optional[dict]):
    """Step 1b: split into N altitude segments (up to the shared max altitude, if given)."""
    cfg = prep["cfg"]
    run_name = f"{cfg.name}_{cfg.n_segments}seg" + ("_sharedmax" if common_max_alt_m else "")
    seg_dir = os.path.join(cfg.results_dir, run_name)
    _clean_dir(seg_dir)

    kwargs = dict(
        MPVs=list(cfg.MPVs),
        flight_df=prep["trimmed_df"],
        n_segments=cfg.n_segments,
        split_by="altitude",
        run_name=run_name,
        results_dir=seg_dir,
    )
    if common_max_alt_m is not None:
        kwargs["common_max_alt_m"] = common_max_alt_m
    if twodim_hist_args:
        kwargs["twodim_hist_args"] = twodim_hist_args

    segments = analyze_flight_in_segments(prep["trimmed_dl_fp"], prep["trimmed_scint_fps"], **kwargs)
    return segments, seg_dir, run_name


# ------------------- TIME SPLIT (commented out; uncomment to use) -------------------
# def run_segments_by_time(prep: dict, twodim_hist_args: Optional[dict] = None):
#     """Step 1b alternative: split into N equal-time segments (no shared max altitude)."""
#     cfg = prep["cfg"]
#     run_name = f"{cfg.name}_{cfg.n_segments}seg_time"
#     seg_dir = os.path.join(cfg.results_dir, run_name)
#     _clean_dir(seg_dir)
#
#     kwargs = dict(
#         MPVs=list(cfg.MPVs),
#         flight_df=prep["trimmed_df"],
#         n_segments=cfg.n_segments,
#         split_by="time",
#         run_name=run_name,
#         results_dir=seg_dir,
#     )
#     if twodim_hist_args:
#         kwargs["twodim_hist_args"] = twodim_hist_args
#
#     segments = analyze_flight_in_segments(prep["trimmed_dl_fp"], prep["trimmed_scint_fps"], **kwargs)
#     return segments, seg_dir, run_name
# -------------------------------------------------------------------------------------


def summarize_segments(segments: dict, seg_dir: str, run_name: str, cfg: FlightConfig):
    print(f"\n[{cfg.name}] Segment altitude summary")
    peak_label, peak_alt = None, -1
    for lbl, seg in segments.items():
        md = seg.master_df
        mean_ft = md["Altitude[m]"].mean() * 3.281
        max_ft = md["Altitude[m]"].max() * 3.281
        n_events = len(md.query("SiPM_scints_avg_MIP >= 0.1"))
        print(f"  {lbl}: mean alt {mean_ft:8.0f} ft | max alt {max_ft:8.0f} ft | "
              f"{n_events} coincident MIP-avg events")
        if mean_ft > peak_alt:
            peak_alt, peak_label = mean_ft, lbl
    print(f"  Highest-altitude segment: {peak_label} (mean alt {peak_alt:.0f} ft)")
    print(f"  Its density_heatmap.png is in: {seg_dir}/{run_name}_{peak_label}/")
    return peak_label


def count_window_events(segments: dict, seg_dir: str, cfg: FlightConfig, show_plots: bool):
    """Step 2: count events in the (mean MIP) x (spread) window per segment."""
    print(f"\n[{cfg.name}] PART 2: counts in window {cfg.box_x} x {cfg.box_y} "
          f"(mean MIP vs {spread_label(cfg.n_scints)})")
    counts, t_mid, boxes = [], [], []

    for i, (lbl, seg) in enumerate(segments.items(), start=1):
        md = seg.master_df.copy()
        md["spread_metric"] = get_spread_metric(md, cfg.n_scints)

        box = md[md["SiPM_scints_avg_MIP"].between(*cfg.box_x)
                 & md["spread_metric"].between(*cfg.box_y)]
        count = len(box)
        print(f"  Segment {i} ({lbl}): {count} counts")
        if cfg.n_scints == 2:
            print(f"    spread distribution: {md['spread_metric'].describe().to_dict()}")

        counts.append(count)
        t_mid.append(md["Absolute Timer (S)"].mean())
        boxes.append((lbl, box))   # heatmaps are drawn later, once the shared color scale is known

    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_mid, counts, "o-")
    plt.xlabel("Time (s)"); plt.ylabel("Counts")
    plt.title(f"{cfg.name}: counts in window ({cfg.box_x[0]}-{cfg.box_x[1]} mean MIP, "
              f"{cfg.box_y[0]}-{cfg.box_y[1]} {spread_label(cfg.n_scints)}) by segment", fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(seg_dir, "counts_by_segment.png"))
    if show_plots:
        plt.show()
    plt.close(fig)
    return dict(t_mid=t_mid, counts=counts, boxes=boxes, seg_dir=seg_dir)


def analyze_ground(prep: dict, show_plots: bool):
    """Step 3: pre-flight ground data."""
    cfg = prep["cfg"]
    print(f"\n[{cfg.name}] PART 3: pre-flight ground data")

    if cfg.preflight_marks is not None:
        marks = list(cfg.preflight_marks)
    else:
        thresh = cfg.ascent_alt_m if cfg.ascent_alt_m is not None else 50.0
        og = prep["og_df"]
        ascent_start = og[og["Altitude[m]"] > thresh]["Absolute Timer (S)"].min()
        print(f"  Ascent appears to begin around t = {ascent_start:.0f} s (altitude > {thresh:g} m)")
        marks = [0, ascent_start]

    sections = split_by_time_marks(prep["dl_fp"], prep["scint_fps"], time_marks=marks, labels=["pre-flight"])
    pre = next(s for s in sections if s["label"] == "pre-flight")

    df, processor, analysis = process_run(
        pre["datalogger"], pre["scints"],
        MPVs=list(cfg.MPVs),
        results_dir=os.path.join(cfg.results_dir, "pre-flight"),
        twodim_hist_args={"cbar_max": 1},
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(df["Absolute Timer (S)"], df["Altitude[m]"] * 3.281, color="blue", label="Ground")
    plt.xlabel("Timer"); plt.ylabel("Altitude [ft]")
    plt.title(f"{cfg.name}: Altitude vs Time, pre-flight")
    plt.savefig(os.path.join(cfg.results_dir, "altitude_check_pre-flight.png"))
    plt.close(fig)

    # spread-metric distribution (useful for choosing box_y)
    pf = analysis.master_df.copy()
    pf["spread_metric"] = get_spread_metric(pf, cfg.n_scints)
    print(f"  Spread metric ({spread_label(cfg.n_scints)}) on ground data:")
    print(pf["spread_metric"].describe())

    fig = plt.figure(figsize=(10, 5))
    plt.hist(pf["spread_metric"].dropna(), bins=50)
    plt.axvline(pf["spread_metric"].median(), color="red", linestyle="--", label="median")
    plt.xlabel(spread_label(cfg.n_scints)); plt.ylabel("Count")
    plt.title(f"{cfg.name}: spread distribution, pre-flight ground data")
    plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, "spread_metric_histogram_preflight.png"))
    if show_plots:
        plt.show()
    plt.close(fig)
    return analysis


HEATMAP_BINS = 25


def _window_hist(box: pd.DataFrame, cfg: FlightConfig) -> np.ndarray:
    """2D bin counts of the selected window, using fixed edges (the window itself)."""
    return np.histogram2d(box["SiPM_scints_avg_MIP"], box["spread_metric"],
                          bins=HEATMAP_BINS, range=[cfg.box_x, cfg.box_y])[0]


def compute_shared_cbar_max(results: dict) -> float:
    """Color scale max shared by every heatmap: the smallest of the per-flight ranges.
    Each flight's range is its highest single-bin count across all of its segment heatmaps."""
    per_flight = {}
    for name, res in results.items():
        cfg = res["cfg"]
        per_flight[name] = max((_window_hist(box, cfg).max() for _, box in res["window_counts"]["boxes"]),
                               default=0)
    print("\nHeatmap color range (max counts in a single bin) per flight:")
    for name, v in per_flight.items():
        print(f"  {name}: {v:.0f}")
    vmax = max(min(per_flight.values()), 1)
    print(f"Shared color scale for all heatmaps: 0 - {vmax:.0f}")
    return float(vmax)


def save_window_heatmaps(res: dict, vmax: float, show_plots: bool):
    """Save one PNG heatmap per segment for a flight, all using the same color scale."""
    cfg, wc = res["cfg"], res["window_counts"]
    for lbl, box in wc["boxes"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.hist2d(box["SiPM_scints_avg_MIP"], box["spread_metric"],
                      bins=HEATMAP_BINS, range=[cfg.box_x, cfg.box_y],
                      cmap="inferno", vmin=0, vmax=vmax)
        fig.colorbar(h[3], ax=ax, label="Counts per bin")
        ax.set_xlabel("Mean MIP")
        ax.set_ylabel(spread_label(cfg.n_scints))
        ax.set_title(f"{cfg.name} {lbl}: {len(box)} Counts")
        fig.savefig(os.path.join(wc["seg_dir"], f"{cfg.name}_{lbl}_selected_window_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)


def compare_flights(results: dict, comparison_dir: str, show_plots: bool):
    """Overlay window counts per altitude segment for every flight in the run."""
    n_types = {r["n_scints"] for r in results.values()}
    if len(n_types) > 1:
        warnings.warn("Comparing flights with different numbers of scintillators: the spread metric "
                      "and window differ (std vs |diff|), so counts are not directly comparable. "
                      "Use the 2-scint version of the 3-scint flight for a like-for-like comparison.")

    os.makedirs(comparison_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    for name, res in results.items():
        c = res["window_counts"]["counts"]
        plt.plot(range(1, len(c) + 1), c, "o-", label=f"{name} ({res['n_scints']} scints)")
    plt.xlabel("Altitude segment number")
    plt.ylabel("Counts in selected window")
    plt.title("Counts in selected window by segment (shared max altitude)")
    plt.xticks(range(1, max(len(r["window_counts"]["counts"]) for r in results.values()) + 1))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(comparison_dir, "counts_by_segment_comparison.png"))
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"\nComparison plot saved in: {comparison_dir}")


#main function!!!!
def run_flight_analysis(configs: Union[FlightConfig, Sequence[FlightConfig]],
                        shared_max_alt: Union[None, float, str] = None,
                        twodim_hist_args: Optional[dict] = None,
                        run_ground: bool = True,
                        show_plots: bool = False,
                        comparison_dir: str = "./Results/comparison") -> dict:
    """Run the full analysis on one or more flights (2- and 3-scint flights can be mixed)."""
    if isinstance(configs, FlightConfig):
        configs = [configs]

    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Flight names must be unique, got {names}")
    dirs = [os.path.abspath(c.results_dir) for c in configs]
    if len(set(dirs)) != len(dirs):
        raise ValueError(f"Each flight needs its own results_dir, got {[c.results_dir for c in configs]}")

    # pass 1: prepare all flights (so "auto" can see every flight's max altitude)
    prepared = [prepare_flight(c) for c in configs]
    common_alt = resolve_shared_max_alt(prepared, shared_max_alt)

    # pass 2: segment + count + ground. Each flight is isolated: if one fails, the error is printed
    # in full and the other flight(s) still run and save into their own folder.
    results, failed = {}, {}
    for prep in prepared:
        cfg = prep["cfg"]
        try:
            # ---- ALTITUDE SPLIT (active) ----
            segments, seg_dir, run_name = run_segments_by_altitude(prep, common_alt, twodim_hist_args)
            # ---- TIME SPLIT (swap with the line above; also uncomment run_segments_by_time) ----
            # segments, seg_dir, run_name = run_segments_by_time(prep, twodim_hist_args)

            peak = summarize_segments(segments, seg_dir, run_name, cfg)
            window = count_window_events(segments, seg_dir, cfg, show_plots)
            ground = analyze_ground(prep, show_plots) if run_ground else None
            results[cfg.name] = dict(segments=segments, peak_segment=peak, window_counts=window,
                                     ground_analysis=ground, common_max_alt_m=common_alt,
                                     trimmed_df=prep["trimmed_df"], n_scints=cfg.n_scints, cfg=cfg)
        except Exception as e:
            failed[cfg.name] = e
            print(f"\n!!! [{cfg.name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    # heatmaps last, so every flight shares one color scale (smallest range among the flights)
    if results:
        vmax = compute_shared_cbar_max(results)
        for res in results.values():
            save_window_heatmaps(res, vmax, show_plots)

        if len(results) > 1:
            compare_flights(results, comparison_dir, show_plots)

    # final report: where each flight's output lives
    print("\n================ RUN SUMMARY ================")
    for c in configs:
        status = "FAILED" if c.name in failed else "ok"
        print(f"  {c.name}: {status}  ->  {os.path.abspath(c.results_dir)}")
    if failed:
        print("  (see the traceback(s) above for the failed flight(s); successful flights were still saved)")
    return results


#running
if __name__ == "__main__":
    DATA_DIR = "./Data"
    RESULTS = "./Results"

    # ------------------------------------------------------------------
    # All available flights. Add/remove as needed.
    # ------------------------------------------------------------------
    FLIGHTS = {
        # May 31st, all 3 scintillators (with time alignment)
        "may31_3": FlightConfig(
            name="may31_3scint",
            datalogger=f"{DATA_DIR}/May_31st_Flight/AHD011 copy.csv",
            scints={"Top":    f"{DATA_DIR}/May_31st_Flight/left_AxLab_M_038.txt",
                    "Middle": f"{DATA_DIR}/May_31st_Flight/middle_AxLab_M_037 copy.txt",
                    "Bottom": f"{DATA_DIR}/May_31st_Flight/right_AxLab_M_038 copy.txt"},
            MPVs=[21.229423631297788, 22.107396502852346, 21.014219627449336],
            results_dir=f"{RESULTS}/May_31st_Flight/3scint",
            align=True,
            battery_cut_s=27500,
        ),

        # May 31st, mid + bottom only (like-for-like comparison with Aug 21st)
        "may31_2": FlightConfig(
            name="may31_2scint",
            datalogger=f"{DATA_DIR}/May_31st_Flight/AHD011 copy.csv",
            scints={"Middle": f"{DATA_DIR}/May_31st_Flight/middle_AxLab_M_037 copy.txt",
                    "Bottom": f"{DATA_DIR}/May_31st_Flight/right_AxLab_M_038 copy.txt"},
            MPVs=[22.107396502852346, 21.014219627449336],   # middle, bottom from the 3-scint list
            results_dir=f"{RESULTS}/May_31st_Flight/2scint",
        ),

        # Aug 21st, 2 scintillators (AxLab_M_001)
        "aug21_2": FlightConfig(
            name="aug21_2scint",
            datalogger=f"{DATA_DIR}/Aug_21st_Flight/dataloggerAHD000-flight.csv",
            scints={"Bottom": f"{DATA_DIR}/Aug_21st_Flight/bot-board-AxLab_M_001-flight.txt",
                    "Middle": f"{DATA_DIR}/Aug_21st_Flight/mid-AxLab_M_001-flight.txt"},
            MPVs=[22.107396502852346, 21.014219627449336],   # TODO: replace with AxLab_M_001 calibration
            mpvs_are_placeholder=True,
            results_dir=f"{RESULTS}/Aug21st_Flight",
        ),
    }

    # choose how it runs!!
    # one name = single flight, two names = both + shared max altitude
    RUN = ["may31_2", "aug21_2"]          # or ["may31_3"] for just one flight w 3 scintillators

    # "auto"  means lowest max altitude of the flights in RUN (ignored (so no cut) if only one flight)
    # 16532.40 (or any meters value) gives a fixed cut, works for one or two flights
    # None = no shared cut, full flight data
    SHARED_MAX_ALT = "auto"

    results = run_flight_analysis(
        [FLIGHTS[k] for k in RUN],
        shared_max_alt=SHARED_MAX_ALT,
        twodim_hist_args={"cbar_max": 140},
        comparison_dir=f"{RESULTS}/comparison",
    )