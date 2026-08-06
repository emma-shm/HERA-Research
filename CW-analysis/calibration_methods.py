import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import linregress
import os


# SAVE_PLOTS  = False        # off by default → methods still just .show()
# RESULTS_DIR = "plots"      # overwritten by set_results_dir()

class PlotOutput:
    """Save/show policy shared by Datalogger_Processing, Scintillators_Processing,
    and Detector_Analysis. Subclasses call _init_output() in their __init__, then
    end every figure with _finish_mpl() / _finish_plotly() instead of plt.show().

    Two flags drive everything, set once in _init_output:
        save_plots : True whenever results_dir is given (the directory is created then)
        show_plots : None -> show only when not saving; True/False -> forced

    Resulting behaviour:
        results_dir given, show_plots=None/False  -> save only
        results_dir given, show_plots=True        -> save and show
        results_dir None,  show_plots=None/True   -> show only
        results_dir None,  show_plots=False       -> no output at all; the
            `if not (self.show_plots or self.save_plots): return` guards in
            subplots(), plot_SiPM_histograms(), and plot_all_SiPM_distributions()
            skip those figures entirely

    Methods:
        _init_output(results_dir, show_plots) : set the flags, mkdir if saving
        _finish_mpl(fig, name)     : save (if name is not None), show, always close
        _finish_plotly(fig, name)  : same, minus the close; needs kaleido to write PNG
        _save_table(df, name)      : write a CSV; gated on save_plots only

    Passing name=None to the _finish_* methods suppresses saving for that one
    figure while leaving show behaviour intact — used by Datalogger_Processing.subplots()
    so the raw Timer[S] plot is never written, only the Absolute Timer version.
    """

    def _init_output(self, results_dir=None, show_plots=None):
        
        # Called from each subclass's __init__. Sets the two flags every
        # plotting method downstream reads: self.save_plots and self.show_plots.
        self.results_dir = results_dir
        self.save_plots  = results_dir is not None          # saving is implied by giving a directory
        
        # show_plots=None means "infer": show only if we're NOT saving.
        # Passing True/False overrides that inference explicitly.
        if show_plots is None:
            show_plots = not self.save_plots      # default: show only when we're not saving
        self.show_plots = bool(show_plots)
        if self.save_plots:
            os.makedirs(self.results_dir, exist_ok=True)    # create the folder up front, no-op if it exists
            print(f"[plots] saving to {self.results_dir}/")

    def _finish_mpl(self, fig, name=None):
        # Terminal call for every matplotlib figure: save, show, close — in that order.
        if self.save_plots and name is not None:            # name=None => figure is deliberately not saved
            path = os.path.join(self.results_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[saved] {path}")
        if self.show_plots:
            plt.show()
        plt.close(fig)                                      # always close, so long loops don't leak figures

    def _finish_plotly(self, fig, name=None):
        # Same thing as _finish_mpl but for plotly, which I use for 2D histograms. No close() — plotly figures
        # aren't held by a global figure manager the way matplotlib's are.
        if self.save_plots and name is not None:
            path = os.path.join(self.results_dir, f"{name}.png")
            fig.write_image(path, scale=2)                  # static PNG export; needs kaleido installed
            print(f"[saved] {path}")
        if self.show_plots:
            fig.show()

    def _save_table(self, df, name):
        # CSV counterpart to the figure writers. Gated on save_plots only —
        # there's no "show" equivalent for a dataframe.
        if self.save_plots:
            path = os.path.join(self.results_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"[saved] {path}")

# ================== Main processing code ==================
class Datalogger_Processing(PlotOutput):
    def __init__(self, filepath, show_plots=None, debug=False, results_dir=None):
        '''
        Class for processing datalogger files. Detects timer resets in the datalogger file, and adds a column showing the absolute timer without any resets.

        Creates key dataframes used in the rest of the analysis/by the other classes:
            - self.df: the raw datalogger dataframe, read from the CSV, with no processing yet
            - self.timerreset_segments: list of DataFrames, one per continuous run between timer resets

        Arg:
        filepath: string of filepath to datalogger CSV
        '''
        self.fp = filepath
        self.fp_string = self.fp.split('/')[-1]
        self.df = pd.read_csv(self.fp)
        self.debug = debug
        self._init_output(results_dir, show_plots)

    def process(self, plotting_title=None, name="datalogger_abs"):
      '''
      Full pipeline: plots raw data vs Timer[S], detects timer resets, creates continuous Absolute Timer (S) column,
      and plots everything again vs Absolute Timer (S). Returns self.df so it can be chained into one line.

      Usage: datalogger_df = Datalogger_Processing(fp).process()
      '''
      self.subplots()                                                  # initial plot with raw Timer[S]
      self.separate_timer_resets(plotting_title=plotting_title, name=name) # detects resets, builds Absolute Timer (S), re-plots vs it
      print(f'Total run time: {self.df["Absolute Timer (S)"].max():.2f} s, or {self.df["Absolute Timer (S)"].max()/60:.2f} min')
      return self.df                                                   # return the fully-processed dataframe so caller can grab it directly

    def subplots(self, xaxis='Timer[S]', title=None, name=None):
      '''
      Method for subplots of datalogger data, mostly just for visualization. Plots the number of events, pressure, and temperature vs the specified x-axis (Default is Absolute Timer (S)).
      '''
      if not (self.show_plots or self.save_plots):
          return
      if title is None:
          title = self.fp_string

      fig1, axes = plt.subplots(1, 3, figsize=(20, 5))

      axes[0].plot(self.df[xaxis], self.df['Events CW1&2'], marker='o', markersize=0.7, color='blue', label="CW 1 & 2")
      axes[0].plot(self.df[xaxis], self.df['Events CW1&2&3'], marker='o', markersize=0.7, color='magenta', label="CW 1 & 2 & 3")
      axes[0].set_xlabel(xaxis)
      axes[0].set_ylabel('Events')
      axes[0].legend()
      axes[0].set_title('CosmicWatch Events over Time')

      axes[1].plot(self.df[xaxis], self.df['Pressure[Pa]'], marker='o', markersize=0.7, color='green', label="Pressure")
      axes[1].set_xlabel(xaxis)
      axes[1].set_ylabel('Pressure[Pa]')
      axes[1].legend()
      axes[1].set_title('Pressure over Time')

      axes[2].plot(self.df[xaxis], self.df['Temperature[C]'], marker='o', markersize=0.7, color='red', label="Temperature")
      axes[2].set_xlabel(xaxis)
      axes[2].set_ylabel('Temperature[C]')
      axes[2].legend()
      axes[2].set_title('Temperature over Time')

      plt.suptitle(title)
      self._finish_mpl(fig1, name)
    #   plt.close(fig1)

    def separate_timer_resets(self, plotting_title=None, name="datalogger_abs"):
        '''
        Detect any timer resets in the datalogger data, and creates separate DataFrames for each run. Then creates an Absolute Timer column to add to the dataframe, and plots the data as a function of the absolute timer.
        '''
        if plotting_title is None:
            plotting_title = self.fp_string

        self.df_sorted = self.df.copy().sort_index().reset_index(drop=True)                      # make sure rows are in original order + give clean 0-based index

        time_col = 'Timer[S]'                                                    # name of the column that holds the timer values

        reset_mask = self.df_sorted[time_col].diff() < -10                                # .diff() computes row-to-row differences; this creates a boolean mask that's True wherever the time decreased by more than 10 seconds (negative value indicates a reset/jump backward)

        new_segment_starts = [0] + (reset_mask[reset_mask].index).tolist()
        print(f"Detected {len(new_segment_starts)-1} separate timer reset(s) at rows {new_segment_starts}")

        self.timerreset_segments = []                                                            # empty list that will store one DataFrame per continuous run

        for i in range(len(new_segment_starts)):                                 # loop over each detected starting point
            start = new_segment_starts[i]                                        # determine starting row index of current segment; store in "start"
            end = new_segment_starts[i+1] if i+1 < len(new_segment_starts) else None   # ending row = start of next segment, or None (means go to end of DataFrame)
            seg = self.df_sorted.iloc[start:end].copy()                                  # slice out this segment and make independent copy; iloc
            seg['Timer_rel'] = seg[time_col] - seg[time_col].iloc[0]             # add relative time column that starts at ~0 for this run
            self.timerreset_segments.append(seg)                                                 # store this segment in the list

        for i, seg in enumerate(self.timerreset_segments, 1):
            t_min = seg[time_col].min()
            t_max = seg[time_col].max()
            nrows = len(seg)
            print(f"  Run {i:2d}: {nrows:5d} rows, timer {t_min:6.0f} → {t_max:6.0f} s")

        self.create_absolute_timer()

        if self.debug:
            for i, start_idx in enumerate(new_segment_starts):
                print(f"\n--- Around start of Timer Reset {i+1} (row {start_idx}) ---")

                start = max(0, start_idx - 5)
                end   = start_idx + 6

                slice_df = self.df.iloc[start:end] # Changed from self.df_sorted to self.df

                print(slice_df[[time_col, 'Events CW1&2', 'Events CW1&2&3', 'Absolute Timer (S)']].to_string(index=True))

        self.subplots(xaxis='Absolute Timer (S)', title=plotting_title, name=name)


    def create_absolute_timer(self):
      '''
      Creates continuous Absolute Timer by chaining last value from before timer reset
      '''
      absolute_timers = []
      previous_end = 0.0

      for i, seg in enumerate(self.timerreset_segments):
        current_timers = seg['Timer[S]'].values

        if len(current_timers) == 0:
            continue

        absolute_this_segment = current_timers + previous_end

        absolute_timers.extend(absolute_this_segment)

        previous_end = absolute_this_segment[-1]

      self.df['Absolute Timer (S)'] = absolute_timers

    
    # def _finish_mpl(self, fig, name=None):
    #     save = getattr(self, 'save_plots', False)
    #     show = getattr(self, 'show_plots', False)
    #     if save and name is not None:
    #         path = os.path.join(self.results_dir, f"{name}.png")
    #         fig.savefig(path, dpi=150, bbox_inches='tight')
    #         print(f"[saved] {path}")
    #     if show or not save:
    #         plt.show()
    #     plt.close(fig)


class Scintillators_Processing(PlotOutput):
    def __init__(self, filepaths, datalogger_df, show_plots=None, debug=False, results_dir=None):
        '''
        Class for processing scintillator files, for any number of scintillators. Instantiate the class once with a list of filepaths for all scintillator TXT files and the datalogger dataframe from the Datalogger_Processing class.
        The class has the following key methods, all of which are executed immediately when the class is instantiated:
                - static methods: _coinc_tag(k) and _coinc_src_col(k) for arbitrary-N coincidence naming, so that if you have 3 scintillators, you get CW12, CW123 for coincidence columns instead of hardcoding it.
                - apply_deadtime_correction(i): computes deadtime-corrected per-event livetime for a given scintillator, returning the results as an array
                - align_with_datalogger(): create aligned_scint{i} dataframe from each of the N scintillators (i=1,2,..,N); each dataframe contains the original scintillator columns plus the datalogger columns, aligned by nearest-neighbour timestamp matching;
                                            automatically computes deadtime-corrected livetime for each scintillator and adds it as a column
                - plot_all_SiPM_distributions(): plots the combined SiPM voltage distribution across all scintillators
                - plot_SiPM_histograms(i): plots the SiPM voltage distribution for a given scintillator, with separate histograms for all events and each coincidence order (CW12, CW123, etc.)
                
        Arg:
            filepaths: list of filepath strings for each scintillator TXT
            datalogger_df: datalogger dataframe from Datalogger_Processing() class (with Absolute Timer column already created)
        '''
        self.fps = filepaths
        self.debug = debug
        self._init_output(results_dir, show_plots)

        # Helpers for arbitrary-N coincidence naming
        n = len(filepaths)
        self.coinc_orders = list(range(2, n + 1))                       # [2,3] for N=3; [2,3,4] for N=4
        self.top_tag = 'CW' + ''.join(str(i) for i in range(1, n + 1))  # 'CW123' or 'CW1234'

        columns = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]',
                   'SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']

        for i, fp in enumerate(filepaths, start=1):
            df = pd.read_csv(fp, sep='\t', comment='#', header=None,
                             skiprows=3, names=columns, engine='python')
            setattr(self, f'scint_{i}', df)

        self.align_with_datalogger(datalogger_df)

        # ── individual histograms ─────────────────────────────────────────────
        for i in range(1, len(self.fps) + 1):
            self.plot_SiPM_histograms(i)

        self.plot_all_SiPM_distributions()

    @staticmethod
    def _coinc_tag(k):
        # k=2 -> 'CW12', k=3 -> 'CW123', k=4 -> 'CW1234'
        return 'CW' + ''.join(str(i) for i in range(1, k + 1))

    @staticmethod
    def _coinc_src_col(k):
        # k=2 -> 'Events CW1&2', k=3 -> 'Events CW1&2&3', k=4 -> 'Events CW1&2&3&4'
        return 'Events CW' + '&'.join(str(i) for i in range(1, k + 1))

    def apply_deadtime_correction(self, i):
        """
        Computes deadtime-corrected per-event livetime for scintillator i.

        For each event row, livetime = (time elapsed since last event) - (deadtime accumulated since last event)
        This is then attached to the scintillator DataFrame as a column before merging.

        Args:
            i: scintillator index (1-based)

        Returns:
            pd.Series of deadtime-corrected livetime values, aligned to scint_i's index
        """
        scint_df = getattr(self, f'scint_{i}')
        time  = scint_df['Time[s]'].values
        deadt = scint_df['Deadtime[s]'].values

        # delta deadtime per event
        event_deadt_s    = np.diff(np.append([0], deadt)) # add 0 at the start to align with the first event, then compute the difference between consecutive deadtime values
        event_livetime_s = np.diff(np.append([0], time)) - event_deadt_s # take livetime as the difference between consecutive time values minus the deadtime values for each time value

        # clip negatives — can occur at file boundaries or timer resets
        event_livetime_s = event_livetime_s.clip(min=0)

        return pd.Series(event_livetime_s, index=scint_df.index, name=f'livetime_scint{i}[s]')

    def align_with_datalogger(self, datalogger_df, tolerance=0.5):
        '''
        Independently align each scintillator to the datalogger. Every scintillator dataframe preserves ALL of its original rows.
        Datalogger columns are merged onto each scintillator independently using nearest-neighbour timestamp matching, creating a new aligned_scint{i} dataframe for each scintillator which contains the original scintillator columns plus the datalogger columns.
        The datalogger columns are only filled in for rows where a match was found within the specified tolerance (default 0.5 seconds).
        '''

        # Create a copy of the datalogger dataframe sorted by Absolute Timer (S) for merging
        dl_df = datalogger_df.sort_values('Absolute Timer (S)').reset_index(drop=True)

        # Add columns to the datalogger dataframe giving the increase in double and triple coincidence events
        for k in self.coinc_orders:
            src = self._coinc_src_col(k)
            tag = self._coinc_tag(k)
            dl_df[f'delta_{tag}'] = dl_df[src].diff().clip(lower=0).fillna(0)

        # ── Independently align each scintillator ──────────────────────────
        for i in range(1, len(self.fps) + 1): # loop through the scintillators

            print(f"\nAligning scintillator {i}...")

            # create scintillator dataframe with only subset of the full set of colummns
            scint_df = getattr(self, f'scint_{i}')[['Time[s]', 'SiPM[mV]', 'ADC[0-4095]','Coincident[bool]', 'Deadtime[s]']].copy()

            # apply deadtime corrections using the apply_deadtime_correction method, add the live times to the scintillator data
            livetime_series = self.apply_deadtime_correction(i)
            scint_df[f'livetime_scint{i}[s]'] = livetime_series.values

            # rename columns to include scintillator index, for clarity when merging with datalogger; also sort by timer column
            scint_df = scint_df.rename(columns={
                'Time[s]':          f'Time_scint{i}[s]',
                'SiPM[mV]':         f'SiPM_scint{i}[mV]',
                'ADC[0-4095]':      f'ADC_scint{i}[0-4095]',
                'Coincident[bool]': f'Coincident_scint{i}[bool]',
                'Deadtime[s]':      f'Deadtime_scint{i}[s]',
            }).sort_values(f'Time_scint{i}[s]').reset_index(drop=True)

            # ── Merge datalogger INTO scintillator ─────────────────────────
            # scintillator is LEFT table → every scintillator row preserved
            aligned = pd.merge_asof(scint_df, dl_df,
                left_on=f'Time_scint{i}[s]',
                right_on='Absolute Timer (S)',
                direction='nearest',
                tolerance=tolerance
            )

            # ── Diagnostics ────────────────────────────────────────────────
            if self.debug:
                aligned[f'dt_match_scint{i}[s]'] = (
                    aligned['Absolute Timer (S)']
                    - aligned[f'Time_scint{i}[s]']).abs()

                print(f"Aligned dataframe shape: {aligned.shape}")

                print(f"\nScintillator {i} match offset stats (seconds):")
                print(aligned[f'dt_match_scint{i}[s]'].describe().round(4))

            # ── Total livetime ─────────────────────────────────────────────
            total_livetime_s = (aligned[f'livetime_scint{i}[s]'].clip(lower=0).fillna(0).sum())

            print(f"Total run livetime "
                f"(deadtime-corrected, scint {i}): "
                f"{total_livetime_s:.2f} s")

            # ── Run-averaged rates ─────────────────────────────────────────
            for k in self.coinc_orders:
                tag = self._coinc_tag(k)                    # 2 -> 'CW12', 3 -> 'CW123'
                label = '+'.join(str(j) for j in range(1, k + 1))   # 2 -> '1+2', 3 -> '1+2+3'
                rate = aligned[f'delta_{tag}'].fillna(0).sum() / total_livetime_s
                print(f"Run-averaged CW{label} rate: {rate:.3f} s^-1")

            # ── Per-order coincidence tagging ─────────────────────────────
            for k in self.coinc_orders:
                tag = self._coinc_tag(k)
                aligned[f'SiPM_mV_{tag}_scint{i}'] = (aligned[f'SiPM_scint{i}[mV]'].where(aligned[f'delta_{tag}'] > 0)) # create a new column for each coincidence order, where the SiPM voltage is only filled in for rows where the corresponding delta_CW{tag} is greater than 0 (i.e., a coincidence event occurred), to be used later for MIP normalization and cross-scintillator analysis

            # ── Store aligned dataframe for given scintillator, setting as a class attribute ─────────────
            setattr(self, f'aligned_scint{i}', aligned)
            setattr(self, f'total_livetime_scint{i}_s', total_livetime_s)

            print(f"Stored as self.aligned_scint{i}") if self.debug else None

    def plot_all_SiPM_distributions(self):
        '''Combined SiPM voltage distribution across all scintillators.'''
        if not (self.show_plots or self.save_plots):
            return

        colors = ['blue', 'orange', 'green', 'red', 'purple']
        fig, ax = plt.subplots(figsize=(8, 4))

        for i in range(1, len(self.fps) + 1):
            df = getattr(self, f'aligned_scint{i}')
            cw123_data = df[f'SiPM_mV_{self.top_tag}_scint{i}'].dropna()

            if self.debug:
                print(f"\nScintillator {i} — SiPM_mV_{self.top_tag} stats:")
                print(f" Non-null count: {len(cw123_data)}")
                print(f" Min: {cw123_data.min():.2f} mV, Max: {cw123_data.max():.2f} mV, "
                    f"Mean: {cw123_data.mean():.2f} mV")
            ax.hist(cw123_data, bins=100, color=colors[(i-1) % len(colors)],
                    edgecolor='black', alpha=0.5, label=f'Scintillator {i}')

        ax.set_xlabel('SiPM[mV]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('SiPM Voltage Distribution — All Scintillators', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._finish_mpl(fig, "sipm_all_scints")
        # plt.close(fig)

    def plot_SiPM_histograms(self, i, bins=100):
        if not (self.show_plots or self.save_plots):
            return
        
        df = getattr(self, f'aligned_scint{i}')

        all_events = df[f'SiPM_scint{i}[mV]'].dropna()

        palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        data    = [all_events]
        labels  = ['All events']
        colors  = [palette[0]]

        # One entry per coincidence order (CW12, CW123, ..., up to CW1...N)
        fold_names = {2: 'Double', 3: 'Triple', 4: 'Quadruple', 5: 'Quintuple', 6: 'Sextuple'}
        for idx, k in enumerate(self.coinc_orders, start=1):
            tag = self._coinc_tag(k)                                    # 'CW123', etc.
            events_k = df.loc[df[f'delta_{tag}'] > 0, f'SiPM_scint{i}[mV]'].dropna()
            data.append(events_k)
            fold = fold_names.get(k, f'{k}-fold')
            labels.append(f'{fold} coincidence events ({tag})')
            colors.append(palette[idx % len(palette)])

        fig, ax = plt.subplots(figsize=(8, 4))

        for dataset, color, label in zip(data, colors, labels):

            ax.hist(dataset,bins=bins,color=color,edgecolor='black',alpha=0.5,label=label)

        ax.set_xlabel('SiPM [mV]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'SiPM Voltage Distribution — Scintillator {i}',fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._finish_mpl(fig, f"sipm_scint{i}")
        # plt.close(fig)

    # def _finish_mpl(self, fig, name):
    #     if self.save_plots:
    #         path = os.path.join(self.results_dir, f"{name}.png")
    #         fig.savefig(path, dpi=150, bbox_inches='tight')
    #         print(f"[saved] {path}")
    #         plt.close(fig)
    #     else:
    #         plt.show()








class Detector_Analysis(PlotOutput):
    def __init__(self, processor, datalogger_df, debug=False, results_dir=None, show_plots=None):
        '''
        Class for analyzing the CosmicWatch data after the Datalogger_Processing() and Scintillator_Processing() classes have been run. Pass a filepath for results_dir in order for results to be saved class-wide, otherwise results just appear in interactive window.
        This class takes the datalogger dataframe and the instance of the Scintillators_Processing class as inputs, and has the following methods:

            Two top-level pipelines (call one of these directly; each runs the full analysis/plotting chain):
                - calibrate_and_analyze_grounddata(moyal_fit_ranges): fits a Moyal distribution per scintillator to extract the MPV, then runs the full chain using those fitted MPVs
                - analyze_calibrated_data_with_fixed_MPVs(MPVs, noise_threshold, mip_window): same chain, but uses externally supplied per-scint MPVs (e.g. from a prior ground/temperature calibration) instead of fitting; no Moyal fit is performed, so the dashed fit curve is absent from every panel

            Methods blocks called by the pipelines above:
                - build_master_df(tolerance, MPVs): builds a master dataframe anchored on the datalogger timeline, merging each scintillator onto it independently by nearest-neighbour timestamp. Adds MIP-normalized, amplitude-calibrated, and cross-scintillator mean/std columns; also computes and stores the calibration constants (global_mean_mpv, amp_shifts) that the plotting methods read back
                - fit_moyal(centers, rates, fit_x_min, fit_x_max, fit_x_n): fits a Moyal distribution to a single rate spectrum, returning the fit curve, a legend label, and the (MPV, eta, A) parameters (or None on failure)
                - fill_between_steps(x, y1, y2, h_align, ax, lw, **kwargs): draws a fill band as horizontal step segments matching the histogram bin structure, rather than a smooth interpolation
                - plot_rate_spectra_preview(moyal_fit_ranges): per-scint rate spectra (all / non-coincident / coincident events) with Poisson uncertainty bands, plus a lower ratio panel showing each population's fraction of all events
                - fit_and_normalize_spectra(moyal_fit_ranges, noise_threshold, MPVs, no_fit): coincident rate spectra (top row) and their MIP-normalized versions (bottom row, x-axis rescaled mV → MIP by each scint's MPV); stores per-scint fit/histogram results in self.scintillator_moyal_fit_results for later methods to reuse
                - calibration_comparison(moyal_fit_ranges, noise_threshold): 3-row figure (raw MIP-normalized → + amplitude calibration → + amplitude and rate calibration) used to diagnose whether cross-scint detector differences are additive (overlay everywhere after the shift) or multiplicative (agree at 1 MIP but fan out off-peak)
                - two_dimensional_histograms(): 2D density heatmap of cross-scint spread (std) vs. mean amplitude for top-fold coincidence events, MIP-normalized and rate-normalized by livetime
                - MIPregion_two_dimensional_histograms(mip_window): side-by-side spread-vs-mean heatmaps, uncalibrated vs amplitude-calibrated, zoomed to the MIP peak region to show whether the calibration tightens the cross-scint spread

            Key attributes created (in the order the methods run):
                - self.coinc_tag, self.coinc_col : top-fold coincidence names built from the scint count (e.g. 'CW123', 'delta_CW123'); set in __init__
                - self.scintillator_moyal_fit_results          : list of per-scint dicts (rate, bin edges/centers, Moyal fit, popt with MPV = popt[0]); set in fit_and_normalize_spectra
                - self.noise_threshold : MIP threshold separating noise from signal; set in fit_and_normalize_spectra
                - self.master_df       : master dataframe on the datalogger timeline, with all normalized/calibrated/cross-scint columns; set in build_master_df
                - self.mpv_per_scint, self.global_mean_mpv, self.amp_shifts : per-scint MPVs, their mean, and the per-scint mV shifts used for amplitude calibration; set in build_master_df
        '''
        self._init_output(results_dir, show_plots)

        self.processor = processor
        self.datalogger_df = datalogger_df
        self.moyal_fit_ranges = None
        self.scintillator_moyal_fit_results = None
        # NEW: build the coincidence column/tag from the number of scintillators
        n = len(processor.fps)
        self.coinc_tag = 'CW' + ''.join(str(i) for i in range(1, n + 1))  # 'CW123', 'CW1234', ...
        self.coinc_col = f'delta_{self.coinc_tag}'                         # 'delta_CW123', 'delta_CW1234', ...

        self.debug = debug

    def calibrate_and_analyze_grounddata(self, moyal_fit_ranges=None, twodim_hist_args=None, noise_threshold=0.1):
        """
        Fit a Moyal distribution per scintillator to determine and extract the MPV, then run the full pipeline using those fitted MPVS.

        twodim_hist_args : optional dict passed through to two_dimensional_histograms,
                            e.g. {'col': 'MIP_ampcal', 'cbar_max': 300e-6}. If None, uses
                            two_dimensional_histograms' own defaults (col='MIP', cbar_max=None).
        """
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        if self.moyal_fit_ranges is None:
            raise ValueError("moyal_fit_ranges not set — pass it to this method or set self.moyal_fit_ranges first.")

        self.plot_rate_spectra_preview(self.moyal_fit_ranges)
        self._run_pipeline(twodim_hist_args=twodim_hist_args, noise_threshold=noise_threshold)

    def analyze_calibrated_data_with_fixed_MPVs(self, MPVs, noise_threshold=0.1, mip_window=(0.8, 1.2), twodim_hist_args=None):
        """
        Same full plotting pipeline as calibrate_and_analyze_grounddata, but using externally
        supplied per-scint MPVs (e.g. from a prior ground/temperature calibration)
        instead of fitting a Moyal per run. No fit is performed → the dashed Moyal
        curve is absent from every panel; everything else (rate spectra, MIP rescaling,
        amp + rate calibration, density + std heatmaps) is produced identically.

        MPVs             : list of per-scint MPVs in mV, ordered scint1..scintN
        mip_window       : (lo, hi) MIP zoom window for the std heatmaps, replacing the
                            fit-range auto-zoom (which needs a fit to derive).
        twodim_hist_args : optional dict passed through to two_dimensional_histograms,
                            e.g. {'col': 'MIP_ampcal', 'cbar_max': 300e-6}. If None, uses
                            two_dimensional_histograms' own defaults (col='MIP', cbar_max=None).
        """
        self.plot_rate_spectra_preview()
        self._run_pipeline(MPVs=MPVs, noise_threshold=noise_threshold, mip_window=mip_window, twodim_hist_args=twodim_hist_args)

    def _run_pipeline(self, MPVs=None, noise_threshold=0.1, mip_window=None, twodim_hist_args=None):
        self.fit_and_normalize_spectra(MPVs=MPVs, noise_threshold=noise_threshold, no_fit=(MPVs is not None))
        self.build_master_df(MPVs=MPVs)
        self.calibration_comparison()
        if twodim_hist_args is None:
            twodim_hist_args = {}
        col      = twodim_hist_args.get('col', 'MIP')
        cbar_max = twodim_hist_args.get('cbar_max', None)
        self.two_dimensional_histograms(col=col, cbar_max=cbar_max)
        self.MIPregion_two_dimensional_histograms(mip_window=mip_window)

    def build_master_df(self, tolerance=0.5, MPVs=None):
        '''
        Build a master dataframe anchored on the datalogger timeline.
        Each scintillator is merged independently (no scint-to-scint chaining).
        MIP-normalized columns are added using the per-scintillator MPVs read
        from self.scintillator_moyal_fit_results (the Moyal fits produced by fit_and_normalize_spectra).
        Global-mean amplitude-calibrated columns are also added for the top fold.

        Inputs (read from self, no arguments):
            tolerance: float
                merge_asof tolerance in seconds (default 0.5).

        Side effects:
            self.master_df        : the assembled master dataframe
            self.mpv_per_scint    : {scint_index: MPV_mV}
            self.global_mean_mpv  : mean of the per-scint MPVs
            self.amp_shifts       : per-scint additive mV shift (0-based list)

        Parameters
        ----------
        tolerance : float
            merge_asof tolerance in seconds (default 0.5).
        '''

        # Sort the datalogger by the absolute timer and reset the index for merging
        dl_df = self.datalogger_df.sort_values('Absolute Timer (S)').reset_index(drop=True)
        for k in self.processor.coinc_orders:
            src = self.processor._coinc_src_col(k)
            tag = self.processor._coinc_tag(k)
            dl_df[f'delta_{tag}'] = dl_df[src].diff().clip(lower=0).fillna(0)
        master = dl_df.copy()

        # ── Independently merge each scintillator onto the datalogger ────
        for i in range(1, len(self.processor.fps) + 1): # loop through each of the N scintillator
            scint_raw = getattr(self.processor, f'scint_{i}')[['Time[s]', 'SiPM[mV]', 'ADC[0-4095]']].sort_values('Time[s]').reset_index(drop=True) # select for the relevant columns from the scintillator data
            scint_raw = scint_raw.rename(columns={ # rename columns for clarity
                'Time[s]':     f'Time_scint{i}[s]',
                'SiPM[mV]':    f'SiPM_scint{i}[mV]',
                'ADC[0-4095]': f'ADC_scint{i}[0-4095]',
            })
            master = pd.merge_asof(
                master,
                scint_raw,
                left_on='Absolute Timer (S)',
                right_on=f'Time_scint{i}[s]',
                direction='nearest',
                tolerance=tolerance
            )

        # ── Tag CW12 and CW123 SiPM values per scintillator (mV) ────────
        for i in range(1, len(self.processor.fps) + 1):
            for k in self.processor.coinc_orders:
                tag = self.processor._coinc_tag(k)
                master[f'SiPM_mV_{tag}_scint{i}'] = (
                    master[f'SiPM_scint{i}[mV]'].where(master[f'delta_{tag}'] > 0))

        # Creating dictionary of MPVs per scintillator
        mpv_per_scint = {i: self._resolve_mpv(i, MPVs) for i in range(1, len(self.processor.fps) + 1)} # dictionary of MPVs per scintillator found using the _resolve_mpv helper function


        # ── MIP-normalized per-channel columns ──────────────────────────
        for i in range(1, len(self.processor.fps) + 1):
            mpv_i = mpv_per_scint[i]
            for k in self.processor.coinc_orders:
                tag = self.processor._coinc_tag(k)
                master[f'SiPM_MIP_{tag}_scint{i}'] = master[f'SiPM_mV_{tag}_scint{i}'] / mpv_i

        # ── Cross-scintillator derived columns in mV (triple coincidence) ─
        top = self.processor.top_tag # top_tag is either CW123 or CW1234, or however many CWs you have, and indicates triple coincidence
        sipm_top_mv_cols = [f'SiPM_mV_{top}_scint{i}'  for i in range(1, len(self.processor.fps) + 1)]
        master['SiPM_scints_avg']  = master[sipm_top_mv_cols].mean(axis=1, skipna=False)
        master['SiPM_scints_std']  = master[sipm_top_mv_cols].std(axis=1, ddof=1, skipna=False)

        sipm_top_mip_cols = [f'SiPM_MIP_{top}_scint{i}' for i in range(1, len(self.processor.fps) + 1)]
        master['SiPM_scints_avg_MIP'] = master[sipm_top_mip_cols].mean(axis=1, skipna=False)
        master['SiPM_scints_std_MIP'] = master[sipm_top_mip_cols].std(axis=1, ddof=1, skipna=False) # standard deviation across the SiPM_MIP_{top_tag}_scint1, SiPM_MIP_{top_tag}_scint2, and SiPM_MIP_{top_tag}_scint3 --- std of the MIP amplitude on each scintillator from the mean

        for i in range(1, len(self.processor.fps)):
            master[f'SiPM_diff_scint{i}_minus_scint{i+1}'] = (
                master[f'SiPM_mV_{top}_scint{i}'] - master[f'SiPM_mV_{top}_scint{i+1}'])

        # ── Global-mean amplitude calibration (top fold) ─────────────────
        self.global_mean_mpv, self.amp_shifts = self._compute_calibration(mpv_per_scint)
        global_mean_mpv = self.global_mean_mpv

        for i in range(1, len(self.processor.fps) + 1):
            amp_shift = global_mean_mpv - mpv_per_scint[i] # finding offset of each individual scintillator's MPV from global mean
            master[f'SiPM_mV_{top}_scint{i}_ampcal'] = (master[f'SiPM_mV_{top}_scint{i}'] + amp_shift) # uniformly shifting all SiPM mV values on that scint by the offset from the mean
            master[f'SiPM_MIP_{top}_scint{i}_ampcal'] = (master[f'SiPM_mV_{top}_scint{i}_ampcal'] / global_mean_mpv) # calibrated MIP: shifted mV ÷ global mean MPV (common divisor, so every shifted peak lands at 1.0)

        # ── Cross-scintillator calibrated MIP spread (the column the heatmap payoff lives in) ─
        sipm_top_mip_cal_cols = [f'SiPM_MIP_{top}_scint{i}_ampcal' for i in range(1, len(self.processor.fps) + 1)]
        master['SiPM_scints_avg_MIP_ampcal'] = master[sipm_top_mip_cal_cols].mean(axis=1, skipna=False)
        master['SiPM_scints_std_MIP_ampcal'] = master[sipm_top_mip_cal_cols].std(axis=1, ddof=1, skipna=False)


        self.master_df = master
        self.mpv_per_scint = mpv_per_scint
        print(f"Master dataframe shape: {master.shape}")
        print(f"MPVs used: {mpv_per_scint}")
        print(f"Columns: {master.columns.tolist()}")
        # self._save_table(master, "master_df")
        n = len(self.processor.fps)
        self._save_summary({
            f"{self.processor.top_tag}_rate_hz":        master[f'delta_{self.processor.top_tag}'].sum() / self._mean_livetime(),
            "mpv_per_scint":        self.mpv_per_scint,
            "global_mean_mpv":      self.global_mean_mpv,
            "amp_shifts":           self.amp_shifts,
            "noise_threshold":      getattr(self, "noise_threshold", None),
            "livetime_s_per_scint": [getattr(self.processor, f"total_livetime_scint{i}_s", None)
                                     for i in range(1, n + 1)],
        }, "calibration_summary")

    # ── Moyal fit function ──────────────────────────────────────────────────
    def fit_moyal(self, centers, rates, fit_x_min=40, fit_x_max=120, fit_x_n=300):
        """
        Fits a Moyal distribution to a rate spectrum.

        Args:
            centers:   array of bin center SiPM voltages [mV]
            rates:     array of differential rates [s^-1 per bin], aligned to centers
            fit_x_min: lower bound of fit range [mV]
            fit_x_max: upper bound of fit range [mV]
            fit_x_n:   number of points in the fit curve

        Returns:
            fit_x:          x values of the fitted curve (or None on failure)
            moyal_fit_line: y values of the fitted curve (or None on failure)
            moyal_label:    legend label string with fit params (or None on failure)
            popt:           (mpv, eta, A) fit parameters (or None on failure)
        """

        def moyal(x, mpv, eta, A):
            return A * np.exp( -0.5 * ( ((x - mpv) / eta) + np.exp(-(x - mpv) / eta)) )

        fit_x = np.linspace(fit_x_min, fit_x_max, fit_x_n)
        # creating x values over which the Moyal fit will be evaluated

        rates_filled = np.nan_to_num(rates, nan=0.0)
        # replacing NaNs with zeros before interpolation

        fit_y = np.interp(fit_x, centers, rates_filled)
        # interpolating the binned rate spectrum onto a smooth x grid

        try:

            p0 = [centers[rates_filled.argmax()], 15, rates_filled.max()] # MPV initial guess; eta initial guess; amplitude initial guess
            popt, _ = curve_fit(moyal,fit_x,fit_y,p0=p0, maxfev=10000)
            moyal_fit_line = moyal(fit_x, *popt)
            moyal_label = (rf'Moyal Fit '
                            rf'($\mu={popt[0]:.1f}$ mV, '
                            rf'$\eta={popt[1]:.1f}$ mV)')
            print(f"Moyal fit — "
                f"MPV: {popt[0]:.2f} mV, "
                f"eta: {popt[1]:.2f} mV, "
                f"A: {popt[2]:.4f}"
            )

            return fit_x, moyal_fit_line, moyal_label, popt

        except RuntimeError:

            print(
                "Moyal fit did not converge — "
                "try adjusting fit_x_min/fit_x_max or p0"
            )

            return None, None, None, None
    
    def _resolve_mpv(self, scint_idx, MPVs=None):
        '''
        Resolve the MPV for an individual scintillators, which are either passed in as an argument or generated in the script
        '''
        # Use the list of MPVs if provided (which should be provided in the build_master_df and fit_and_normalize_spectra methods if the user has indicated that they want to use externally supplied MPVs instead of fitting a Moyal distribution to the data)
        if MPVs is not None:
            return MPVs[scint_idx - 1]
        return self.scintillator_moyal_fit_results[scint_idx - 1]['popt'][0]
    
    def _compute_calibration(self, mpv_per_scint):
        '''
        Helper method for computing the calibration constant (global mean MPV for a stack of scintillators) using the individual MPVs per scintillator
        stored in the class (either as an arugment loaded in from a previous ground run)
        '''
        global_mean_mpv = np.mean(list(mpv_per_scint.values()))
        amp_shifts = [global_mean_mpv - mpv_per_scint[i] for i in range(1, len(mpv_per_scint) + 1)]
        return global_mean_mpv, amp_shifts
    
    def _mean_livetime(self):
        n = len(self.processor.fps)
        return float(np.mean([getattr(self.processor, f'total_livetime_scint{i}_s') for i in range(1, n + 1)]))

    # ── Raw spectra + MIP-normalized, combined ───────────────────────────────
    def fit_and_normalize_spectra(self, moyal_fit_ranges=None, noise_threshold=0.1, MPVs=None, no_fit=False):
        '''
        Build the per-scintillator coincident rate spectra and their MIP-normalized
        versions, and cache the fit/histogram results other methods read back.

        Produces a 2-row × N-scint figure:
            Row 0: coincident rate spectrum per scint [s^-1 per bin] vs SiPM mV,
                with a Poisson error band and (unless no_fit) a dashed Moyal
                fit overlay.
            Row 1: same spectra rescaled onto a MIP x-axis (mV ÷ MPV), split into
                above/below noise_threshold, with a vertical line at 1 MIP.

        Two ways to get a MPV per scintillator, determined by the no_fit argument:
            - Fit a Moyal to each scint's spectrum via fit_moyal(), which is the default (no_fit=False).
                Uses moyal_fit_ranges[i] as the (min, max) mV window.
                moyal_fit_ranges must be supplied here or already set on self.
            - Use given MPVs (no_fit=True), which means skipping the fit entirely and uses MPVs[i] directly.

        Args:
            moyal_fit_ranges: list of (fit_x_min, fit_x_max) per scint, in mV.
                Ignored if no_fit=True. If None and no_fit=False, falls back to
                self.moyal_fit_ranges (raises if that's also unset).
            noise_threshold: MIP value separating "noise" from "signal" on the
                row-1 plots; also cached to self.noise_threshold for later use
                by the heatmap methods.
            MPVs: list of per-scint MPVs in mV, ordered scint1..scintN. Required
                if no_fit=True; ignored otherwise.
            no_fit: if True, use MPVs instead of fitting (see above).

        Side effects:
            self.scintillator_moyal_fit_results          : list of per-scint dicts (rate, err, bin_edges,
                                    bin_centers, fit_x, moyal_y, peak_rate, popt),
                                    one entry per scintillator. popt = [mpv] when
                                    no_fit=True, or (mpv, eta, A) from the fit.
            self.noise_threshold : set from the noise_threshold argument.
            self.moyal_fit_ranges : set from moyal_fit_ranges if provided and
                                    no_fit=False.

        Raises:
            ValueError: if no_fit=False and no fit range is available (neither
                passed in nor previously set on self).

        Note:
            Downstream methods (build_master_df, calibration_comparison,
            two_dimensional_histograms, MIPregion_two_dimensional_histograms) all read
            self.scintillator_moyal_fit_results, so this must run before any of them — _run_pipeline()
            already enforces that ordering.
        '''
        if not no_fit:
            if moyal_fit_ranges is not None:
                self.moyal_fit_ranges = moyal_fit_ranges
            if self.moyal_fit_ranges is None:
                raise ValueError("moyal_fit_ranges not set — pass it to this method or set self.moyal_fit_ranges first.")
            moyal_fit_ranges = self.moyal_fit_ranges

        self.noise_threshold = noise_threshold

        x = len(self.processor.fps)

        fig, axes_all = plt.subplots(2, x, figsize=(6 * x, 10), squeeze=False)
        axes_top = axes_all[0]
        stored = []
        colors = ['teal', 'darkorange', 'steelblue']


        for scint_idx, ax in enumerate(axes_top, start=1):
            df_scint = getattr(self.processor, f'aligned_scint{scint_idx}')
            livetime = getattr(self.processor, f'total_livetime_scint{scint_idx}_s')

            col = f'SiPM_scint{scint_idx}[mV]'
            data = df_scint.loc[df_scint[self.coinc_col] > 0, col].dropna().values

            bin_edges = np.logspace(np.log10(max(data.min(), 0.1)), np.log10(data.max()), 51)
            counts, _ = np.histogram(data, bins=bin_edges)
            rate = counts / livetime
            err  = np.sqrt(counts) / livetime
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

            if no_fit:
                # No fit: supplied MPV stands in for the fitted one. popt=[mpv] keeps
                # downstream code (which reads popt[0]) working; fit_x/moyal_y stay None
                # so the dashed curve is suppressed everywhere.
                mpv = MPVs[scint_idx - 1]
                fit_x, moyal_y, label, popt = None, None, None, [mpv]
            else:
                FIT_X_MIN, FIT_X_MAX = moyal_fit_ranges[scint_idx - 1]
                fit_x, moyal_y, label, popt = self.fit_moyal(bin_centers, rate, FIT_X_MIN, FIT_X_MAX)

            ax.stairs(rate, bin_edges, color=colors[(scint_idx - 1) % len(colors)], linewidth=1.5)
            self.fill_between_steps(bin_centers, rate + err, rate - err,
                                    color=colors[(scint_idx - 1) % len(colors)], alpha=0.3, ax=ax)
            if moyal_y is not None:
                ax.plot(fit_x, moyal_y, 'k--', linewidth=2, label=label)
                ax.legend(fontsize=10)

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('SiPM Peak Voltage [mV]', fontsize=12)
            ax.set_title(f'Scintillator {scint_idx}', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)

            peak_rate = max(moyal_y) if moyal_y is not None else None
            stored.append({'rate': rate, 'err': err, 'bin_edges': bin_edges, 'bin_centers': bin_centers,
                        'fit_x': fit_x, 'moyal_y': moyal_y, 'peak_rate': peak_rate, 'popt': popt})

        axes_top[0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12)

        self.scintillator_moyal_fit_results = stored


        # ── Normalized to 1 MIP ─────────────────────────────────────────────────
        axes = axes_all[1]
        colors = ['teal', 'darkorange', 'steelblue']
        colors_dim = ['lightgray', 'bisque', 'lightblue']

        # i=0
        for scint_idx, ax in enumerate(axes):
            s = stored[scint_idx]
            if s['popt'] is None:          # real fit failure only; no_fit sets popt=[mpv]
                ax.set_title(f'Scintillator {scint_idx+1} — fit failed', fontsize=13)
                continue

            mpv = self._resolve_mpv(scint_idx + 1, MPVs)

            # Rescale x-axis: mV → MIP
            bin_edges_mip   = s['bin_edges']   / mpv
            bin_centers_mip = s['bin_centers'] / mpv
            fit_x_mip = s['fit_x'] / mpv if s['fit_x'] is not None else None   # guard: no fit → no curve x

            # Split at specified MIP — now a constant on the rescaled axis
            below_mask = bin_centers_mip < noise_threshold
            above_mask = ~below_mask
            rate_below = np.where(below_mask, s['rate'], np.nan)
            rate_above = np.where(above_mask, s['rate'], np.nan)

            ax.stairs(rate_above, bin_edges_mip, color=colors[scint_idx % len(colors)], linewidth=1.5, label=f'≥ {noise_threshold} MIP')
            ax.stairs(rate_below, bin_edges_mip, color=colors_dim[scint_idx % len(colors_dim)], linewidth=1.5, label=f'< {noise_threshold} MIP')
            if s['moyal_y'] is not None:                                       # guard: suppress dashed curve in no_fit
                ax.plot(fit_x_mip, s['moyal_y'], 'k--', linewidth=2)
            err = s['err']
            self.fill_between_steps(bin_centers_mip, s['rate'] + err, s['rate'] - err,
                                    color=colors[scint_idx % len(colors)], alpha=0.3, ax=ax)

            ax.axvline(1.0, color=colors[scint_idx % len(colors)], linestyle=':', linewidth=1.5)
            ax.text(1.05, 0.5, f'MPV = {mpv:.1f} mV',
                transform=ax.get_xaxis_transform(),
                fontsize=10, color=colors[scint_idx % len(colors)],
                rotation=90, va='center')
            # ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1e-1)
            ax.set_xlim(0, 2.5)
            ax.set_xlabel('Amplitude [MIP]', fontsize=12)   # ← units changed
            ax.set_title(f'Scintillator {scint_idx+1}: 1 MIP = {mpv:.1f} mV', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)
            # i+=1
        axes[0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12) #axes[0].set_ylabel('Rate / Moyal Peak', fontsize=12)
        plt.tight_layout()
        self._finish_mpl(fig, "mip_normalized")

    def calibration_comparison(self, moyal_fit_ranges=None, noise_threshold=None):
        """
        Draws the calibration-comparison figure (3 rows × N scintillators):
        Row 0: raw MIP-normalized spectra (each scint ÷ its own MPV)
        Row 1: + amplitude calibration  (whole spectrum shifted by amp_shift in mV, then ÷ MPV)
        Row 2: + amplitude AND rate calibration (additional uniform rate shift per scint)

        This method PLOTS; it does not compute the calibration constants.
        global_mean_mpv and the per-scint amp_shifts are computed once in
        build_master_df and read from self here, so the figure, the master
        dataframe, and the aligned _cal columns all use identical constants.

        The amplitude shift is applied across the ENTIRE spectrum (all bins),
        not just at the peak — that is what makes this figure the
        additive-vs-multiplicative diagnostic: additive detector differences
        overlay everywhere after the shift; multiplicative differences agree
        at 1 MIP but fan out off-peak.

        When the spectra come from the fixed-MPV (no_fit) path, fit_x/moyal_y are
        None, so the dashed Moyal curve is simply absent; everything else is identical.

        Reads:
            self.scintillator_moyal_fit_results          : per-scint Moyal-fit / histogram dicts (from fit_and_normalize_spectra)
            self.global_mean_mpv : common MPV reference (from build_master_df)
            self.amp_shifts      : per-scint additive mV shift, 0-based (from build_master_df)
        """
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges

        if noise_threshold is None:
            noise_threshold = getattr(self, 'noise_threshold', 0.1)

        if self.scintillator_moyal_fit_results is None:
            raise ValueError("self.scintillator_moyal_fit_results is not set — run fit_and_normalize_spectra() first.")
        stored = self.scintillator_moyal_fit_results

        # ── Calibration constants: READ from build_master_df (single source) ──────
        # build_master_df computes global_mean_mpv + amp_shifts from self.scintillator_moyal_fit_results and
        # stores them on self. We read them here so this figure can't drift from the
        # master dataframe's calibrated columns.
        if not hasattr(self, 'amp_shifts'):
            raise ValueError("amp_shifts not set — run build_master_df() first.")
        global_mean_mpv = self.global_mean_mpv
        amp_shifts      = self.amp_shifts                          # 0-based list, scint 1 at index 0

        x = len(self.processor.fps)
        mpvs = [stored[i]['popt'][0] for i in range(x)]            # per-scint MIP divisor for the plot rescaling (≠ global_mean_mpv)
        print(f"MPVs for each scintillator: {mpvs}")
        print(f"Global mean MPV: {global_mean_mpv}")
        print(f"Amplitude shifts: {amp_shifts}")

        colors      = ['teal', 'darkorange', 'steelblue']
        colors_dim  = ['lightgray', 'bisque', 'lightblue']

        fig, axes_all = plt.subplots(3, x, figsize=(6 * x, 15), squeeze=False)
        axes_raw    = axes_all[0]   # row 0: original MIP-normalized
        axes_amp    = axes_all[1]   # row 1: + amplitude calibration
        axes_both   = axes_all[2]   # row 2: + amplitude AND rate calibration

        # ── Full-resolution calibrated mV column on each aligned frame ────────────
        # KEPT (not redundant with master): master only holds top-fold COINCIDENT
        # values, whereas this shifts EVERY event. It's the only calibrated column
        # at full event resolution — the source for any future all-events calibrated
        # spectrum. Currently unused downstream, but cheap and not recoverable from master.
        for i, shift in enumerate(amp_shifts):
            getattr(self.processor, f'aligned_scint{i+1}')[f'SiPM_scint{i+1}[mV]_cal'] = (getattr(self.processor, f'aligned_scint{i+1}')[f'SiPM_scint{i+1}[mV]'] + shift)

        # ── Row 0: reproduce raw MIP-normalized from stored ─────────────────────
        for scint_idx, ax in enumerate(axes_raw, start=1):
            s = stored[scint_idx - 1]
            if s['popt'] is None:
                ax.set_title(f'Scintillator {scint_idx} — fit failed', fontsize=13)
                continue

            mpv = s['popt'][0]
            bin_edges_mip   = s['bin_edges']   / mpv               # rescale plot x-axis mV → MIP (÷ this scint's own MPV)
            bin_centers_mip = s['bin_centers'] / mpv
            fit_x_mip = s['fit_x'] / mpv if s['fit_x'] is not None else None   # guard

            below_mask = bin_centers_mip < noise_threshold
            ax.stairs(np.where(~below_mask, s['rate'], np.nan), bin_edges_mip,
                    color=colors[(scint_idx-1) % len(colors)], linewidth=1.5, label=f'≥ {noise_threshold} MIP')
            ax.stairs(np.where(below_mask,  s['rate'], np.nan), bin_edges_mip,
                    color=colors_dim[(scint_idx-1) % len(colors_dim)], linewidth=1.5, label=f'< {noise_threshold} MIP')
            if s['moyal_y'] is not None:                           # guard
                ax.plot(fit_x_mip, s['moyal_y'], 'k--', linewidth=2)
            err = s['err']
            self.fill_between_steps(bin_centers_mip, s['rate'] + err, s['rate'] - err,
                                    color=colors[(scint_idx-1) % len(colors)], alpha=0.3, ax=ax)
            ax.axvline(1.0, color=colors[(scint_idx-1) % len(colors)], linestyle=':', linewidth=1.5)
            ax.text(1.05, 0.5, f'MPV = {mpv:.1f} mV',
                    transform=ax.get_xaxis_transform(), fontsize=10,
                    color=colors[(scint_idx-1) % len(colors)], rotation=90, va='center')
            ax.set_xscale('linear'); ax.set_yscale('log')
            ax.set_ylim(1e-4, 1e-1); ax.set_xlim(0, 2.5)
            ax.set_xlabel('Amplitude [MIP]', fontsize=12)
            ax.set_title(f'Scintillator {scint_idx}: 1 MIP = {mpv:.1f} mV\n(no calibration)', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)
        axes_raw[0].set_ylabel('Rate/bin [s⁻¹]', fontsize=12)

        # ── Row 1: amplitude calibration ────────────────────────────────────────
        # Shift the WHOLE spectrum (all bins) by amp_shift in mV, THEN convert to MIP.
        # Divisor here is still the per-scint MPV, so the shifted peak lands at
        # global_mean_MPV / per_scint_MPV ≈ 1.0 for all scints. The off-peak behaviour
        # is the diagnostic: overlay = additive offsets; fan-out = multiplicative gain.
        amp_cal_rates = []   # collect for the rate-calibration step below

        for scint_idx, ax in enumerate(axes_amp, start=1):
            s = stored[scint_idx - 1]
            if s['popt'] is None:
                ax.set_title(f'Scintillator {scint_idx} — fit failed', fontsize=13)
                amp_cal_rates.append(None)
                continue

            mpv        = s['popt'][0]
            amp_shift  = amp_shifts[scint_idx - 1]
            print(f"Scintillator {scint_idx}: amp shift = {amp_shift:+.1f} mV")

            # Shift bin centers/edges in mV across the full spectrum, then convert to MIP
            bin_edges_mip   = (s['bin_edges']   + amp_shift) / mpv
            bin_centers_mip = (s['bin_centers'] + amp_shift) / mpv
            fit_x_mip = (s['fit_x'] + amp_shift) / mpv if s['fit_x'] is not None else None   # guard

            rate = s['rate']
            amp_cal_rates.append(rate)   # rate unchanged in row 1; only the x-axis was shifted

            below_mask = bin_centers_mip < noise_threshold
            ax.stairs(np.where(~below_mask, rate, np.nan), bin_edges_mip,
                    color=colors[(scint_idx-1) % len(colors)], linewidth=1.5, label=f'≥ {noise_threshold} MIP')
            ax.stairs(np.where(below_mask,  rate, np.nan), bin_edges_mip,
                    color=colors_dim[(scint_idx-1) % len(colors_dim)], linewidth=1.5, label=f'< {noise_threshold} MIP')
            if s['moyal_y'] is not None:                           # guard
                ax.plot(fit_x_mip, s['moyal_y'], 'k--', linewidth=2)
            err = s['err']
            self.fill_between_steps(bin_centers_mip, rate + err, rate - err,
                                    color=colors[(scint_idx-1) % len(colors)], alpha=0.3, ax=ax)
            ax.axvline(1.0, color=colors[(scint_idx-1) % len(colors)], linestyle=':', linewidth=1.5)
            ax.text(1.05, 0.5, f'amp shift = {amp_shift:+.1f} mV',
                    transform=ax.get_xaxis_transform(), fontsize=10,
                    color=colors[(scint_idx-1) % len(colors)], rotation=90, va='center')
            ax.set_xscale('linear'); ax.set_yscale('log')
            ax.set_ylim(1e-4, 1e-1); ax.set_xlim(0, 2.5)
            ax.set_xlabel('Amplitude [MIP] (amp-calibrated)', fontsize=12)
            peak_mip = (mpv + amp_shift) / global_mean_mpv   # = 1.00 by construction
            ax.set_title(f'Scintillator {scint_idx}: amp shift = {amp_shift:+.1f} mV\n'
                        f'({mpv:.1f} {amp_shift:+.1f}) / {global_mean_mpv:.1f} = {peak_mip:.2f} MIP',
                        fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            # Store calibrated bin geometry back into stored (row 2 reuses it)
            stored[scint_idx - 1]['bin_edges_amp_cal_mip']   = bin_edges_mip
            stored[scint_idx - 1]['bin_centers_amp_cal_mip'] = bin_centers_mip
            stored[scint_idx - 1]['amp_shift_mV']            = amp_shift

        axes_amp[0].set_ylabel('Rate/bin [s⁻¹]', fontsize=12)

        # ── Row 2: amplitude + rate calibration ─────────────────────────────────
        # rate-calibration constants are computed HERE (nothing else computes them):
        # mean rate per scint = mean across all 50 bins of the coincident rate spectrum.
        mean_rates    = [r.mean() for r in amp_cal_rates if r is not None]
        global_mean_rate = np.mean(mean_rates)
        rate_shifts   = [global_mean_rate - r.mean() for r in amp_cal_rates if r is not None]

        rate_shift_iter = iter(rate_shifts)

        for scint_idx, ax in enumerate(axes_both, start=1):
            s = stored[scint_idx - 1]
            if s['popt'] is None:
                ax.set_title(f'Scintillator {scint_idx} — fit failed', fontsize=13)
                continue

            amp_shift  = amp_shifts[scint_idx - 1]
            rate_shift = next(rate_shift_iter)

            bin_edges_mip   = s['bin_edges_amp_cal_mip']           # reuse amplitude-shifted geometry from row 1
            bin_centers_mip = s['bin_centers_amp_cal_mip']
            fit_x_mip = (s['fit_x'] + amp_shift) / s['popt'][0] if s['fit_x'] is not None else None   # guard

            rate_cal = s['rate'] + rate_shift                      # additive shift in linear rate space
            rate_cal_clipped = np.clip(rate_cal, 0, None)          # clip negatives so log plot doesn't choke

            below_mask = bin_centers_mip < noise_threshold
            ax.stairs(np.where(~below_mask, rate_cal_clipped, np.nan), bin_edges_mip,
                    color=colors[(scint_idx-1) % len(colors)], linewidth=1.5, label=f'≥ {noise_threshold} MIP')
            ax.stairs(np.where(below_mask,  rate_cal_clipped, np.nan), bin_edges_mip,
                    color=colors_dim[(scint_idx-1) % len(colors_dim)], linewidth=1.5, label=f'< {noise_threshold} MIP')
            if s['moyal_y'] is not None:                           # guard
                ax.plot(fit_x_mip, s['moyal_y'] + rate_shift, 'k--', linewidth=2)
            err = s['err']
            self.fill_between_steps(bin_centers_mip,
                                    np.clip(rate_cal + err, 0, None),
                                    np.clip(rate_cal - err, 0, None),
                                    color=colors[(scint_idx-1) % len(colors)], alpha=0.3, ax=ax)
            ax.axvline(1.0, color=colors[(scint_idx-1) % len(colors)], linestyle=':', linewidth=1.5)
            ax.text(1.05, 0.5, f'amp={amp_shift:+.1f} mV  rate={rate_shift:+.4f} s⁻¹',
                    transform=ax.get_xaxis_transform(), fontsize=9,
                    color=colors[(scint_idx-1) % len(colors)], rotation=90, va='center')
            ax.set_xscale('linear'); ax.set_yscale('log')
            ax.set_ylim(1e-4, 1e-1); ax.set_xlim(0, 2.5)
            ax.set_xlabel('Amplitude [MIP] (amp + rate calibrated)', fontsize=12)
            ax.set_title(f'Scintillator {scint_idx}: amp={amp_shift:+.1f} mV, rate={rate_shift:+.4f} s⁻¹\n'
                        f'global mean rate = {global_mean_rate:.4f} s⁻¹', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            # Store back
            stored[scint_idx - 1]['rate_cal']   = rate_cal
            stored[scint_idx - 1]['rate_shift'] = rate_shift

        axes_both[0].set_ylabel('Rate/bin [s⁻¹]', fontsize=12)
        plt.suptitle('MIP Spectra: raw → amplitude calibrated → amplitude + rate calibrated\n'
                    f'global mean MPV = {global_mean_mpv:.1f} mV  |  '
                    f'global mean rate = {global_mean_rate:.4f} s⁻¹',
                    fontsize=13)
        plt.tight_layout()
        self._finish_mpl(fig, "calibration_comparison")

    def two_dimensional_histograms(self, col='MIP', cbar_max=None):
        '''
        Wide-view 2D density heatmap: cross-scintillator spread (std) vs. mean amplitude, MIP-normalized, across the FULL amplitude range (no zoom).

        Dataset: rows where the AVERAGE MIP amplitude across scints is above noise_threshold — i.e. "the event's mean signal cleared the noise floor,"
        even if one individual scint's reading is low. No amplitude calibration is applied here; this is the raw (uncalibrated) MIP-normalized spread.

        This is the diagnostic view for seeing the overall shape of cross-scint spread across the entire spectrum (noise floor through MIP peak and beyond) — one heatmap, uncalibrated only. Contrast with
        MIPregion_two_dimensional_histograms(), which restricts to a narrow window around the MIP peak and compares raw vs. amplitude-calibrated side by side.
        
        Args:
            col      : str, suffix of the columns to use (e.g. 'MIP', 'MIP_ampcal', or 'MIP_Tcal'); defaults to 'MIP' (uncalibrated) ie. if you want a 2D histogram of the regular MIP amplitudes, use the default; if you want a 2D histogram of the amplitude-calibrated MIP amplitudes, use 'MIP_ampcal'
            cbar_max : optional fixed upper bound for the colorbar; if None, uses the original fixed 500e-6 default for col='MIP', else auto-scales
        
        '''
        if not hasattr(self, 'master_df'):
            raise ValueError("master_df not set — run build_master_df() first.")
        if not hasattr(self, 'noise_threshold'):
            raise ValueError("noise_threshold not set — run fit_and_normalize_spectra() first.")
        
        master = self.master_df
        n = len(self.processor.fps)

        min_mip = self.noise_threshold

        tag = self.coinc_tag
        coinc_label = '&'.join(str(i) for i in range(1, n + 1))
        # mv_col  = f"SiPM_mV_{tag}_scint1"
        # mip_col = f"SiPM_MIP_{tag}_scint1" # choosing scint1 arbitrarily for the x-axis, since all scints are MIP-normalized and should be comparable

        avg_col, std_col = f"SiPM_scints_avg_{col}", f"SiPM_scints_std_{col}"
        cal_label = {'MIP': 'MIP-normalized', 'MIP_ampcal': 'amplitude-calibrated MIP',
                     'MIP_Tcal': 'temperature-calibrated MIP'}.get(col, col)

        # ── Livetime normalization ───────────────────────────────────────
        # Per-bin counts → per-bin rate [s⁻¹] so heatmaps are comparable across
        # runs of different duration. Coincidence events are cross-scint, so using
        # the mean deadtime-corrected livetime across scintillators
        livetime = self._mean_livetime()
        print(f"[density heatmap] normalizing by livetime = {livetime:.2f} s")

        # ── DEBUG: overview of master + the columns each plot needs ──────
        if self.debug:
            mv_col  = f"SiPM_mV_{tag}_scint1"   # scint1 chosen arbitrarily; all scints are MIP-normalized
            mip_col = f"SiPM_MIP_{tag}_scint1"
            print("=" * 70)
            print(f"DEBUG two_dimensional_histograms | master rows: {len(master)} | min_mip={min_mip}")
            for c in [mv_col, mip_col, avg_col, "SiPM_scints_std", std_col]:
                column = master[c]
                print(f"  {c:30s} non-null={column.notna().sum():5d} "
                    f"min={column.min():.4g} max={column.max():.4g}")
            print("=" * 70)

        # ── MIP-normalized: average vs spread ───────────────────────────
        # Filter is on the AVERAGE MIP across scints, not each scint individually —
        # an event can pass here even if one scint's reading is below noise_threshold,
        # as long as the mean clears it. Contrast with MIPregion_two_dimensional_histograms(),
        # which requires EVERY scint individually >= min_mip.
        sub_avg = master[master[avg_col] >= min_mip].copy() # slicing the master df to only include the rows where the average MIP amplitude across all three scints is above threshold
        sub_avg["_rate_weight"] = 1.0 / livetime
        print(f"[MIP avg]  rows with {avg_col} >= {min_mip}: {len(sub_avg)}")
        print(f"           of those, finite {std_col} (y-axis): "
              f"{sub_avg[std_col].notna().sum()}")
        
        if cbar_max is not None:
            range_color = (0, cbar_max)
        else:
            range_color = None

        fig_mip = px.density_heatmap(
            sub_avg,
            x=avg_col,
            y=std_col,
            # z="_rate_weight",
            histfunc="sum",
            nbinsx=50, nbinsy=50, width=800, height=600, range_color=range_color,
            color_continuous_scale="Inferno",
            labels={
                avg_col: f"Mean across {n} scints [MIP]",
                std_col: f"Std across {n} scints [MIP]",
            },
            title=(f"CW{coinc_label} coincidence: cross-scint spread vs. mean ({cal_label})<br>"
                   f"(N={len(sub_avg)} events)"),
                #    f"(N={len(sub_avg)} events, rate-normalized by livetime = {livetime:.1f} s)"),
        )
        fig_mip.update_coloraxes(colorbar_title="Counts")
        # fig_mip.update_coloraxes(colorbar_title="Normalized Counts")
        self._finish_plotly(fig_mip, "density_heatmap")

        # ── Additional: raw mV (not MIP-normalized) — mean vs spread ─────
        min_mv = min_mip * self.global_mean_mpv   # convert MIP noise threshold to mV so both plots filter consistently
        sub_mv = master[master["SiPM_scints_avg"] >= min_mv].copy()
        print(f"[raw mV]   rows with SiPM_scints_avg >= {min_mv}: {len(sub_mv)}")
        print(f"           of those, finite SiPM_scints_std (y-axis): "
              f"{sub_mv['SiPM_scints_std'].notna().sum()}")
        
        # ── min/max mV across any/all scints in this filtered set ────────
        scint_mv_cols = [f"SiPM_mV_{tag}_scint{i}" for i in range(1, n + 1)]
        mv_min = sub_mv[scint_mv_cols].min().min()
        mv_max = sub_mv[scint_mv_cols].max().max()
        print(f"[raw mV]   per-scint mV range across {scint_mv_cols}: [{mv_min:.4g}, {mv_max:.4g}]")
        
        fig_mv = px.density_heatmap(
            sub_mv,
            x="SiPM_scints_avg",
            y="SiPM_scints_std",
            nbinsx=50, nbinsy=50, width=800, height=600,
            color_continuous_scale="Inferno",
            labels={
                "SiPM_scints_avg": f"Mean across {n} scints [mV]",
                "SiPM_scints_std": f"Std across {n} scints [mV]",},
            title=(f"CW{coinc_label} coincidence: cross-scint spread vs. mean (raw mV)<br>"
                   f"(N={len(sub_mv)} events)"),
        )
        fig_mv.update_coloraxes(colorbar_title="Counts")
        self._finish_plotly(fig_mv, "density_heatmap_mV")

    def MIPregion_two_dimensional_histograms(self, mip_window=None):
        '''
        Zoomed, side-by-side 2D density heatmaps around the MIP peak region:
        uncalibrated (row/col 1) vs. amplitude-calibrated (row/col 2), so the
        effect of amplitude calibration on cross-scint spread can be read off
        directly, panel-to-panel.

        Dataset is filtered more strictly than two_dimensional_histograms():
            1. EVERY individual scint's MIP amplitude must clear noise_threshold
            (not just the average) — a tighter, per-channel noise cut.
            2. Rows are further restricted to a narrow MIP window (mip_window if
            given, else auto-derived from the Moyal fit ranges) — i.e. only
            events actually near the 1-MIP peak, not the full spectrum.

        Because of that second cut, this method answers a different question
        than two_dimensional_histograms(): not "what does spread look like
        across the whole spectrum," but "right at the peak, does amplitude
        calibration tighten the cross-scint spread?" — hence the raw-vs-cal
        pair of panels instead of a single heatmap.
        '''
        if not hasattr(self, 'master_df'):
            raise ValueError("master_df not set — run build_master_df() first.")
        if not hasattr(self, 'noise_threshold'):
            raise ValueError("noise_threshold not set — run fit_and_normalize_spectra() first.")

        master = self.master_df
        n = len(self.processor.fps)
        min_mip = self.noise_threshold
        tag = self.coinc_tag
        coinc_label = '&'.join(str(i) for i in range(1, n + 1))

        # Livetime for rate normalization (mean across scints; see note above)
        livetime = self._mean_livetime()
        print(f"[std heatmap] normalizing by livetime = {livetime:.2f} s")

        avg_raw, std_raw = "SiPM_scints_avg_MIP",     "SiPM_scints_std_MIP"
        avg_cal, std_cal = "SiPM_scints_avg_MIP_ampcal", "SiPM_scints_std_MIP_ampcal"

        # ── Shared row filter ────────────────────────────────────────────
        master = master[master[avg_raw] >= min_mip] # same data cut as the non-zoomed 2D histogram
        # mip_cols = [f"SiPM_MIP_{tag}_scint{i}" for i in range(1, n + 1)]
        # master = master[(master[mip_cols] >= min_mip).all(axis=1)] # Stricter than two_dimensional_histograms(): requires ALL scints individually
        #                                                             # >= min_mip, not just the average — a per-channel noise cut rather than a
        #                                                             # mean-based one, making it a more selective cut of the data

        # ── Zoom window: explicit if given, else auto from fit ranges ────
        # mip_window lets the fixed-MPV (no-fit) path supply the zoom directly, since
        # there are no fit ranges to derive it from. With a fit, fall back to the old
        # behaviour: convert each per-scint mV fit range to MIP via that scint's MPV
        # and take the mean lower/upper bound (x-axis is the mean MIP across scints).
        if mip_window is not None:
            bump_lo, bump_hi = mip_window
        else:
            if self.moyal_fit_ranges is None:
                raise ValueError("moyal_fit_ranges not set — needed to auto-zoom.")
            mpv_per_scint = self.mpv_per_scint              # set in build_master_df
            fit_los = [self.moyal_fit_ranges[i-1][0] / mpv_per_scint[i] for i in range(1, n+1)]
            fit_his = [self.moyal_fit_ranges[i-1][1] / mpv_per_scint[i] for i in range(1, n+1)]
            bump_lo, bump_hi = float(np.mean(fit_los)), float(np.mean(fit_his))
        master = master[(master[avg_raw] >= bump_lo) & (master[avg_raw] <= bump_hi)] # Zooms to a narrow MIP-peak window — the other method plots the full range.
                                                                                        # This is what makes the raw-vs-amplitude-calibrated comparison meaningful:
                                                                                        # outside this window there isn't a peak to compare calibration against.
        print(f"  rows surviving fit-window [{bump_lo:.3f}, {bump_hi:.3f}] MIP (raw mean): {len(master)}")

        if self.debug:
            print("=" * 70)
            print("DEBUG MIPregion_two_dimensional_histograms")
            print(f"  global_mean_mpv : {getattr(self, 'global_mean_mpv', 'NOT SET')}")
            print(f"  amp_shifts (mV) : {getattr(self, 'amp_shifts', 'NOT SET')}")

            def _stat(col):
                v = master[col].to_numpy()
                v = v[np.isfinite(v)]
                return f"med={np.median(v):.3f} mean={v.mean():.3f} max={v.max():.3f}" if v.size else "(empty)"

            print(f"  MEAN  raw: {_stat(avg_raw)}")
            print(f"        cal: {_stat(avg_cal)}")
            print(f"  STD   raw: {_stat(std_raw)}")
            print(f"        cal: {_stat(std_cal)}")
            print("=" * 70)

        # ── Plotting ─────────────────────────────────────────────────────
        x_range = [bump_lo, bump_hi]
        y_range = [0, 2]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Uncalibrated (÷ per-scint MPV)",
                f"Amplitude-calibrated (shifted, ÷ global mean MPV)",
            ),
            horizontal_spacing=0.12,
        )

        # constant weight → histfunc='sum' yields count/livetime = rate per bin
        w = np.full(len(master), 1.0 / livetime)

        h_raw = go.Histogram2d(
            x=master[avg_raw], y=master[std_raw],
            #z=w,
            histfunc="sum",
            nbinsx=20, nbinsy=20,
            colorscale="Inferno", zmin=0, #zmax=200e-6,
            xbins=dict(start=x_range[0], end=x_range[1]),
            ybins=dict(start=y_range[0], end=y_range[1]),
            colorbar=dict(title="Counts", x=0.455, len=0.9),
            # colorbar=dict(title="Rate [s⁻¹]", x=1.0),
        )
        fig.add_trace(h_raw, row=1, col=1)

        h_cal = go.Histogram2d(
            x=master[avg_cal], y=master[std_cal],
            #z=w,
            histfunc="sum",
            nbinsx=20, nbinsy=20,
            colorscale="Inferno", zmin=0, #zmax=200e-6,
            xbins=dict(start=x_range[0], end=x_range[1]),
            ybins=dict(start=y_range[0], end=y_range[1]),
            colorbar=dict(title="Counts", x=1.02, len=0.9),
            # colorbar=dict(title="Rate [s⁻¹]", x=1.0),
        )
        fig.add_trace(h_cal, row=1, col=2)

        fig.update_xaxes(title_text=f"Mean across {n} scints [MIP]",       range=x_range, row=1, col=1)
        fig.update_xaxes(title_text=f"Mean across {n} scints [MIP] (cal)", range=x_range, row=1, col=2)
        fig.update_yaxes(title_text=f"Std across {n} scints [MIP]",        range=y_range, row=1, col=1)
        fig.update_yaxes(title_text=f"Std across {n} scints [MIP] (cal)",  range=y_range, row=1, col=2)

        fig.update_layout(
            width=1300, height=600,
            title_text=(f"CW{coinc_label} coincidence: cross-scint spread vs. mean "
                        f"— uncalibrated vs amplitude-calibrated  "
                        f"(N={len(master)} events, avg scints ≥ {min_mip} MIP)"),
        )
        self._finish_plotly(fig, "MIP_heatmap")

    # ------------ ALL SCINTS: Plotting All events, No coincidence events, and coincidence events WITH error ------------
    def fill_between_steps(self, x, y1, y2=0, h_align='mid', ax=None, lw=2, **kwargs):
        if ax is None:
            ax = plt.gca()
        xx = np.ravel(np.column_stack((x, x)))[1:]
        xstep = np.ravel(np.column_stack((x[1:] - x[:-1], x[1:] - x[:-1])))
        xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
        xx = np.append(xx, xx.max() + xstep[-1])
        if h_align == 'mid':
            xx -= xstep / 2.
        elif h_align == 'right':
            xx -= xstep
        y1 = np.ravel(np.column_stack((y1, y1)))
        if isinstance(y2, np.ndarray):
            y2 = np.ravel(np.column_stack((y2, y2)))
        ax.fill_between(xx, y1, y2=y2, lw=lw, **kwargs)
        return ax


    def plot_rate_spectra_preview(self, moyal_fit_ranges=None):
        # ── Resolve fit ranges ───────────────────────────────────────────────
        # If the user passed a new set of fit ranges, update the class attribute so
        # subsequent methods (e.g. fit_and_normalize_spectra) reuse the same ranges.
        # If nothing is passed and nothing is set, ranges stay None → the Moyal fit
        # is skipped below and the rate spectra are drawn without a dashed curve.
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        moyal_fit_ranges = self.moyal_fit_ranges   # may be None → fit skipped per-scint below

        # Number of scintillators = number of subplot columns
        x = len(self.processor.fps)

        # ── Figure layout: 2 rows × N columns ────────────────────────────────
        # Row 0 = rate spectra with Poisson uncertainty bands (tall, height ratio 3)
        # Row 1 = ratio panel showing each spectrum / all_events (short, height ratio 1)
        # sharex='col' ties each column's two panels to the same x-axis so the ratio plot lines up visually under the spectrum plot.
        fig, axes = plt.subplots(2, x, figsize=(6 * x, 6), sharex='col', gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, squeeze=False)

        # Loop over each scintillator, pulling the top-row and bottom-row axes together
        for scint_idx, (ax, ax2) in enumerate(zip(axes[0], axes[1]), start=1):

            # Aligned dataframe produced by Scintillators_Processing for this scintillator
            df = getattr(self.processor, f'aligned_scint{scint_idx}')

            # Column holding the SiPM peak voltage (in mV) for this scintillator
            col = f'SiPM_scint{scint_idx}[mV]'

            # ── Define coincidence mask ──────────────────────────────────────
            # delta_CW123 > 0 means "this row corresponds to a triple-coincidence event".
            # Computed once and reused to split events into coincident / non-coincident.
            coinc_mask = df[self.coinc_col] > 0

            # Mask of rows belonging to ANY coincidence order (double, triple, ...).
            # coinc_orders = [2, 3] for N=3 → ['delta_CW12', 'delta_CW123'].
            coinc_order_cols = [f'delta_{self.processor._coinc_tag(k)}' for k in self.processor.coinc_orders]
            any_coinc_mask = (df[coinc_order_cols] > 0).any(axis=1)


            # ── Split events into three populations ──────────────────────────
            # all_events       = every recorded event in this scintillator
            # coinc_events     = events tagged as triple-coincident
            # no_coinc_events  = everything that is NOT triple-coincident
            # .dropna() removes rows where the SiPM column is NaN.
            all_events = df[col].dropna()
            coinc_events = df.loc[coinc_mask, col].dropna()
            no_coinc_events = df.loc[~any_coinc_mask, col].dropna()

            old = (~(df[self.coinc_col] > 0)).sum()
            new = (~any_coinc_mask).sum()
            print(f"non-coinc events: old={old}, new={new}, removed by adding doubles={old-new}")


            # ── Build log-spaced histogram bins ──────────────────────────────
            # Using log-spaced bins because SiPM amplitudes span several orders of magnitude.
            # Bin edges are computed over the union of all three populations so that the
            # three spectra use IDENTICAL bins (otherwise they couldn't be compared bin-to-bin).
            NUM_BINS = 50

            all_sipm = np.concatenate((all_events.values, coinc_events.values, no_coinc_events.values))
            # Drop non-finite values and non-positive entries (log10 requires > 0).
            all_sipm = all_sipm[np.isfinite(all_sipm) & (all_sipm > 0)]

            # 51 edges → 50 bins. Lower edge is clamped at 0.1 mV to avoid log10(0).
            bin_edges = np.logspace(
                np.log10(max(all_sipm.min(), 0.1)),
                np.log10(all_sipm.max()),
                NUM_BINS + 1
            )

            # ── Histogram each population into the shared bin_edges ──────────
            # np.histogram returns (counts, edges); we keep only counts
            # counts_X is a length-50 (bc 50 bins) array where counts_X[j] is the number of
            # events from population X that fell into a bin number j
            counts_all, _ = np.histogram(all_events, bins=bin_edges)
            counts_coinc, _ = np.histogram(coinc_events, bins=bin_edges)
            counts_no_coinc, _ = np.histogram(no_coinc_events, bins=bin_edges)

            # Per-scintillator livetime (deadtime-corrected) produced by Scintillators_Processing.
            # Each scintillator gets its OWN livetime — they can't share one.
            total_livetime_s = getattr(self.processor, f'total_livetime_scint{scint_idx}_s')

            # ── Convert counts → rate ────────────────────────────────────────
            # rate [events/s per bin] = counts / livetime.
            # Dividing by livetime makes the y-axis independent of how long the run was,
            # so spectra from different runs can be compared on equal footing.
            rate_all = counts_all / total_livetime_s
            rate_coinc = counts_coinc / total_livetime_s
            rate_no_coinc = counts_no_coinc / total_livetime_s

            # ── Poisson uncertainty on the rate ──────────────────────────────
            # If you count N events in a bin, the statistical uncertainty is σ_N = √N (Poisson rule).
            # Since rate = N / T and T is a constant, the uncertainty divides the same way:
            #     σ_rate = √N / T
            err_all = np.sqrt(counts_all) / total_livetime_s
            err_coinc = np.sqrt(counts_coinc) / total_livetime_s
            err_no_coinc = np.sqrt(counts_no_coinc) / total_livetime_s

            # Geometric-mean bin centers (correct choice for log-spaced bins,
            # rather than the arithmetic mean which would bias toward the upper edge).
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

            # ── Moyal fit on the coincident spectrum ─────────────────────────
            # Only the coincident events get a Moyal fit here — they form a clean MIP peak.
            # FIT_X_MIN / FIT_X_MAX restrict the fit window to the peak region (per-scint).
            # With no fit ranges set (fixed-MPV path), the fit is skipped and all fit
            # outputs are None so the overlay below is suppressed.
            if moyal_fit_ranges is not None:
                FIT_X_MIN, FIT_X_MAX = moyal_fit_ranges[scint_idx - 1]
                fit_x_coinc, moyal_y_coinc, label_coinc, popt_coinc = self.fit_moyal(bin_centers, rate_coinc, FIT_X_MIN, FIT_X_MAX)
            else:
                fit_x_coinc = moyal_y_coinc = label_coinc = popt_coinc = None

            # ── Upper panel: rate spectra with error bands ───────────────────
            # Plot rate ± σ_rate as a shaded band for each population.
            # fill_between_steps draws the band as horizontal step segments (matching
            # the histogram bin structure) rather than a smooth interpolation.
            colors_list = ['teal', 'darkorange', 'steelblue']
            labels_list = ['All Events', 'Non-Coincident', 'Coincident']

            for rate, err, c, lbl in zip(
                [rate_all, rate_no_coinc, rate_coinc],
                [err_all, err_no_coinc, err_coinc],
                colors_list, labels_list
            ):
                # Draw the ±σ uncertainty band on the upper panel.
                self.fill_between_steps(bin_centers, rate + err, rate - err, color=c, alpha=0.5, ax=ax)
                ax.stairs(rate, bin_edges, color=c, linewidth=1.5)
                # Off-screen dummy line so the legend shows a solid line entry
                # (fill_between alone doesn't produce a clean legend handle).
                ax.plot([1e14], [1e14], color=c, linewidth=2, label=lbl)

            # Overlay the Moyal fit if it converged.
            if moyal_y_coinc is not None:
                ax.plot(fit_x_coinc, moyal_y_coinc, 'k--', linewidth=2, label=label_coinc)

            # Log–log axes for the rate spectrum.
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            # Leave headroom above the maximum rate so the legend doesn't overlap data.
            ax.set_ylim(1e-5, rate_all.max() * 5)
            ax.set_title(f'Scintillator {scint_idx}', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            # ── Lower panel: ratio of each spectrum to all_events ────────────
            # For each (rate, err) pair, plot (rate ± err) / rate_all as a band.
            # This shows what FRACTION of all events in each bin is coincident vs non-coincident,
            # with the uncertainty band propagated from the numerator only (denominator treated as exact).
            #
            # np.divide(..., out=zeros, where=rate_all != 0) safely handles bins where
            # rate_all is zero (which would otherwise raise a divide-by-zero warning and
            # produce inf/NaN). In those bins the ratio is set to 0 instead.
            for rate, err, c in zip(
                                    [rate_no_coinc, rate_coinc],
                                    [err_no_coinc, err_coinc],
                                    ['darkorange', 'steelblue']):
                upper = np.divide(rate + err, rate_all, out=np.zeros_like(rate), where=rate_all != 0)
                lower = np.divide(rate - err, rate_all, out=np.zeros_like(rate), where=rate_all != 0)
                central = np.divide(rate, rate_all, out=np.zeros_like(rate), where=rate_all != 0)
                ax2.stairs(central, bin_edges, color=c, linewidth=1.5)
                self.fill_between_steps(bin_centers, upper, lower, color=c, alpha=0.7, ax=ax2)

            # Reference line at ratio = 1 (i.e. that population = all events in that bin).
            ax2.axhline(1.0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('SiPM Peak Voltage [mV]', fontsize=12)
            # Ratios are bounded between 0 and 1 by construction (each subset ⊆ all_events).
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, which='both', linestyle='--', alpha=0.4)

        # Shared y-axis labels (only on the leftmost column to avoid clutter).
        axes[0][0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12)
        axes[1][0].set_ylabel('Ratio', fontsize=12)
        plt.suptitle(r'Rate Spectra with Poisson Uncertainty Bands ($\sigma_i = \sqrt{N_i}\,/\,T_{\mathrm{live}}$)', fontsize=14)
        plt.tight_layout()
        self._finish_mpl(fig, "rate_spectra")


def process_run(datalogger_fp, scint_fps, moyal_fit_ranges=None, MPVs=None,
                results_dir=None, save_all=False, show_plots=False,
                debug=False, dl_name="datalogger_abs", **method_kwargs):
    '''
    Run the full Datalogger_Processing -> Scintillators_Processing -> Detector_Analysis
    chain in one call, dispatching to the calibration or fixed-MPV pipeline.

    Exactly one of moyal_fit_ranges / MPVs must be given:
        moyal_fit_ranges -> CALIBRATION RUN, calls calibrate_and_analyze_grounddata()
        MPVs             -> ANALYSIS RUN,    calls analyze_calibrated_data_with_fixed_MPVs()

    Args:
        datalogger_fp    : path to the datalogger CSV
        scint_fps        : list of scintillator TXT paths, ordered scint1..scintN
        moyal_fit_ranges : list of (min, max) mV fit windows per scint
        MPVs             : list of per-scint MPVs in mV from a prior calibration
        results_dir      : where Detector_Analysis writes its figures/summary
        save_all         : if True, results_dir is also given to Datalogger_Processing
                           and Scintillators_Processing, so their plots are saved too;
                           if False (default) only the analysis figures are written
        show_plots       : forwarded to both upstream classes. Default False, which
                           with save_all=False means their plots are suppressed entirely
        dl_name          : filename stem for the Absolute Timer datalogger figure
        **method_kwargs  : passed straight to whichever terminal method is dispatched
                           (noise_threshold, twodim_hist_args; mip_window is valid on
                           the fixed-MPV path only)

    Returns:
        (datalogger_df, processor, analysis) — analysis still carries master_df,
        mpv_per_scint, global_mean_mpv, etc. for chaining into a later run.
    '''
    if (moyal_fit_ranges is None) == (MPVs is None):
        raise ValueError("Pass exactly one of moyal_fit_ranges or MPVs.")

    upstream_dir = results_dir if save_all else None

    df        = Datalogger_Processing(datalogger_fp, show_plots=show_plots, debug=debug,
                                      results_dir=upstream_dir).process(name=dl_name)
    processor = Scintillators_Processing(scint_fps, df, show_plots=show_plots, debug=debug,
                                         results_dir=upstream_dir)
    analysis  = Detector_Analysis(processor, df, debug=debug, results_dir=results_dir)

    if moyal_fit_ranges is not None:
        analysis.calibrate_and_analyze_grounddata(moyal_fit_ranges=moyal_fit_ranges, **method_kwargs)
    else:
        analysis.analyze_calibrated_data_with_fixed_MPVs(MPVs=MPVs, **method_kwargs)

    return df, processor, analysis


    # def _finish_mpl(self, fig, name):
    #     if self.save_plots:
    #         path = os.path.join(self.results_dir, f"{name}.png")
    #         fig.savefig(path, dpi=150, bbox_inches='tight')
    #         print(f"[saved] {path}")
    #         plt.close(fig)
    #     else:
    #         plt.show()

    # def _finish_plotly(self, fig, name):
    #     if self.save_plots:
    #         path = os.path.join(self.results_dir, f"{name}.png")
    #         fig.write_image(path, scale=2)
    #         print(f"[saved] {path}")
    #     else:
    #         fig.show()

    # def _save_table(self, df, name):
    #     if self.save_plots:
    #         path = os.path.join(self.results_dir, f"{name}.csv")
    #         df.to_csv(path, index=False)
    #         print(f"[saved] {path}")

    def _save_summary(self, d, name):
        if self.save_plots:
            path = os.path.join(self.results_dir, f"{name}.csv")
            n = len(d["mpv_per_scint"])
            rows = pd.DataFrame({
                "scint":        list(range(1, n + 1)),
                "mpv_mV":       [float(d["mpv_per_scint"][i]) for i in range(1, n + 1)],
                "amp_shift_mV": [float(x) for x in d["amp_shifts"]],
                "livetime_s":   [float(x) if x is not None else float("nan")
                                for x in d["livetime_s_per_scint"]],
            })
            rows["global_mean_mpv_mV"] = float(d["global_mean_mpv"])
            rows["noise_threshold"]    = d["noise_threshold"]
            rows.to_csv(path, index=False)
            print(f"[saved] {path}") 

# # ================== Helper functions for plotting in other scripts ==================
# def plot_density_heatmap_ampcal(analysis, col='MIP_ampcal', normalize_by_livetime=True, cbar_max=None):
#     """
#     2D density heatmap of the amplitude-calibrated cross-scint MIP:
#         x = mean across scints  (SiPM_scints_avg_MIP_ampcal)
#         y = std  across scints  (SiPM_scints_std_MIP_ampcal)

#     Mirrors Detector_Analysis.two_dimensional_histograms but on the ampcal columns.

#     col : str, suffix of the columns to use (e.g. 'MIP_ampcal' or 'MIP_Tcal'), defaults to 'MIP_ampcal';
#     normalize_by_livetime : if True, z = sum(1/livetime) -> rate [s^-1]
#                             (comparable across runs of different duration);
#                             if False, z = raw counts.
#     """
#     master  = analysis.master_df
#     n       = len(analysis.processor.fps)
#     min_mip = analysis.noise_threshold
#     tag     = analysis.coinc_tag
#     coinc_label = '&'.join(str(i) for i in range(1, n + 1))

#     avg_col, std_col = "SiPM_scints_avg_" + col, "SiPM_scints_std_" + col

#     # keep rows where the (calibrated) mean MIP clears the noise threshold
#     sub = master[master[avg_col] >= min_mip].copy()

#     if normalize_by_livetime:
#         livetime = analysis._mean_livetime() # get the average livetime across the n scintillators, find the average
#         print(f"[ampcal heatmap] normalizing by livetime = {livetime:.2f} s")
#         sub["_rate_weight"] = 1.0 / livetime # weight each event by the inverse of the average livetime to get a rate in s^-1; this way, the heatmap's z-axis will represent a rate that is comparable across runs of different duration
#         z, histfunc, cbar = "_rate_weight", "sum", "Normalized counts [s\u207b\u00b9]" # z is the column to aggregate for the heatmap, histfunc is the aggregation function to apply to that column (sum of weights gives a rate), cbar is the colorbar title
#         norm_note = f"rate-normalized by livetime = {livetime:.1f} s" # note to include in the plot title about the normalization
#     else:
#         z, histfunc, cbar = None, "count", "Counts" # if no rate normalizing, then z is None (so heatmap will just count rows), histfunc is "count" to count rows, and cbar title is just "Counts"
#         norm_note = "raw counts"

#     print(f"[ampcal heatmap] rows with {avg_col} >= {min_mip}: {len(sub)} "
#           f"(finite std: {sub[std_col].notna().sum()})")

#     fig = px.density_heatmap(
#         sub,
#         x=avg_col, y=std_col,
#         z=z, histfunc=histfunc,
#         nbinsx=50, nbinsy=50, width=800, height=600,
#         color_continuous_scale="Inferno",
#         range_color=(0, cbar_max) if cbar_max is not None else None,
#         labels={
#             avg_col: f"Mean across {n} scints [MIP, ampcal]",
#             std_col: f"Std across {n} scints [MIP, ampcal]",
#         },
#         title=(f"CW{coinc_label} coincidence: cross-scint spread vs. mean "
#                f"(amplitude-calibrated MIP)<br>(N={len(sub)} events, {norm_note})"),
#     )
#     fig.update_coloraxes(colorbar_title=cbar)
#     finish_plotly(fig, "ampcal_heatmap")

# def split_flight_and_background(datalogger_fp, scint_fps, ground_band=50.0):
#     # Build Absolute Timer with the existing class.
#     dl = Datalogger_Processing(datalogger_fp, show_plots=False)
#     dl.process()
#     df = dl.df.copy()
#     df['Timer[S]'] = df['Absolute Timer (S)']        # flatten resets so halves reprocess cleanly

#     # Cut where altitude returns to ground after apogee.
#     alt = df['Altitude[m]']
#     ground   = alt.iloc[:50].median()
#     peak_idx = alt.idxmax()
#     landed   = alt.loc[peak_idx:][alt.loc[peak_idx:] <= ground + ground_band]
#     t_cut    = df['Absolute Timer (S)'].loc[landed.index[0]]
#     print(f"Cutting at t = {t_cut:.0f} s")

#     out_dir = os.path.dirname(datalogger_fp)
#     paths = {'flight': {'scints': []}, 'background': {'scints': []}}

#     # Split datalogger (drop helper cols so the class rebuilds them on re-read).
#     df = df.drop(columns=[c for c in ['Absolute Timer (S)', 'Timer_rel'] if c in df.columns])
#     base = os.path.splitext(os.path.basename(datalogger_fp))[0]
#     for tag, mask in [('flight', df['Timer[S]'] <= t_cut), ('background', df['Timer[S]'] > t_cut)]:
#         out = os.path.join(out_dir, f"{base}_{tag}.csv")
#         df[mask].to_csv(out, index=False)
#         paths[tag]['datalogger'] = out

#     # Split scintillators at the same t_cut, keeping the 3-line header + tab format.
#     cols = ['Event','Time[s]','Coincident[bool]','ADC[0-4095]','SiPM[mV]','Deadtime[s]','Temp[C]','Pressure[Pa]']
#     for fp in scint_fps:
#         with open(fp) as f:
#             header = [next(f) for _ in range(3)]
#         sdf = pd.read_csv(fp, sep='\t', comment='#', header=None, skiprows=3, names=cols, engine='python')
#         sbase = os.path.splitext(os.path.basename(fp))[0]
#         for tag, mask in [('flight', sdf['Time[s]'] <= t_cut), ('background', sdf['Time[s]'] > t_cut)]:
#             out = os.path.join(out_dir, f"{sbase}_{tag}.txt")
#             with open(out, 'w') as f:
#                 f.writelines(header)
#                 sdf[mask].to_csv(f, sep='\t', header=False, index=False)
#             paths[tag]['scints'].append(out)

#     return paths



# def processing_pipeline(datalogger, scintillators, Moyal_fit_ranges=None, MPVs=None, Show_plots=False, Debug=False):
#     """
#     Runs the full processing pipeline on the given datalogger and scintillator files:
    
#     Args:
#         datalogger (str): File path to the datalogger CSV file.
#         scintillators (list of str): List of file paths to the scintillator TXT files.
#         Moyal_fit_ranges (list of tuples, optional): List of (low, high) fit ranges for each scintillator. If provided, spectra will be fitted with Moyal distributions.
#         MPVs (list of floats, optional): List of fixed MPV values for each scintillator. If provided, spectra will be normalized by these fixed MPVs without fitting.
#         Show_plots (bool, optional): Whether to display plots during processing. Default is False.
#         Debug (bool, optional): Whether to print debug information during processing. Default is False.

#     Returns:
#         tuple containing the datalogger processor instance, scintillators processor instance, and analysis instance.
#     """
#     dl_processor = Datalogger_Processing(datalogger, show_plots=Show_plots, debug=Debug).process()

#     scintillators_processor = Scintillators_Processing(scintillators, dl_processor, show_plots=Show_plots, debug=Debug)
#     analysis = Detector_Analysis(scintillators_processor, dl_processor, debug=Debug)
    
#     if Moyal_fit_ranges is not None:
#         analysis.calibrate_and_analyze_grounddata(moyal_fit_ranges=Moyal_fit_ranges)

#     if MPVs is not None:
#         analysis.analyze_calibrated_data_with_fixed_MPVs(MPVs=MPVs)

#     if MPVs is None and Moyal_fit_ranges is None:
#         raise ValueError("Must provide either Moyal fit ranges or fixed MPVs for the spectra.")
    
#     return dl_processor, scintillators_processor, analysis
