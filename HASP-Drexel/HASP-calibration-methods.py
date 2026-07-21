import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
from scipy.optimize import curve_fit


def build_absolute_timer(df, time_col='microseconds_since_boot', reset_threshold=-10_000_000):
    '''
    Detects timer resets in the given time column, builds a continuous
    absolute timer by chaining segments end-to-end, and returns df with
    a new absolute timer column appended.

    Args:
        df:                pd.DataFrame containing the time column
        time_col:          string name of the column holding timer values
        reset_threshold:   a drop below this value (default -10_000_000 us)
                        is treated as a timer reset

    ReturnsE
        df with new 'Absolute Timer (us)' column added
    '''

    df_sorted = df.copy().sort_index().reset_index(drop=True)                      # make sure rows are in original order + give clean 0-based index

    time_col = str(time_col)                                            # ensure time_col is a string (in case it was passed as something else)

    reset_mask = df_sorted[time_col].diff() < reset_threshold                                # .diff() computes row-to-row differences; this creates a boolean mask that's True wherever the time decreased by more than 3 seconds (negative value indicates a reset/jump backward)

    new_segment_starts = [0] + (reset_mask[reset_mask].index).tolist()   # list of row numbers where each new segment begins: 0 + (index of each True + 1 to point to first row of next segment)
    print(f"Detected {len(new_segment_starts)-1} separate timer reset(s) at rows {new_segment_starts}")

    timerreset_segments = []                                                            # empty list that will store one DataFrame per continuous run

    for i in range(len(new_segment_starts)):                                 # loop over each detected starting point
        start = new_segment_starts[i]                                        # determine starting row index of current segment; store in "start"
        end = new_segment_starts[i+1] if i+1 < len(new_segment_starts) else None   # ending row = start of next segment, or None (means go to end of DataFrame)
        seg = df_sorted.iloc[start:end].copy()                                  # slice out this segment and make independent copy; iloc
        seg['Timer_rel'] = seg[time_col] - seg[time_col].iloc[0]             # add relative time column that starts at ~0 for this run
        timerreset_segments.append(seg)                                                 # store this segment in the list

    for i, seg in enumerate(timerreset_segments, 1):
        t_min = seg[time_col].min()
        t_max = seg[time_col].max()
        nrows = len(seg)
        print(f"  Run {i:2d}: {nrows:5d} rows, timer {t_min:6.0f} → {t_max:6.0f} s")

    absolute_timers = []
    previous_end = 0.0

    for i, seg in enumerate(timerreset_segments):
        current_timers = seg[time_col].values

        if len(current_timers) == 0:
            continue

        absolute_this_segment = current_timers + previous_end

        absolute_timers.extend(absolute_this_segment)

        previous_end = absolute_this_segment[-1]

    df['Absolute Timer (us)'] = absolute_timers

    return df


# COINCIDENCE_GROUPS = {
#     'col1': [1, 5, 9, 13],
#     'col2': [2, 6, 10, 14],
#     'col3': [3, 7, 11, 15],
#     'col4': [4, 8, 12, 16],}
# and the merged dataframe should have ADC columns like sipm_01_adc ... sipm_16_adc, so the grouping for the columns and plotting can be done using this logic

def plot_ADC_histograms(df, coincidence_groups, bins=100):
    n_layers = 4
    columns = len(coincidence_groups) # number of physical columns, one per key in coincidence_groups

    for ch in range(1, columns + 1): # looping over each physical column (col1, col2, col3, col4)

        col_name = f'col{ch}'
        trigger_nums = coincidence_groups[col_name]
        all_events = pd.concat([df[f'sipm_{(layer - 1) * 4 + ch:02d}_adc'] for layer in range(1, n_layers + 1)]).dropna() # concatenate ADC values from all layers in this column

        palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        data    = [all_events]
        labels  = ['All events']
        colors  = [palette[0]]

        # One entry per coincidence order (CW12, CW123, ..., up to CW1...N)
        fold_names = {2: 'Double', 3: 'Triple', 4: 'Quadruple', 5: 'Quintuple', 6: 'Sextuple'}
        for idx, order in enumerate(range(2, len(trigger_nums) + 1), start=1):
            label = '&'.join(str(n) for n in trigger_nums[:order])
            delta_col = f'delta_{col_name}_CW_{label}'
            events_k = pd.concat([
                df.loc[df[delta_col] > 0, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, n_layers + 1)
            ]).dropna()
            fold = fold_names.get(order, f'{order}-fold')
            data.append(events_k) # add this coincidence level's ADC values to the list to be plotted
            labels.append(f'{fold} coincidence events ({col_name}_CW_{label})')
            colors.append(palette[idx % len(palette)])

        fig, ax = plt.subplots(figsize=(8, 4))

        for dataset, color, label in zip(data, colors, labels):
            ax.hist(dataset, bins=bins, color=color, edgecolor='black', alpha=0.5, label=label)

        ax.set_xlabel('ADC [0-4095]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'ADC Distribution — Channel {ch}', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()


    # def apply_deadtime_correction(self):
    #     """
    #     Computes deadtime-corrected per-event livetime for each scintillator,
    #     using the same approach as getCosmicWatch() in the reference script.

    #     For each event row, livetime = (time elapsed since last event) - (deadtime accumulated since last event)
    #     This is then attached to the scintillator DataFrame as a column before merging.

    #     Args:
    #         i: scintillator index (1-based)

    #     Returns:
    #         pd.Series of deadtime-corrected livetime values, aligned to scint_i's index
    #     """
    #     scint_df = getattr(self, f'scint_{i}')
    #     time  = s['Time[s]'].values
    #     deadt = scint_df['Deadtime[s]'].values

    #     # delta deadtime per event
    #     event_deadt_s    = np.diff(np.append([0], deadt))
    #     event_livetime_s = np.diff(np.append([0], time)) - event_deadt_s

    #     # clip negatives — can occur at file boundaries or timer resets
    #     event_livetime_s = event_livetime_s.clip(min=0)

    #     return pd.Series(event_livetime_s, index=scint_df.index,
    #                     name=f'livetime_scint{i}[s]')

class CW_Analysis:
    def __init__(self, processor):
        self.processor = processor
        self.moyal_fit_ranges = None
        self.stored = None
        self.results_dir = os.path.join(os.getcwd(), 'analysis_results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def apply_deadtime_correction(self):
        """
        Computes deadtime-corrected per-event livetime for all 16 scintillators
        across the merged df in one pass. For each sipm, filters to rows where
        it fired, computes livetime = (time elapsed since last event) - (deadtime
        of this event), and attaches the result as a new column on self.df.
        """
        for i in range(1, 17):
            trigger_col  = f'sipm_{i:02d}_trigger'
            deadtime_col = f'trigger_{i:02d}_dead_time_t1'

            scint_df = self.processor.df[self.df[trigger_col] == 1]

            time  = scint_df['Absolute Timer (us)'].values / 1e6   # to seconds
            deadt = scint_df[deadtime_col].values / 1e6            # to seconds

            event_livetime_s = np.diff(np.append([0], time)) - deadt
            event_livetime_s = event_livetime_s.clip(min=0)

            self.processor.df.loc[scint_df.index, f'livetime_scint{i:02d}[s]'] = event_livetime_s

    def fit_moyal(self, centers, rates, fit_x_min=200, fit_x_max=1000, fit_x_n=300):
        def moyal(x, mpv, eta, A):
            return A * np.exp(-0.5 * (((x - mpv) / eta) + np.exp(-(x - mpv) / eta)))

        # Define range of ADC values to fit over, and interpolate the rates to get y-values at those x points
        fit_x = np.linspace(fit_x_min, fit_x_max, fit_x_n)
        rates_filled = np.nan_to_num(rates, nan=0.0)
        fit_y = np.interp(fit_x, centers, rates_filled)
        
        # Use curve_fit to find best-fit parameters for the Moyal function, starting from an initial guess based on the data
        try:
            p0 = [centers[rates_filled.argmax()], 15, rates_filled.max()]
            popt, _ = curve_fit(moyal, fit_x, fit_y, p0=p0, maxfev=10000)
            moyal_fit_line = moyal(fit_x, *popt)
            moyal_label = (rf'Moyal Fit '
                           rf'($\mu={popt[0]:.1f}$ ADC, '
                           rf'$\eta={popt[1]:.1f}$ ADC)')
            print(f"Moyal fit — "
                  f"MPV: {popt[0]:.2f} ADC, "
                  f"eta: {popt[1]:.2f} ADC, "
                  f"A: {popt[2]:.4f}")
            return fit_x, moyal_fit_line, moyal_label, popt
        except RuntimeError:
            print("Moyal fit did not converge — try adjusting fit_x_min/fit_x_max or p0")
            return None, None, None, None

    def rate_spectra_with_moyal(self, moyal_fit_ranges=None, save_path=None):
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        if self.moyal_fit_ranges is None:
            raise ValueError("moyal_fit_ranges not set.")
        moyal_fit_ranges = self.moyal_fit_ranges


        if save_path is not None:
            self.path = save_path

        self.plot_all_scints_no_error(moyal_fit_ranges)
        self.plot_all_scints_with_error(moyal_fit_ranges)
        self.compute_and_normalize()
        self.add_mip_columns()
        self.plot_density_heatmaps()

    def plot_all_scints_no_error(self, moyal_fit_ranges=None):
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        if self.moyal_fit_ranges is None:
            raise ValueError("moyal_fit_ranges not set.")
        moyal_fit_ranges = self.moyal_fit_ranges

        x = self.processor.n_channels

        fig, axes = plt.subplots(1, x, figsize=(6 * x, 5), sharey=True, squeeze=False)
        axes = axes[0]

        for ch, ax in enumerate(axes, start=1):
            col_name = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            top_label = '&'.join(str(n) for n in trigger_nums)
            delta_col = f'delta_{col_name}_CW_{top_label}'

            coinc_mask = self.processor.df[delta_col] > 0

            all_events = pd.concat([
                self.processor.df[f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()
            coinc_events = pd.concat([
                self.processor.df.loc[coinc_mask, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()
            no_coinc_events = pd.concat([
                self.processor.df.loc[~coinc_mask, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()

            NUM_BINS = 50
            bin_edges = np.linspace(0, 4095, NUM_BINS + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            counts_all, _      = np.histogram(all_events, bins=bin_edges)
            counts_coinc, _    = np.histogram(coinc_events, bins=bin_edges)
            counts_no_coinc, _ = np.histogram(no_coinc_events, bins=bin_edges)

            total_livetime_s = getattr(self.processor, f'livetime_scint{(ch-1)*4+1:02d}[s]').sum()
            
            rate_all      = counts_all      / total_livetime_s
            rate_coinc    = counts_coinc    / total_livetime_s
            rate_no_coinc = counts_no_coinc / total_livetime_s

            FIT_X_MIN, FIT_X_MAX = moyal_fit_ranges[ch - 1]
            fit_x_coinc, moyal_y_coinc, label_coinc, popt_coinc = self.fit_moyal(
                bin_centers, rate_coinc, FIT_X_MIN, FIT_X_MAX)

            ax.stairs(rate_all,      bin_edges, color='teal',      linewidth=1.5, label='All Events')
            ax.stairs(rate_no_coinc, bin_edges, color='darkorange', linewidth=1.5, label='Non-Coincident')
            ax.stairs(rate_coinc,    bin_edges, color='steelblue',  linewidth=1.5, label='Coincident')

            colors_list = ['teal', 'darkorange', 'steelblue']
            labels_list = ['All Events', 'Non-Coincident', 'Coincident']
            for rate, c, lbl in zip([rate_all, rate_no_coinc, rate_coinc], colors_list, labels_list):
                ax.plot([1e14], [1e14], color=c, linewidth=2, label=lbl)

            if moyal_y_coinc is not None:
                ax.plot(fit_x_coinc, moyal_y_coinc, 'k--', linewidth=2, label=label_coinc)

            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.set_xlabel('ADC [0-4095]', fontsize=12)
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_ylim(1e-5, rate_all.max() * 5)
            ax.set_title(f'Channel {ch}', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

        axes[0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12)
        plt.suptitle(r'Rate Spectra', fontsize=14)
        plt.savefig(os.path.join(self.results_dir, 'rate_spectra_no_error.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def fill_between_steps(self, x, y1, y2=0, h_align='mid', ax=None, lw=2, **kwargs):
        # unchanged
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

    def plot_all_scints_with_error(self, moyal_fit_ranges=None):
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        if self.moyal_fit_ranges is None:
            raise ValueError("moyal_fit_ranges not set.")
        moyal_fit_ranges = self.moyal_fit_ranges

        x = self.processor.n_channels

        fig, axes = plt.subplots(2, x, figsize=(6 * x, 6), sharex='col',
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, squeeze=False)

        for ch, (ax, ax2) in enumerate(zip(axes[0], axes[1]), start=1):
            col_name = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            top_label = '&'.join(str(n) for n in trigger_nums)
            delta_col = f'delta_{col_name}_CW_{top_label}'

            coinc_mask = self.processor.df[delta_col] > 0

            all_events = pd.concat([
                self.processor.df[f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()
            coinc_events = pd.concat([
                self.processor.df.loc[coinc_mask, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()
            no_coinc_events = pd.concat([
                self.processor.df.loc[~coinc_mask, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna()

            NUM_BINS = 50
            bin_edges   = np.linspace(0, 4095, NUM_BINS + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            counts_all, _      = np.histogram(all_events, bins=bin_edges)
            counts_coinc, _    = np.histogram(coinc_events, bins=bin_edges)
            counts_no_coinc, _ = np.histogram(no_coinc_events, bins=bin_edges)

            total_livetime_s = 1  # TODO: replace once deadtime columns confirmed

            rate_all      = counts_all      / total_livetime_s
            rate_coinc    = counts_coinc    / total_livetime_s
            rate_no_coinc = counts_no_coinc / total_livetime_s

            err_all      = np.sqrt(counts_all)      / total_livetime_s
            err_coinc    = np.sqrt(counts_coinc)    / total_livetime_s
            err_no_coinc = np.sqrt(counts_no_coinc) / total_livetime_s

            FIT_X_MIN, FIT_X_MAX = moyal_fit_ranges[ch - 1]
            fit_x_coinc, moyal_y_coinc, label_coinc, popt_coinc = self.fit_moyal(
                bin_centers, rate_coinc, FIT_X_MIN, FIT_X_MAX)

            colors_list = ['teal', 'darkorange', 'steelblue']
            labels_list = ['All Events', 'Non-Coincident', 'Coincident']

            for rate, err, c, lbl in zip(
                [rate_all, rate_no_coinc, rate_coinc],
                [err_all,  err_no_coinc,  err_coinc],
                colors_list, labels_list
            ):
                self.fill_between_steps(bin_centers, rate + err, rate - err, color=c, alpha=0.5, ax=ax)
                ax.stairs(rate, bin_edges, color=c, linewidth=1.5)
                ax.plot([1e14], [1e14], color=c, linewidth=2, label=lbl)

            if moyal_y_coinc is not None:
                ax.plot(fit_x_coinc, moyal_y_coinc, 'k--', linewidth=2, label=label_coinc)

            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_ylim(1e-5, rate_all.max() * 5)
            ax.set_title(f'Channel {ch}', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            for rate, err, c in zip(
                [rate_no_coinc, rate_coinc],
                [err_no_coinc,  err_coinc],
                ['darkorange', 'steelblue']
            ):
                upper = np.divide(rate + err, rate_all, out=np.zeros_like(rate), where=rate_all != 0)
                lower = np.divide(rate - err, rate_all, out=np.zeros_like(rate), where=rate_all != 0)
                ax.stairs(rate, bin_edges, color=c, linewidth=1.5)
                self.fill_between_steps(bin_centers, upper, lower, color=c, alpha=0.7, ax=ax2)

            ax2.axhline(1.0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('ADC [0-4095]', fontsize=12)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, which='both', linestyle='--', alpha=0.4)

        axes[0][0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12)
        axes[1][0].set_ylabel('Ratio', fontsize=12)
        plt.suptitle(r'Rate Spectra with Poisson Uncertainty Bands ($\sigma_i = \sqrt{N_i}\,/\,T_{\mathrm{live}}$)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'rate_spectra_with_error.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def compute_and_normalize(self, moyal_fit_ranges=None):
        if moyal_fit_ranges is not None:
            self.moyal_fit_ranges = moyal_fit_ranges
        if self.moyal_fit_ranges is None:
            raise ValueError("moyal_fit_ranges not set.")
        moyal_fit_ranges = self.moyal_fit_ranges

        x = self.processor.n_channels

        fig, axes_all = plt.subplots(2, x, figsize=(6 * x, 10), squeeze=False)
        axes_top = axes_all[0]
        stored = []
        colors = ['teal', 'darkorange', 'steelblue']

        for ch, ax in enumerate(axes_top, start=1):
            col_name     = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            top_label    = '&'.join(str(n) for n in trigger_nums)
            delta_col    = f'delta_{col_name}_CW_{top_label}'

            total_livetime_s = 1  # TODO: replace once deadtime columns confirmed

            coinc_mask = self.processor.df[delta_col] > 0
            data = pd.concat([
                self.processor.df.loc[coinc_mask, f'sipm_{(layer - 1) * 4 + ch:02d}_adc']
                for layer in range(1, self.processor.n_layers + 1)
            ]).dropna().values

            bin_edges   = np.linspace(0, 4095, 51)
            counts, _   = np.histogram(data, bins=bin_edges)
            rate        = counts / total_livetime_s
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            FIT_X_MIN, FIT_X_MAX = moyal_fit_ranges[ch - 1]
            fit_x, moyal_y, label, popt = self.fit_moyal(bin_centers, rate, FIT_X_MIN, FIT_X_MAX)

            ax.stairs(rate, bin_edges, color=colors[(ch - 1) % len(colors)], linewidth=1.5)
            if moyal_y is not None:
                ax.plot(fit_x, moyal_y, 'k--', linewidth=2, label=label)
                ax.legend(fontsize=10)

            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.set_xlabel('ADC [0-4095]', fontsize=12)
            ax.set_title(f'Channel {ch}', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)

            peak_rate = max(moyal_y) if moyal_y is not None else None
            stored.append({'rate': rate, 'bin_edges': bin_edges, 'bin_centers': bin_centers,
                           'fit_x': fit_x, 'moyal_y': moyal_y, 'peak_rate': peak_rate, 'popt': popt})

        axes_top[0].set_ylabel(r'Rate/bin [s$^{-1}$]', fontsize=12)
        self.stored = stored

        axes = axes_all[1]
        colors     = ['teal', 'darkorange', 'steelblue']
        colors_dim = ['lightgray', 'bisque', 'lightblue']

        for ch_idx, ax in enumerate(axes):
            s = stored[ch_idx]
            if s['peak_rate'] is None:
                ax.set_title(f'Channel {ch_idx + 1} — fit failed', fontsize=13)
                continue

            mpv = s['popt'][0]

            bin_edges_mip   = s['bin_edges']   / mpv
            bin_centers_mip = s['bin_centers'] / mpv
            fit_x_mip       = s['fit_x']       / mpv

            below_mask = bin_centers_mip < 0.5
            above_mask = ~below_mask
            rate_below = np.where(below_mask, s['rate'], np.nan)
            rate_above = np.where(above_mask, s['rate'], np.nan)

            ax.stairs(rate_above, bin_edges_mip, color=colors[ch_idx % len(colors)],         linewidth=1.5, label='≥ 0.5 MIP')
            ax.stairs(rate_below, bin_edges_mip, color=colors_dim[ch_idx % len(colors_dim)], linewidth=1.5, label='< 0.5 MIP')
            ax.plot(fit_x_mip, s['moyal_y'], 'k--', linewidth=2)

            ax.axvline(1.0, color=colors[ch_idx % len(colors)], linestyle=':', linewidth=1.5)
            ax.text(1.05, 0.5, f'MPV = {mpv:.1f} ADC',
                    transform=ax.get_xaxis_transform(),
                    fontsize=10, color=colors[ch_idx % len(colors)],
                    rotation=90, va='center')
            ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1e-2)
            ax.set_xlim(0, 2.5)
            ax.set_xlabel('Amplitude [MIP]', fontsize=12)
            ax.set_title(f'Channel {ch_idx + 1}: 1 MIP = {mpv:.1f} ADC', fontsize=13)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

        axes[0].set_ylabel('Rate / Moyal Peak', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'rate_spectra_normalized.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def add_mip_columns(self):
        if self.stored is None:
            raise ValueError("Run compute_and_normalize() first.")

        master = self.processor.df

        mpv_per_layer_ch = {}
        for ch in range(1, self.processor.n_channels + 1):
            popt = self.stored[ch - 1]['popt']
            if popt is not None:
                for layer in range(1, self.processor.n_layers + 1):
                    mpv_per_layer_ch[(layer, ch)] = popt[0]

        for ch in range(1, self.processor.n_channels + 1):
            col_name     = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            for order in range(2, len(trigger_nums) + 1):
                label     = '&'.join(str(n) for n in trigger_nums[:order])
                delta_col = f'delta_{col_name}_CW_{label}'
                for layer in range(1, self.processor.n_layers + 1):
                    adc_col = f'sipm_{(layer - 1) * 4 + ch:02d}_adc'
                    master[f'ADC_{col_name}_CW_{label}_layer{layer}'] = (master[adc_col].where(master[delta_col] > 0))

        for ch in range(1, self.processor.n_channels + 1):
            col_name     = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            for order in range(2, len(trigger_nums) + 1):
                label = '&'.join(str(n) for n in trigger_nums[:order])
                for layer in range(1, self.processor.n_layers + 1):
                    if (layer, ch) not in mpv_per_layer_ch:
                        continue
                    mpv        = mpv_per_layer_ch[(layer, ch)]
                    tagged_col = f'ADC_{col_name}_CW_{label}_layer{layer}'
                    master[f'ADC_MIP_{col_name}_CW_{label}_layer{layer}'] = master[tagged_col] / mpv

        for ch in range(1, self.processor.n_channels + 1):
            col_name     = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            top_label    = '&'.join(str(n) for n in trigger_nums)

            adc_top_cols = [f'ADC_{col_name}_CW_{top_label}_layer{layer}' for layer in range(1, self.processor.n_layers + 1)]
            master[f'ADC_{col_name}_layers_avg'] = master[adc_top_cols].mean(axis=1, skipna=False)
            master[f'ADC_{col_name}_layers_std'] = master[adc_top_cols].std(axis=1, ddof=1, skipna=False)

            mip_top_cols = [f'ADC_MIP_{col_name}_CW_{top_label}_layer{layer}' for layer in range(1, self.processor.n_layers + 1)]
            master[f'ADC_MIP_{col_name}_layers_avg'] = master[mip_top_cols].mean(axis=1, skipna=False)
            master[f'ADC_MIP_{col_name}_layers_std'] = master[mip_top_cols].std(axis=1, ddof=1, skipna=False)

            for layer in range(1, self.processor.n_layers):
                master[f'ADC_{col_name}_diff_layer{layer}_minus_layer{layer+1}'] = (
                    master[f'ADC_{col_name}_CW_{top_label}_layer{layer}'] -
                    master[f'ADC_{col_name}_CW_{top_label}_layer{layer+1}'])

        self.processor.master_df = master
        self.mpv_per_layer_ch = mpv_per_layer_ch
        print(f"Master dataframe shape: {master.shape}")
        print(f"MPVs used: {mpv_per_layer_ch}")
        print(f"Columns: {master.columns.tolist()}")

    def plot_density_heatmaps(self, mip_noise_threshold=0.5):
        master = self.processor.master_df

        for ch in range(1, self.processor.n_channels + 1):
            col_name     = f'col{ch}'
            trigger_nums = self.processor.coincidence_groups[col_name]
            top_label    = '&'.join(str(n) for n in trigger_nums)
            coinc_label  = top_label

            adc_col     = f'ADC_{col_name}_CW_{top_label}_layer1'
            mip_col     = f'ADC_MIP_{col_name}_CW_{top_label}_layer1'
            std_col     = f'ADC_{col_name}_layers_std'
            avg_mip_col = f'ADC_MIP_{col_name}_layers_avg'
            std_mip_col = f'ADC_MIP_{col_name}_layers_std'

            mip_cols = [f'ADC_MIP_{col_name}_CW_{top_label}_layer{layer}' for layer in range(1, self.processor.n_layers + 1)]
            df_ch = master[(master[mip_cols] >= mip_noise_threshold).all(axis=1)]

            fig1 = px.density_heatmap(
                df_ch[df_ch[adc_col] >= 10],
                x=adc_col, y=std_col,
                nbinsx=50, nbinsy=50, width=800, height=600,
                color_continuous_scale="Inferno", range_color=[0, 10],
                labels={adc_col: f'Layer 1 ADC ({col_name})',
                        std_col: f'Std across {self.processor.n_layers} layers (ADC)'},
                title=f'{col_name} CW {coinc_label}: layer 1 ADC vs. spread (raw ADC)',
            )
            fig1.write_image(os.path.join(self.results_dir, f'{col_name}_heatmap_raw_adc.png'))
            fig1.show()

            fig2 = px.density_heatmap(
                df_ch[df_ch[mip_col] >= mip_noise_threshold],
                x=mip_col, y=std_mip_col,
                nbinsx=50, nbinsy=50, width=800, height=600,
                color_continuous_scale="Inferno", range_color=[0, 10],
                labels={mip_col:     f'Layer 1 [MIP] ({col_name})',
                        std_mip_col: f'Std across {self.processor.n_layers} layers [MIP]'},
                title=f'{col_name} CW {coinc_label}: layer 1 vs. spread (MIP-normalized)',
            )
            fig2.write_image(os.path.join(self.results_dir, f'{col_name}_heatmap_mip_layer1.png'))
            fig2.show()

            fig3 = px.density_heatmap(
                df_ch[df_ch[avg_mip_col] >= mip_noise_threshold],
                x=avg_mip_col, y=std_mip_col,
                nbinsx=50, nbinsy=50, width=800, height=600,
                color_continuous_scale="Inferno", range_color=[0, 10],
                labels={avg_mip_col: f'Mean across {self.processor.n_layers} layers [MIP] ({col_name})',
                        std_mip_col: f'Std across {self.processor.n_layers} layers [MIP]'},
                title=f'{col_name} CW {coinc_label}: cross-layer spread vs. mean (MIP-normalized)',
            )
            fig3.write_image(os.path.join(self.results_dir, f'{col_name}_heatmap_mip_avg.png'))
            fig3.show()