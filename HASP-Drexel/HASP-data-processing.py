"""
HASP-data-processing.py — pre-processing of two SiPM teensy CSVs into a unified
per-event DataFrame for downstream coincidence-spectrum analysis.

Steps:
  1. Load both teensy CSVs.
  2. Compute a per-row event_time (earliest non-zero signal_time on that row).
  3. Merge the two teensies row-by-row by nearest event_time within a tolerance.
  4. Detect orphan rows on either side (events that didn't match across teensies).
  5. Validate that the 16-bit trigger_binary pattern agrees on matched rows.
  6. Build a unified DataFrame plus a drift-diagnostic histogram.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === File paths (edit per run) ============================================
teensy1_fp = '/Users/emmamartignoni/Desktop/Thesis/sipm_teensy_1_template.csv'
teensy2_fp = '/Users/emmamartignoni/Desktop/Thesis/sipm_teensy_2_template.csv'

# Load raw teensy CSVs
df1 = pd.read_csv(teensy1_fp)
df2 = pd.read_csv(teensy2_fp)

# 1. Right after loading — confirm shape and column names
print("=== RAW TEENSY 1 ===")
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print(df1.head(3))

print("\n=== RAW TEENSY 2 ===")
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
print(df2.head(3))



# === Coincidence-group config ============================================
# Maps each "column" (physical stack of 4 scints) to the trigger numbers
# believed to belong to it. EDIT THIS once colleagues confirm wiring.
# Order within each list matters: it defines the cumulative coincidence
# chain (e.g. col1_CW_1&2 uses the first two, col1_CW_1&2&3 the first three).
COINCIDENCE_GROUPS = {
    'col1': [1, 5, 9, 13],
    'col2': [2, 6, 10, 14],
    'col3': [3, 7, 11, 15],
    'col4': [4, 8, 12, 16],
}

# All plots get saved to ./results/ relative to wherever this script is run.
# exist_ok=True so re-runs don't crash if the folder is already there.
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# === Config ===============================================================
# Max allowed time gap between matched events on the two teensies.
# Units = whatever trigger_NN_signal_time uses (TBD; assumed seconds).
MERGE_TOLERANCE = 0.1 # PLACEHOLDER — tune once you've seen the real drift distribution.


# Column-name groups, used repeatedly below.
TRIGGER_BINARY_COLS = [f'sipm_{i:02d}_trigger' for i in range(1, 17)] # creates ['sipm_01_trigger', 'sipm_02_trigger', ..., 'sipm_16_trigger'] list
TRIGGER_DEAD_TIME_COLS  = ['trigger_' + str(i).zfill(2) + '_dead_time'  for i in range(1, 17)] # creates ['trigger_01_dead_time', 'trigger_02_dead_time', ..., 'trigger_16_dead_time'] list




# === Per-row event_time ===================================================
df1['event_time_unix_s'] = pd.to_datetime(df1['utc_time']).astype(np.int64) / 1e9 # adding column for event time in unix seconds since its easier to work with downstream
df2['event_time_unix_s'] = pd.to_datetime(df2['utc_time']).astype(np.int64) / 1e9 # same for teensy 2
df2['event_time_unix_s_t2'] = df2['event_time_unix_s'] # rename teensy2 unix timestamps so they dont get lost in merge and can be used for drift diagnostic later


# 2. After event_time is computed — sanity check the timestamp range
print("\n=== EVENT_TIME RANGES ===")
print(f"T1 event_time_unix_s: min={df1['event_time_unix_s'].min():.3f}  max={df1['event_time_unix_s'].max():.3f}  "
      f"span={df1['event_time_unix_s'].max() - df1['event_time_unix_s'].min():.3f}s")
print(f"T2 event_time_unix_s: min={df2['event_time_unix_s'].min():.3f}  max={df2['event_time_unix_s'].max():.3f}  "
      f"span={df2['event_time_unix_s'].max() - df2['event_time_unix_s'].min():.3f}s")

# 

# === Sort by event_time_unix_s (required by merge_asof) ==========================
df1 = df1.sort_values('event_time_unix_s').reset_index(drop=True)
df2 = df2.sort_values('event_time_unix_s').reset_index(drop=True)


# === Tag original row indices so we can detect orphans ====================
# After merging, any df2 row whose original index doesn't appear in the
# merge output is a teensy-2 orphan (no df1 row within tolerance).
df1['_t1_orig_idx'] = df1.index
df2['_t2_orig_idx'] = df2.index


# === Rename teensy 2 columns to disambiguate from teensy 1 ================
# merge_asof resolves shared column names by appending _x/_y, which we don't
# want. By renaming first, the columns arrive at the merge already labeled
# correctly so pandas has nothing to resolve.
#
# For teensy 1: only signal_time and cpu_temperature get _t1 added.
#   e.g. {'trigger_01_signal_time': 'trigger_01_signal_time_t1', ...}
#   The binary columns are intentionally left as-is — they become the
#   canonical trigger pattern in the merged output.
#
# For teensy 2: signal_time, binary, AND cpu_temperature all get _t2 added.
#   The binary cols are only kept temporarily for the pattern-agreement check
#   (match_status) and are dropped afterward.
# --- Rename teensy 1 columns ---
t1_time_renames = {} # dictionary mapping column renames to easily identify teensy1 and teensy2 in merged dataframe and preserve all data
for col in TRIGGER_DEAD_TIME_COLS:
    t1_time_renames[col] = col + '_t1'
t1_time_renames['microseconds_since_boot'] = 'microseconds_since_boot_t1' # 'microseconds_since_boot' -> 'microseconds_since_boot_t1'
t1_time_renames['utc_time'] = 'utc_time_t1' # 'utc_time' -> 'utc_time_t1'
for col in TRIGGER_BINARY_COLS: # renaming t1 binary cols with _t1 suffix to match t2, so all columns in merged are symmetrically labeled
    t1_time_renames[col] = col + '_t1'
df1 = df1.rename(columns=t1_time_renames)
df1 = df1.rename(columns={'cpu_temperature': 'cpu_temperature_t1'})

# --- Rename teensy 2 columns ---
t2_time_renames = {}
for col in TRIGGER_DEAD_TIME_COLS:
    t2_time_renames[col] = col + '_t2'
t2_time_renames['microseconds_since_boot'] = 'microseconds_since_boot_t2'
t2_time_renames['utc_time'] = 'utc_time_t2' # e.g. 'trigger_01_event_time' -> 'trigger_01_event_time_t2'
df2 = df2.rename(columns=t2_time_renames)
t2_binary_renames = {}
for col in TRIGGER_BINARY_COLS: # looping over the trigger event detection binary columns to rename with suffic t2, not doing this with t1 since want to keep them as the canonical trigger pattern (ie. not renamed)
    t2_binary_renames[col] = col + '_t2'
df2 = df2.rename(columns=t2_binary_renames)
df2 = df2.rename(columns={'cpu_temperature': 'cpu_temperature_t2'})


# === Merge: nearest event_time_unix_s within tolerance ===========================
# For every df1 row, find the df2 row with the closest event_time_unix_s;
# if none is within MERGE_TOLERANCE, df2 columns come back as NaN.
merged = pd.merge_asof(
    df1, df2,
    on='event_time_unix_s',
    direction='nearest',
    tolerance=MERGE_TOLERANCE,
)


# 3. Right after merge_asof, before orphans are appended — see how many matched vs dropped
n_matched = merged['_t2_orig_idx'].notna().sum()
n_unmatched_t1 = merged['_t2_orig_idx'].isna().sum()
print(f"\n=== AFTER merge_asof (before orphan concat) ===")
print(f"T1 rows with a T2 match within tolerance: {n_matched}")
print(f"T1 rows with NO T2 match (will be orphan_t1): {n_unmatched_t1}")


# Add teensy-2 orphans (events that didn't have time match across teensies)
matched_t2_idx = merged['_t2_orig_idx'].dropna().astype(int).tolist() # get list of original indices of ALL teensy2 rows that made it into the merged dataframe
t2_orphans = df2[~df2['_t2_orig_idx'].isin(matched_t2_idx)].copy() # slice df2 by row, down to just those where the original index is NOT in the merged dataframe's list of matches indices; result is row of teensy2 orphan indices
merged = pd.concat([merged, t2_orphans], ignore_index=True, sort=False) # stack the teensy2 orphan rows onto the bottom of the merged DataFrame, resetting the index to be continuous from 0 again


# === Build match_status column ============================================
# Four possible states per row:
#   matched          : both teensies have data and trigger patterns agree
#   pattern_mismatch : both teensies have data but patterns disagree
#   orphan_t1        : df1 row, no df2 match within tolerance
#   orphan_t2        : df2 row appended above with no df1 match
has_t1 = merged['_t1_orig_idx'].notna()                                    # boolean Series, one entry per row of merged: True where _t1_orig_idx is not NaN, meaning that row of merged has data from teensy 1
has_t2 = merged['_t2_orig_idx'].notna()                                    # same for teensy 2 → True where this row has df2 data attached (matched or orphan_t2)
t1_pattern_array  = merged[[c + '_t1' for c in TRIGGER_BINARY_COLS]].values                          # create numpy array of just the teensy-1 trigger binary columns (those are still named 'trigger_01_binary', etc. since we left them as-is in the rename step); shape is (N_rows, 16)
t2_pattern_array  = merged[[c + '_t2' for c in TRIGGER_BINARY_COLS]].values     # same but build the teensy-2 column names on the fly by appending '_t2' to each name in TRIGGER_BINARY_COLS (those were the names after the rename step) → matching (N_rows, 16) array
patterns_agree  = np.all(t1_pattern_array == t2_pattern_array, axis=1)         # boolean array thats True where all 16 trigger binary columns match between the teensy 1 and 2, only checking in the rows where both teensies have data (has_t1 & has_t2); shape is (N_rows,); True where all 16 triggers match, False if any trigger disagrees

# adding a new column 'match_status' to merged dataframe; np.select chooses values based on whether the conditions in condlist are True for each row
merged['match_status'] = np.select(
    condlist=[
        has_t1 & has_t2 &  patterns_agree, # if this row has data from both teensies and their trigger patterns agree, then match_status is 'matched'
        has_t1 & has_t2 & ~patterns_agree, # if this row has data from both teensies but their trigger patterns disagree (in other words, at least one of the 16 scintillator triggers doesn't match), then match_status is 'pattern_mismatch'
        has_t1 & ~has_t2, # if this row has data from teensy 1 but no matching data from teensy 2 (no df2 row within tolerance), then match_status is 'orphan_t1'
        ~has_t1 & has_t2,], # if this row has data from teensy 2 but no matching data from teensy 1 (this would be the rows we appended from df2 that had no match), then match_status is 'orphan_t2'
    choicelist=['matched', 'pattern_mismatch', 'orphan_t1', 'orphan_t2'], # the corresponding values to assign to match_status for each condition
    default='unknown',
)


# 4. After match_status is assigned — full breakdown
print("\n=== MATCH STATUS BREAKDOWN ===")
print(merged['match_status'].value_counts())
print(f"Total rows: {len(merged)}")

# 5. Spot-check a few rows of each type
for status in ['matched', 'pattern_mismatch', 'orphan_t1', 'orphan_t2']:
    subset = merged[merged['match_status'] == status]
    if len(subset) > 0:
        print(f"\n-- Sample '{status}' row --")
        print(subset[['event_time_unix_s', 'match_status', '_t1_orig_idx', '_t2_orig_idx']].head(2))


# === Drop redundant teensy-2 binary columns (keeping teensy-1's as canonical) ===
# at this point, matches have been validated and time columns have been kept for drift analysis, so the teensy-2 binary cols are no longer needed and just take up space.
merged = merged.drop(columns=[c + '_t2' for c in TRIGGER_BINARY_COLS])
merged = merged.sort_values('event_time_unix_s').reset_index(drop=True) # final tidy: sort by event_time_unix_s

# 6. Final shape and a peek at the finished DataFrame
print("\n=== FINAL MERGED DATAFRAME ===")
print(f"Shape: {merged.shape}")
print(f"Columns ({len(merged.columns)} total):\n{merged.columns.tolist()}")
print("\nFirst 3 rows:")
print(merged.head(3))

# === Summary print ========================================================
print("Match status counts:")
print(merged['match_status'].value_counts())
print(f"\nTotal rows in unified DataFrame: {len(merged)}")


# === Inter-teensy drift diagnostic ========================================
# For every matched event and every trigger that fired on that event,
# compute Δt = signal_time_t1 - signal_time_t2 and aggregate.
matched_only = merged[merged['match_status'] == 'matched'] # slicing the dataframe to only include rows where there was a time match and trigger patterns agreed
drift = (matched_only['event_time_unix_s'] - matched_only['event_time_unix_s_t2']).dropna()

print("\nInter-teensy signal-time drift (t1 - t2) stats:")
print(drift.describe())
print(
    "\nHow to interpret:\n"
    "  count : number of matched events contributing (pattern_mismatch and orphans excluded)\n"
    "  mean  : average t1-minus-t2 offset in seconds; sign tells which teensy runs ahead\n"
    "          (negative -> t1 stamps earlier than t2; positive -> t2 stamps earlier than t1)\n"
    "  std   : event-to-event jitter in the offset; small std + nonzero mean = stable clock\n"
    "          offset (correctable), large std = offset itself is wandering (investigate)\n"
    "  min   : largest offset in the t1-earlier direction (most negative value)\n"
    "  max   : largest offset in the t2-earlier direction (most positive value)\n"
    "          compare |min| and |max| to MERGE_TOLERANCE -- if either is close to it,\n"
    "          some matches may be borderline and tolerance should be tightened\n"
    "  25/50/75% : distribution quartiles; if 50% (median) differs notably from mean,\n"
    "          the drift distribution is skewed or has outliers worth plotting"
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(drift, bins=100, color='steelblue', edgecolor='black')
ax.set_xlabel('Δt = event_time_unix_s_t1 - event_time_unix_s_t2 (UTC-derived, seconds, matched rows)')
ax.set_ylabel('Count')
ax.set_title('Inter-teensy signal-time drift distribution')
plt.tight_layout()
plt.show()




# === Cumulative coincidence counts per column ============================
# For each configured column, creating columns that give the cumulative count of coincidence events on any combination of the four scintillators in that row
# ie. col1_CW_1&5 counts events where trigger 1 AND trigger 5 fired on the same row (regardless of what else fired); col1_CW_1&5&9 counts events where
# trigger 1 AND trigger 5 AND trigger 9 all fired on the same row; etc.

for col_name, trigger_nums in COINCIDENCE_GROUPS.items(): # looping through the column names and list of trigger numbers defined in COINCIDENCE_GROUPS dictionary
                                                            # for each iteration col_name is the key (ie. col1) and trigger_nums is the list of trigger numbers that go in one column

    # Boolean mask: True only for fully matched rows
    matched_mask = merged['match_status'] == 'matched'

    # This will accumulate the AND condition across triggers as we loop.
    # Starts as all-True so the first AND doesn't wipe everything out.
    running_and = pd.Series(True, index=merged.index)

    # Keeps track of which triggers we've added so far, for the column name
    triggers_so_far = []

    for nn in trigger_nums:

        # On each iteration, build the name of the binary column for the current trigger number nn, e.g. if nn=1 then binary_col is 'trigger_01_binary'
        binary_col = f'sipm_{nn:02d}_trigger_t1' # _t1 suffix since all teensy1 columns are now symmetrically labeled

        # Creating boolean array that is True when trigger number nn fired (aka when binary is 1) in a given row of the merged dataframe
        this_trigger_fired = merged[binary_col] == 1

        # Creating a boolean array that is True only for rows where all the triggers we've looped through so far are true, and the row is a matched event
        # Accumulates across iterations — after iter 1: "did trigger 1 fire?", after iter 2: "did trigger 1 AND 5 fire?", etc. — so that each coincidence level builds on the last
        # while also masking to only count matched events (not orphans or pattern mismatches)
        running_and = running_and & this_trigger_fired & matched_mask

        # Add this trigger to our running list
        triggers_so_far.append(str(nn))

        # Only save a coincidence column once we have at least 2 triggers
        if len(triggers_so_far) >= 2:

            # Build the column name, e.g. 'col1_CW_1&5'
            label = '&'.join(triggers_so_far)
            new_col_name = col_name + '_CW_' + label

            # cumsum gives a running total of how many events satisfied the coincidence
            merged[new_col_name] = running_and.cumsum()

# looping through the column names and list of trigger numbers defined in COINCIDENCE_GROUPS dictionary to generate a delta column which is essentially the per-event count of how many new coincidence events of that type occurred in that row (as opposed to the cumulative count up to that row which is what the _CW_ columns give)
# for each iteration col_name is the key (ie. col1) and trigger_nums is the list of trigger numbers that go in one column
for col_name, trigger_nums in COINCIDENCE_GROUPS.items():
    for i in range(2, len(trigger_nums) + 1):
        label = '&'.join(str(n) for n in trigger_nums[:i])
        cum_col = f'{col_name}_CW_{label}'
        merged[f'delta_{cum_col}'] = merged[cum_col].diff().clip(lower=0).fillna(0)


print("\nFinal coincidence counts per column:")
for col_name, trigger_nums in COINCIDENCE_GROUPS.items():                  # print the last value of each cumulative column as a sanity check — that's the total event count for that coincidence level over the whole run
    cw_cols = [c for c in merged.columns if c.startswith(f'{col_name}_CW_')]
    for c in cw_cols:
        print(f"  {c}: {int(merged[c].iloc[-1])}")


# FINAL MERGED DATAFRAME SHOULD HAVE THE FOLLOWING HEADERS:
# event_time_unix_s, event_time_unix_s_t2, _t1_orig_idx, _t2_orig_idx, utc_time_t1, utc_time_t2, microseconds_since_boot_t1, microseconds_since_boot_t2, sipm_01_trigger_t1, sipm_02_trigger_t1, sipm_03_trigger_t1, sipm_04_trigger_t1, sipm_05_trigger_t1, sipm_06_trigger_t1, sipm_07_trigger_t1, sipm_08_trigger_t1, sipm_09_trigger_t1, sipm_10_trigger_t1, sipm_11_trigger_t1, sipm_12_trigger_t1, sipm_13_trigger_t1, sipm_14_trigger_t1, sipm_15_trigger_t1, sipm_16_trigger, trigger_01_dead_time_t1, trigger_02_dead_time_t1, trigger_03_dead_time_t1, trigger_04_dead_time_t1, trigger_05_dead_time_t1, trigger_06_dead_time_t1, trigger_07_dead_time_t1, trigger_08_dead_time_t1, trigger_09_dead_time_t1, trigger_10_dead_time_t1, trigger_11_dead_time_t1, trigger_12_dead_time_t1, trigger_13_dead_time_t1, trigger_14_dead_time_t1, trigger_15_dead_time_t1, trigger_16_dead_time_t1, trigger_01_dead_time_t2, trigger_02_dead_time_t2, trigger_03_dead_time_t2, trigger_04_dead_time_t2, trigger_05_dead_time_t2, trigger_06_dead_time_t2, trigger_07_dead_time_t2, trigger_08_dead_time_t2, trigger_09_dead_time_t2, trigger_10_dead_time_t2, trigger_11_dead_time_t2, trigger_12_dead_time_t2, trigger_13_dead_time_t2, trigger_14_dead_time_t2, trigger_15_dead_time_t2, trigger_16_dead_time_t2, sipm_01_adc, sipm_02_adc, sipm_03_adc, sipm_04_adc, sipm_05_adc, sipm_06_adc, sipm_07_adc, sipm_08_adc, sipm_09_adc, sipm_10_adc, sipm_11_adc, sipm_12_adc, sipm_13_adc, sipm_14_adc, sipm_15_adc, sipm_16_adc, sipm_01_threshold, sipm_02_threshold, sipm_03_threshold, sipm_04_threshold, sipm_05_threshold, sipm_06_threshold, sipm_07_threshold, sipm_08_threshold, sipm_09_threshold, sipm_10_threshold, sipm_11_threshold, sipm_12_threshold, sipm_13_threshold, sipm_14_threshold, sipm_15_threshold, sipm_16_threshold, cpu_temperature_t1, cpu_temperature_t2, match_status, col1_CW_1&5, col1_CW_1&5&9, col1_CW_1&5&9&13, col2_CW_2&6, col2_CW_2&6&10, col2_CW_2&6&10&14, col3_CW_3&7, col3_CW_3&7&11, col3_CW_3&7&11&15, col4_CW_4&8, col4_CW_4&8&12, col4_CW_4&8&12&16, delta_col1_CW_1&5, delta_col1_CW_1&5&9, delta_col1_CW_1&5&9&13, delta_col2_CW_2&6, delta_col2_CW_2&6&10, delta_col2_CW_2&6&10&14, delta_col3_CW_3&7, delta_col3_CW_3&7&11, delta_col3_CW_3&7&11&15, delta_col4_CW_4&8, delta_col4_CW_4&8&12, delta_col4_CW_4&8&12&16

