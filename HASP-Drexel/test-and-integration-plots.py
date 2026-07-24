import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================================================
# PLOTTING RAW DOWNLINK DATA FIRST (NOT FROM SD CARDS, DIRECTLY PLOTTING AS FUNCTION OF PACKET NUMBERS)
# =====================================================================================================

# path to CSV file, downloaded from the GUI
df = pd.read_csv('/Users/emmamartignoni/Downloads/downlink_decoded (1).csv')

# headers = ["sipm_01_trigger_count", 
#                "sipm_02_trigger_count",
#                "sipm_03_trigger_count",
#                "sipm_04_trigger_count",
#                "sipm_05_trigger_count",
#                "sipm_06_trigger_count",
#                "sipm_07_trigger_count",
#                "sipm_08_trigger_count",
#                "sipm_09_trigger_count",
#                "sipm_10_trigger_count",
#                "sipm_11_trigger_count",
#                "sipm_12_trigger_count",
#                "sipm_13_trigger_count",
#                "sipm_14_trigger_count",
#                "sipm_15_trigger_count",
#                "sipm_16_trigger_count"]

headers = [f"sipm_{i:02d}_trigger_count" for i in range(1, 17)]  # Generate headers dynamically

time_marks = [
    (524,   "8:37 am"),
    (5571,  "11:31 am"),
    (8556,  "11:41 am"),
    (11096, "11:58 am"),
    (15584, "12:28 pm"),
    (18582, "12:48 pm"),
    (22937, "1:16 pm"),
    (24942, "1:51 pm"),
    (24952, "1:56 pm"),
    (26764, "2:11 pm"),
]

output_dir = "test_and_integration_plots"
os.makedirs(output_dir, exist_ok=True)

def add_time_marks(ax):
    for pkt, label in time_marks:
        ax.axvline(pkt, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.annotate(label, xy=(pkt, 1.0), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=7, rotation=45, color='gray')


data1 = []
data2 = []

for h in headers:
    data1.append(df[h])

for h in headers:
    data2.append(df[h].cumsum()) # cumsum returns the cumulative sum of the values in the column, so at each packet #, it will give you the total 


fig_reg = plt.figure(figsize=(15, 10))
plt.title("Count versus Time")
plt.plot(df['packet_number'], data1[0], label='CH1', color='tomato', ls='-')
plt.plot(df['packet_number'], data1[1], label='CH2', color='tomato', ls='--')
plt.plot(df['packet_number'], data1[2], label='CH3', color='tomato', ls='-.')
plt.plot(df['packet_number'], data1[3], label='CH4', color='tomato', ls=':')
plt.plot(df['packet_number'], data1[4], label='CH5', color='skyblue', ls='-')
plt.plot(df['packet_number'], data1[5], label='CH6', color='skyblue', ls='--')
plt.plot(df['packet_number'], data1[6], label='CH7', color='skyblue', ls='-.')
plt.plot(df['packet_number'], data1[7], label='CH8', color='skyblue', ls=':')
plt.plot(df['packet_number'], data1[8], label='CH9', color='forestgreen', ls='-')
plt.plot(df['packet_number'], data1[9], label='CH10', color='forestgreen', ls='--')
plt.plot(df['packet_number'], data1[10], label='CH11', color='forestgreen', ls='-.')
plt.plot(df['packet_number'], data1[11], label='CH12', color='forestgreen', ls=':')
plt.plot(df['packet_number'], data1[12], label='CH13', color='purple', ls='-')
plt.plot(df['packet_number'], data1[13], label='CH14', color='purple', ls='--')
plt.plot(df['packet_number'], data1[14], label='CH15', color='purple', ls='-.')
plt.plot(df['packet_number'], data1[15], label='CH16', color='purple', ls=':')

plt.xlabel('Packet')
plt.ylabel('Counts')
plt.legend()
plt.savefig('count_vs_time.png')

fig_cum = plt.figure(figsize=(15, 10))
plt.title("Cumulative Count versus Time")
plt.plot(df['packet_number'], data2[0], label='CH1', color='tomato', ls='-')
plt.plot(df['packet_number'], data2[1], label='CH2', color='tomato', ls='--')
plt.plot(df['packet_number'], data2[2], label='CH3', color='tomato', ls='-.')
plt.plot(df['packet_number'], data2[3], label='CH4', color='tomato', ls=':')
plt.plot(df['packet_number'], data2[4], label='CH5', color='skyblue', ls='-')
plt.plot(df['packet_number'], data2[5], label='CH6', color='skyblue', ls='--')
plt.plot(df['packet_number'], data2[6], label='CH7', color='skyblue', ls='-.')
plt.plot(df['packet_number'], data2[7], label='CH8', color='skyblue', ls=':')
plt.plot(df['packet_number'], data2[8], label='CH9', color='forestgreen', ls='-')
plt.plot(df['packet_number'], data2[9], label='CH10', color='forestgreen', ls='--')
plt.plot(df['packet_number'], data2[10], label='CH11', color='forestgreen', ls='-.')
plt.plot(df['packet_number'], data2[11], label='CH12', color='forestgreen', ls=':')
plt.plot(df['packet_number'], data2[12], label='CH13', color='purple', ls='-')
plt.plot(df['packet_number'], data2[13], label='CH14', color='purple', ls='--')
plt.plot(df['packet_number'], data2[14], label='CH15', color='purple', ls='-.')
plt.plot(df['packet_number'], data2[15], label='CH16', color='purple', ls=':')

plt.xlabel('Packet')
plt.ylabel('Counts')
plt.legend()
plt.savefig('cumulative_count_vs_time.png')




# ======== RAW COUNT DATA SUBPLOTS ===================================================
# ======== 2x2 subplots where each subplot shows the 4 channels in each LAYER ========
fig_layer, axes1 = plt.subplots(2, 2, figsize=(15,10))
plt.suptitle("Layer Graphs")
axes1[0,0].set_title("Layer 1")
axes1[0,0].plot(df['packet_number'], data1[0], label='CH1', color='tomato')
axes1[0,0].plot(df['packet_number'], data1[1], label='CH2', color='skyblue')
axes1[0,0].plot(df['packet_number'], data1[2], label='CH3', color='forestgreen')
axes1[0,0].plot(df['packet_number'], data1[3], label='CH4', color='purple')
axes1[0,0].legend()

axes1[0,1].set_title("Layer 2")
axes1[0,1].plot(df['packet_number'], data1[4], label='CH5', color='tomato')
axes1[0,1].plot(df['packet_number'], data1[5], label='CH6', color='skyblue')
axes1[0,1].plot(df['packet_number'], data1[6], label='CH7', color='forestgreen')
axes1[0,1].plot(df['packet_number'], data1[7], label='CH8', color='purple')
axes1[0,1].legend()

axes1[1,0].set_title("Layer 3")
axes1[1,0].plot(df['packet_number'], data1[8], label='CH9', color='tomato')
axes1[1,0].plot(df['packet_number'], data1[9], label='CH10', color='skyblue')
axes1[1,0].plot(df['packet_number'], data1[10], label='CH11', color='forestgreen')
axes1[1,0].plot(df['packet_number'], data1[11], label='CH12', color='purple')
axes1[1,0].legend()

axes1[1,1].set_title("Layer 4")
axes1[1,1].plot(df['packet_number'], data1[12], label='CH13', color='tomato')
axes1[1,1].plot(df['packet_number'], data1[13], label='CH14', color='skyblue')
axes1[1,1].plot(df['packet_number'], data1[14], label='CH15', color='forestgreen')
axes1[1,1].plot(df['packet_number'], data1[15], label='CH16', color='purple')
axes1[1,1].legend()
for ax in axes1.flat:
    add_time_marks(ax)
plt.savefig('layer_graphs.png')

# ======== 2x2 subplots where each subplot shows the 4 channels in each COLUMN ========
fig_col, axes2 = plt.subplots(2, 2, figsize=(15,10))
plt.suptitle("Column Graphs")
axes2[0,0].set_title("Col 1")
axes2[0,0].plot(df['packet_number'], data1[0], label='CH1', color='tomato')
axes2[0,0].plot(df['packet_number'], data1[4], label='CH5', color='skyblue')
axes2[0,0].plot(df['packet_number'], data1[8], label='CH9', color='forestgreen')
axes2[0,0].plot(df['packet_number'], data1[12], label='CH13', color='purple')
axes2[0,0].legend()

axes2[0,1].set_title("Col 2")
axes2[0,1].plot(df['packet_number'], data1[1], label='CH2', color='tomato')
axes2[0,1].plot(df['packet_number'], data1[5], label='CH6', color='skyblue')
axes2[0,1].plot(df['packet_number'], data1[9], label='CH10', color='forestgreen')
axes2[0,1].plot(df['packet_number'], data1[13], label='CH14', color='purple')
axes2[0,1].legend()

axes2[1,0].set_title("Col 3")
axes2[1,0].plot(df['packet_number'], data1[2], label='CH3', color='tomato')
axes2[1,0].plot(df['packet_number'], data1[6], label='CH7', color='skyblue')
axes2[1,0].plot(df['packet_number'], data1[10], label='CH11', color='forestgreen')
axes2[1,0].plot(df['packet_number'], data1[14], label='CH15', color='purple')
axes2[1,0].legend()

axes2[1,1].set_title("Col 4")
axes2[1,1].plot(df['packet_number'], data1[3], label='CH4', color='tomato')
axes2[1,1].plot(df['packet_number'], data1[7], label='CH8', color='skyblue')
axes2[1,1].plot(df['packet_number'], data1[11], label='CH12', color='forestgreen')
axes2[1,1].plot(df['packet_number'], data1[15], label='CH16', color='purple')
axes2[1,1].legend()
for ax in axes2.flat:
    add_time_marks(ax)
plt.savefig('column_graphs.png')



# ========= CUMULATIVE COUNTS SUBPLOTS =============================================================
# ========= 2x2 subplots where each subplot shows the 4 channels in each LAYER (cumulative) ========
fig_layer_cum, axes1 = plt.subplots(2, 2, figsize=(15,10))
plt.suptitle("Layer Graphs Cumulative")
axes1[0,0].set_title("Layer 1")
axes1[0,0].plot(df['packet_number'], data2[0], label='CH1', color='tomato')
axes1[0,0].plot(df['packet_number'], data2[1], label='CH2', color='skyblue')
axes1[0,0].plot(df['packet_number'], data2[2], label='CH3', color='forestgreen')
axes1[0,0].plot(df['packet_number'], data2[3], label='CH4', color='purple')
axes1[0,0].legend()

axes1[0,1].set_title("Layer 2")
axes1[0,1].plot(df['packet_number'], data2[4], label='CH5', color='tomato')
axes1[0,1].plot(df['packet_number'], data2[5], label='CH6', color='skyblue')
axes1[0,1].plot(df['packet_number'], data2[6], label='CH7', color='forestgreen')
axes1[0,1].plot(df['packet_number'], data2[7], label='CH8', color='purple')
axes1[0,1].legend()

axes1[1,0].set_title("Layer 3")
axes1[1,0].plot(df['packet_number'], data2[8], label='CH9', color='tomato')
axes1[1,0].plot(df['packet_number'], data2[9], label='CH10', color='skyblue')
axes1[1,0].plot(df['packet_number'], data2[10], label='CH11', color='forestgreen')
axes1[1,0].plot(df['packet_number'], data2[11], label='CH12', color='purple')
axes1[1,0].legend()

axes1[1,1].set_title("Layer 4")
axes1[1,1].plot(df['packet_number'], data2[12], label='CH13', color='tomato')
axes1[1,1].plot(df['packet_number'], data2[13], label='CH14', color='skyblue')
axes1[1,1].plot(df['packet_number'], data2[14], label='CH15', color='forestgreen')
axes1[1,1].plot(df['packet_number'], data2[15], label='CH16', color='purple')
axes1[1,1].legend()
for ax in axes1.flat:
    add_time_marks(ax)
plt.savefig('layer_graphs_cumulative.png')


# ======== 2x2 subplots where each subplot shows the 4 channels in each COLUMN (cumulative) ========
fig_col_cum, axes2 = plt.subplots(2, 2, figsize=(15,10))
plt.suptitle("Column Graphs Cumulative")
axes2[0,0].set_title("Col 1")
axes2[0,0].plot(df['packet_number'], data2[0], label='CH1', color='tomato')
axes2[0,0].plot(df['packet_number'], data2[4], label='CH5', color='skyblue')
axes2[0,0].plot(df['packet_number'], data2[8], label='CH9', color='forestgreen')
axes2[0,0].plot(df['packet_number'], data2[12], label='CH13', color='purple')
axes2[0,0].legend()

axes2[0,1].set_title("Col 2")
axes2[0,1].plot(df['packet_number'], data2[1], label='CH2', color='tomato')
axes2[0,1].plot(df['packet_number'], data2[5], label='CH6', color='skyblue')
axes2[0,1].plot(df['packet_number'], data2[9], label='CH10', color='forestgreen')
axes2[0,1].plot(df['packet_number'], data2[13], label='CH14', color='purple')
axes2[0,1].legend()

axes2[1,0].set_title("Col 3")
axes2[1,0].plot(df['packet_number'], data2[2], label='CH3', color='tomato')
axes2[1,0].plot(df['packet_number'], data2[6], label='CH7', color='skyblue')
axes2[1,0].plot(df['packet_number'], data2[10], label='CH11', color='forestgreen')
axes2[1,0].plot(df['packet_number'], data2[14], label='CH15', color='purple')
axes2[1,0].legend()

axes2[1,1].set_title("Col 4")
axes2[1,1].plot(df['packet_number'], data2[3], label='CH4', color='tomato')
axes2[1,1].plot(df['packet_number'], data2[7], label='CH8', color='skyblue')
axes2[1,1].plot(df['packet_number'], data2[11], label='CH12', color='forestgreen')
axes2[1,1].plot(df['packet_number'], data2[15], label='CH16', color='purple')
axes2[1,1].legend()
for ax in axes2.flat:
    add_time_marks(ax)
plt.savefig('column_graphs_cumulative.png')

plt.show()


# =====================================================================================================
# PLOTTING SD CARD DATA
# =====================================================================================================


