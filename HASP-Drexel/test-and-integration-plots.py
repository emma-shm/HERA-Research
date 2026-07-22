import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/emmamartignoni/Downloads/downlink_decoded (1).csv')


headers = ["sipm_01_trigger_count", 
               "sipm_02_trigger_count",
               "sipm_03_trigger_count",
               "sipm_04_trigger_count",
               "sipm_05_trigger_count",
               "sipm_06_trigger_count",
               "sipm_07_trigger_count",
               "sipm_08_trigger_count",
               "sipm_09_trigger_count",
               "sipm_10_trigger_count",
               "sipm_11_trigger_count",
               "sipm_12_trigger_count",
               "sipm_13_trigger_count",
               "sipm_14_trigger_count",
               "sipm_15_trigger_count",
               "sipm_16_trigger_count"]

data = []

for h in headers:
    data.append(df[h].cumsum())

fig_alt = plt.figure(figsize=(10, 6))
plt.plot(df['packet_number'], data[0], label='CH1')
plt.plot(df['packet_number'], data[1], label='CH2')
plt.plot(df['packet_number'], data[2], label='CH3')
plt.plot(df['packet_number'], data[3], label='CH4')
plt.plot(df['packet_number'], data[4], label='CH5')
plt.plot(df['packet_number'], data[5], label='CH6')
plt.plot(df['packet_number'], data[6], label='CH7')
plt.plot(df['packet_number'], data[7], label='CH8')
plt.plot(df['packet_number'], data[8], label='CH9')
plt.plot(df['packet_number'], data[9], label='CH10')
plt.plot(df['packet_number'], data[10], label='CH11')
plt.plot(df['packet_number'], data[11], label='CH12')
plt.plot(df['packet_number'], data[12], label='CH13')
plt.plot(df['packet_number'], data[13], label='CH14')
plt.plot(df['packet_number'], data[14], label='CH15')
plt.plot(df['packet_number'], data[15], label='CH16')

plt.xlabel('Packet')
plt.ylabel('Counts')
plt.legend()

plt.show()