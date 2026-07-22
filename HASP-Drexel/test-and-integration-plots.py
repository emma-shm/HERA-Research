import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

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

data1 = []
data2 = []

for h in headers:
    data1.append(df[h])

for h in headers:
    data2.append(df[h].cumsum())

fig_reg = plt.figure(figsize=(10, 6))
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

fig_cum = plt.figure(figsize=(10, 6))
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

plt.show()