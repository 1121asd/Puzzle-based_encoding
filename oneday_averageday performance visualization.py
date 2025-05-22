# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# one game day probability
df_z1 = pd.read_csv("pred_poly_multigpus_4CLASSES_10AM_t3.csv")
df_x9 = pd.read_csv("pred_9by12_multiGPUs_4CLASSES_10AM_t3.csv")
df_x3 = pd.read_csv("pred_3by4_multiGPUs_4CLASSES_10AM_t3.csv")
df_x6 = pd.read_csv("pred_6by8_multiGPUs_4CLASSES_10AM_t3.csv")
df_x12 = pd.read_csv("pred_12by16_multiGPUs_4CLASSES_10AM_t3.csv")

lz1 = df_z1.loc[300:420, '1'].tolist()
lx3 = df_x3.loc[300:420, '1'].tolist()
lx6 = df_x6.loc[300:420, '1'].tolist()
lx9 = df_x9.loc[300:420, '1'].tolist()
lx12 = df_x12.loc[300:420, '1'].tolist()


plt.figure()
#plt.plot(lz1, label='puzzle', marker='.', markersize=0.5)
plt.plot(lz1, label='Z-encoding', marker='.', markersize=0.5)
plt.plot(lx3, label='X-3by4', marker='.', markersize=0.5)
plt.plot(lx6, label='X-6by8', marker='.', markersize=0.5)
plt.plot(lx9, label='X-9by12', marker='.', markersize=0.5)
plt.plot(lx12, label='X-12by16', marker='.', markersize=0.5)

#plt.xlim(300,420)
plt.ylim(0,1)
plt.ylabel(" Probability of an game occurring on 08/14", fontsize="medium")
plt.xlabel("Time (min)", fontsize="large")
plt.plot([72,72],[0,1],linestyle='--', color='black')
plt.xticks(ticks=[0,24,48,72,96], labels=['-6h','-4h','-2h','0h','+2h'])
plt.legend()
plt.show()



# average game day
ddf_z = pd.read_csv("PEPE_poly_multiGPUs_4CLASSES_10AM_t3.csv")
ddf_x9 = pd.read_csv("PEPE_9by12_multiGPUs_4CLASSES_10AM_t3.csv")
ddf_x3 = pd.read_csv("PEPE_3by4_multiGPUs_4CLASSES_10AM_t3.csv")
ddf_x6 = pd.read_csv("PEPE_6by8_multiGPUs_4CLASSES_10AM_t3.csv")
ddf_x12 = pd.read_csv("PEPE_12by16_multiGPUs_4CLASSES_10AM_t3.csv")


plt.figure()
plt.plot(ddf_z['0'], label='Z-encoding', marker='.', markersize=0.5)
plt.plot(ddf_x3['0'], label='X-3by4', marker='.', markersize=0.5)
plt.plot(ddf_x6['0'], label='X-6by8', marker='.', markersize=0.5)
plt.plot(ddf_x9['0'], label='X-9by12', marker='.', markersize=0.5)
plt.plot(ddf_x12['0'], label='X-12by16', marker='.', markersize=0.5)

plt.xlim(0,len(ddf_z['0']))
plt.ylim(0,1)
plt.ylabel("Average probability of game occuring on game days", fontsize="medium")
plt.xlabel("Time (min)", fontsize="large")
plt.plot([48,48],[0,1],linestyle='--', color='black')
plt.xticks(ticks=[0,12,24,36,48,60,72,84], labels=['-4h','-3h','-2h','-1h','0','1h','2h','3h'])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()