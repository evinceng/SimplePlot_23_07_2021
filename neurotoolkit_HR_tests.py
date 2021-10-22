# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:51:36 2021

@author: Andrej Košir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Tools.signal_analysis_tools as sat
import neurokit2 as nk
import seaborn as sns

plt.rcParams['figure.figsize'] = [15, 5]


#%% Load data
# Test it    
data_path = 'ECG_R_peaks_R2/'#'C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user1/Resampled_128/'#'Data/SmallTestData/'
data_fn = 'uID_34_HR_C1.csv'#"uID-23_empatica_HR_resampled.csv"#'test_HR_df.csv'
sig_x_df = pd.read_csv(data_path + data_fn)

# Load signals
signal_col = 'HR'
sig_x = np.array(sig_x_df[signal_col])
sig_t = np.array(sig_x_df['timestamp_s'])


#%%  Process 
# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
#ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
ecg_signal = sig_x

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=128) #3000)




# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
print(rpeaks['ECG_R_Peaks'])
# Zooming into the first 5 R-peaks
plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:20000])

code = 0
if code == 'ECG_R_Peaks':
    features = rpeaks['ECG_R_Peaks']
    

#%% Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=30, method="peak") #sampling_rate=3000, method="peak")

if code == 'ECG_T_Peaks':
    features = waves_peak['ECG_T_Peaks']

if code == 'ECG_P_Peaks':
    features = waves_peak['ECG_P_Peaks']

if code == 'ECG_Q_Peaks':
    features = waves_peak['ECG_Q_Peaks']

if code == 'ECG_S_Peaks':
    features = waves_peak['ECG_S_Peaks']
    

# Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], ecg_signal)

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                       waves_peak['ECG_P_Peaks'][:3],
                       waves_peak['ECG_Q_Peaks'][:3],
                       waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:12500])




_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="peak", show=True, show_type='peaks')




# Delineate the ECG signal and visualizing all P-peaks boundaries
signal_peak, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="peak", show=True, show_type='bounds_P')



signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="dwt", show=True, show_type='all')