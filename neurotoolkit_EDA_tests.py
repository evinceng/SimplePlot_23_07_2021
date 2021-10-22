# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:51:36 2021

@author: Andrej Košir
"""

# Install: conda install neurokit2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_analysis_tools as sat
import neurokit2 as nk

plt.rcParams['figure.figsize'] = [15, 5]


#%% Load data
# Test it    
data_path = 'SmallTestData/'
data_fn = 'test_EDA_df.csv'
sig_x_df = pd.read_csv(data_path + data_fn)

# Load signals
signal_col = 'EDA'
sig_x = np.array(sig_x_df[signal_col])
sig_t = np.array(sig_x_df['timestamp_s'])


#%%  Process the raw EDA signal
eda_signal = sig_x #nk.eda_simulate(duration=10, sampling_rate=250, scr_number=3, drift=0.01)

cut_f = 0.03
sig_p_x = sat.lowpass_1D(sig_x, cut_f)
eda_signal = sig_p_x
signals, info = nk.eda_process(eda_signal, sampling_rate=30) # 250)


cleaned = signals["EDA_Clean"]
features = [info["SCR_Onsets"], info["SCR_Peaks"], info["SCR_Recovery"]]


plot = nk.events_plot(features, cleaned, color=['red', 'blue', 'orange'])

code = 0
if code == 'EDA_Tonic':
    features = np.array(signals['EDA_Tonic'])

if code == 'EDA_Phasic':
    features = np.array(signals['EDA_Phasic'])

if code == 'SCR_Onsets':
    features = np.array(signals['SCR_Onsets'])

if code == 'SCR_Height':
    features = np.array(signals['SCR_Height'])

if code == 'SCR_Amplitude':
    features = np.array(signals['SCR_Amplitude'])    

if code == 'SCR_RiseTime':
    features = np.array(signals['SCR_RiseTime'])        

if code == 'SCR_Recovery':
    features = np.array(signals['SCR_Recovery']) 

if code == 'SCR_RecoveryTime':
    features = np.array(signals['SCR_RecoveryTime']) 


#%% Decompose
data = nk.eda_phasic(nk.standardize(eda_signal), sampling_rate=250)

if code == 'EDA_Tonic':
    features = np.array(data['EDA_Tonic'])

if code == 'EDA_Phasic':
    features = np.array(data['EDA_Phasic'])


data["EDA_Raw"] = eda_signal  # Add raw signal
data.plot()


plot = nk.eda_plot(signals)