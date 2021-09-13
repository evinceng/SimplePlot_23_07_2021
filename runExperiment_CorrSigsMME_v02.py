# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:33:13 2021

@author: Andrej Košir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Tools.signal_analysis_tools as sat



#%% Functions



#%% Settings
local_data_path = 'SignalAnalysisData/'
lowpassQ = True
scatterQ = True
pVals = True

plotRawQ = True
plotScatteQ = True
plot3D = True




#%% Configuration
uID = 1
uIDs = [1, 8, 10]
mmID = 'C1'
mm_dim = 'AE'
sensor = 'empatica'
signal_name = 'ACC'
signal_col = 'AccZ'
preproc_meth = 'lowpass'
cut_f = 5 # Hertz
feature_code = 'std'
coeff_type = 'Pearson'


# Build dictionary
users = {}
for cuID in uIDs:
    users[cuID] = {}
    users[cuID][signal_name] = {}




#%% Load data
# in usersDict.xlsx: R1, R2, R3, R4 are times when video starts in MM
# in utils.py we have times inside video materials 
rel_path = 'user' + str(uID) + ' Resampled/'
file_name = 'uID-' + str(uID) + '_' + sensor + '_' + signal_name + '_resampled.csv'
sig_x_df = pd.read_csv(local_data_path + rel_path + file_name, index_col=[0]) 

# Load signals
sig_x = sig_x_df[signal_col]
sig_t = sig_x_df['timestamp_s']


# Load MMAES scores
sat.load_MME_scores(users, uIDs)




#%% Preprocessing
if lowpassQ:
    sig_p_x = sat.lowpass_1D(sig_x, cut_f)
else:
    sig_p_x = sig_x
    



# Plot raw data
if plotRawQ:
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(sig_t, sig_x, label='Original')
    axs[1].plot(sig_t, sig_p_x, label='Preprocessed')
    fig.legend()
    fig.tight_layout()
    plt.show()




    
#%% Feature extraction 
time_int = sat.get_time_int(uID, mmID, mm_dim)
sig_features_lst = sat.get_timesingal_feature(sig_t, sig_x, time_int, code=feature_code)

users[uID][signal_name][feature_code] = sat.get_timesingal_feature(sig_t, sig_x, time_int, code=feature_code)




#%% Visual inspection 
if plotScatteQ: # 2D scatter plot
    sat.scatter_sigs_MME(uIDs, users, signal_name, feature_code, mm_dim)






if plot3D:
    feature_codes = ['std', 'slope', 'spec_low']
    sat.plot_features_MME_3D(uIDs, users, signal_name, feature_codes, mm_dim)






#%% Correlate Signal to MMES
if pVals:
    r, p = sat.correlate_sigs_MME(uIDs, users, signal_name, feature_code, mm_dim, coeff_type, plotScatteQ)
    



#%% Create tables of results
#results_df = pd.DataFrame()
#results_df.to_latex(file_name)



