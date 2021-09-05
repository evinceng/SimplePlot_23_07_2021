# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:49:30 2021

@author: evinao
"""
import numpy as np

#%% Get signal features

uID = 1
sig_feature_code = 'std'
signal_code = 'shimmer_accX'
extr_code = 'std'

# @brief git 1D time signal features
def get_timesignal_feature(inp_sig, code):
    
    if code == 'std':
        return np.std(inp_sig)
    
    
# users[uID][sig_feature_code] = get_timesignal_feature(users[uID][signal_code], extr_code)

#%% Correlate Signal to MMES

uIDS = [1, 4] # Select users
signal_feat = 'shimmer_accX_featSD'
factorF = 'mme_F1'
coeff_type = 'Pearson'
plotQ = True



# @brief correlate time signal and MME
def correlate_sigs_MME(uIDS, factorF, signal_feat, coeff_type, plotQ):
    m = len(uIDs)

    # Collect data
    x_data, y_data = np.zeros(m), np.zeros(m)
    
    for uID in uIDs: 
        x_data, y_data = users[uID][signal_feat],  users[uID][factorF]
        
    if plotQ:
        plt.scatter(x_data, y_data)


    # Compute correlation
    if coeff_type == 'Pearson':
        return np.corr(x_data, y_data)