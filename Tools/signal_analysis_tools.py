# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:28:44 2021

@author: Andrej Košir
"""

# File: Signal analysis tools

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sst

# Filtering
from scipy.signal import butter,filtfilt

# Slopes
from scipy.stats import linregress

# Spectrum
from scipy.fft import fft, fftfreq

# @brief a pair of start and end times 
# def get_time_int(uID, mmID, mm_dim):
    
#     # Read from file
#     return [0,1]

def getAdTimeInterval(fileName, uID):
    times_df = pd.read_csv(fileName)
    
    return [times_df.loc[times_df['uID'] == uID]['AS'],times_df.loc[times_df['uID'] == uID]['AE']]

# @brief lowpass filtering of the signal using Butterworth filter
# @arg sig_x signal data
# @arg cut_f cat-off frequency in Hz
def lowpass_1D(sig_x, cut_f):
    
    # Settings
    #T = 5.0         # Sample Period
    fs = 30.0       # sample rate, Hz
    cutoff = cut_f      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = len(sig_x) # total number of samples
        
    # Filter 
    normal_cutoff = cutoff / nyq # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    sig_lowpass_x = filtfilt(b, a, sig_x)    
    
    return sig_lowpass_x    


# @brief git 1D time signal features
# @arg sig_t input signal timestamps
# @arg sig_x input signal
# @arg code: 
#   std: standard deviation
#   exp_fit: exponential function fit
#   slope: slope of the curve
#      
def get_timesingal_feature(sig_t, sig_x, time_int = [], feature_pars = [], code='std'):
    
    if time_int != []:
        sig_x = sig_x # ToDo: cut signal
    
    if code == 'std':
        return [np.std(sig_x)]
    
    
    if code == 'slope':
        lin_reg_mod = linregress(sig_t, sig_x)
        return [lin_reg_mod.slope, lin_reg_mod.intercept]
    
    
    if code == 'spec_amp':
        spec_comp = fft(sig_x)
        spec_amp = np.abs(spec_comp)
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        #plt.plot(sig_f, spec_amp)
        
        # Define bands:
        bands_lst = feature_pars
        M = len(bands_lst)
        feat_vec = []
        feat_vec.append(sum(spec_amp[sig_f <= bands_lst[0]]))
        for ii in range(len(bands_lst)-1):
            feat_vec.append(sum(spec_amp[(bands_lst[ii] <= sig_f) & (sig_f < bands_lst[ii+1])]))
        feat_vec.append(sum(spec_amp[bands_lst[-1] <= sig_f]))
        
        return feat_vec
    
    
    if code == 'spec_phs':
        spec_comp = fft(sig_x)
        spec_phs = np.angle(spec_comp)
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        #plt.plot(sig_f, spec_amp)
        
        # Define bands:
        bands_lst = [2, 5, 10]
        M = len(bands_lst)
        feat_vec = []
        feat_vec.append(sum(spec_phs[sig_f <= bands_lst[0]]))
        for ii in range(len(bands_lst)-1):
            feat_vec.append(sum(spec_phs[(bands_lst[ii] <= sig_f) & (sig_f < bands_lst[ii+1])]))
        feat_vec.append(sum(spec_phs[bands_lst[-1] <= sig_f]))
        
        return feat_vec
    
    
    # if code == 'monotone_ints':
    #     https://docs.sympy.org/latest/modules/calculus/index.html
    
    # if code == 'exp_fit':
    #     return [1,1,1]
    
    
    
    


# @brief correlate time signal and MME
# @arg 
# @arg mm_dim
# @return r, p
# @note: 
def correlate_sigs_MME(uIDs, users, signal_name, feature_code, mm_dim, coeff_type, plotQ):
    m = len(uIDs)

    # Collect data
    x_data, y_data = np.zeros(m), np.zeros(m)
    
    for uID in uIDs: 
        x_data, y_data = users[uID][signal_name][feature_code],  users[uID][mm_dim]
        
    # if plotQ:
    #     plt.scatter(x_data, y_data)


    # Compute correlation
    if coeff_type == 'Pearson':
        r, p = sst.pearsonr(x_data, y_data)
    if coeff_type == 'KendalTau':
        r, p = sst.kendalltau(x_data, y_data)

    return r, p

# @brief correlate time signal and MME
# @arg 
# @arg mm_dim
# @return r, p
# @note: 
def correlate_sigs_MME_OnlyData(x_data, y_data, coeff_type):
    
    # Compute correlation
    if coeff_type == 'Pearson':
        r, p = sst.pearsonr(x_data, y_data)
    if coeff_type == 'KendalTau':
        r, p = sst.kendalltau(x_data, y_data)

    return r, p

# @brief visualise features and MME
def scatter_sigs_MME(uIDs, users, signal_name, feature_code, mm_dim):
    m = len(uIDs)

    # Collect data
    x_data, y_data = np.zeros(m), np.zeros(m)
    
    for uID in uIDs: 
        x_data, y_data = users[uID][signal_name][feature_code],  users[uID][mm_dim]
        
    # scatter plot
    plt.scatter(x_data, y_data)
    
    



# @brief load MM exposer scores for given users form uIDs to users dictionary
def load_MME_scores(users, uIDs):
    
    # Load scores 
    scores_full_fn = ''
    scores_pd = pd.read_csv(scores_full_fn,  index_col=[0])
    
    # Assign ti into dictionary
    for uID in uIDs:
        users[uID]['AE'] = scores_pd['AE']
        users[uID]['RE'] = 1
        users[uID]['AA'] = 1
        users[uID]['PI'] = 1
        
    return 0


# @brief 
def plot_features_MME_3D(uIDs, users, signal_name, feature_codes, mm_dim):
    
    
    
    return 1