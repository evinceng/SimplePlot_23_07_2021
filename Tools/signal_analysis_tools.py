# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:28:44 2021

@author: Andrej Košir
"""

# File: Signal analysis tools



# Notes
# 1. Features: 
# - https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
# - install: conda install -c conda-forge tsfresh

# 2. Image-like features:
# - Gramian Angular Field: https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_gaf.html
# - Recurrence plot: https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_rp.html
# - Markov transition fields: https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_mtf.html
# - Dynamic Time wrapping: https://pyts.readthedocs.io/en/stable/auto_examples/metrics/plot_dtw.html#sphx-glr-auto-examples-metrics-plot-dtw-py
# File: Signal analysis tools
# - install: conda install -c conda-forge pyts

# 3. Heart rate analsys: HeartPy
# - library: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
# - install: conda install -c conda-forge heartpy 
# - install: python -m pip install heartpy
# 
# 
#
#

# ToDo
# 0. Frequencies for EDA and HR
# 1. Row feature extraction
#  - "speed"
#  - "spec_pow"
#  - "volume"
#
# 2. Pupil size: blinking Py
# 3. EDA separation Py
# 4. Gaze secades and fixations
# 5. HR: PQ points
# 
#
#  - scipy.signal.periodogram, see https://www.programcreek.com/python/example/100546/scipy.signal.spectrogram
#  - 

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sst
import scipy.signal as ssig
import Utils as util
# Filtering
from scipy.signal import butter,filtfilt

# Slopes
from scipy.stats import linregress

# Spectrum
from scipy.fft import fft, fftfreq

# Features 
import tsfresh 


# 2D features
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField, RecurrencePlot
from pyts.datasets import load_gunpoint


# @brief a pair of start and end times 
# def get_time_int(uID, mmID, mm_dim):
    
#     # Read from file
#     return [0,1]

def getAdTimeInterval(fileName, uID):
    times_df = pd.read_csv(fileName)
    
    return times_df.loc[times_df['uID'] == uID]['AS'].iloc[0],times_df.loc[times_df['uID'] == uID]['AE'].iloc[0]

def getTimeIntervalofR1andR4(usersDictFileName, uID):
    out_times_lst = util.get_video_and_ad_times(usersDictFileName, uID)
    return [out_times_lst[0][0], out_times_lst[-1][-1]]
    

#argument is the dataframe
def getCutSignal_DF(df, signalName, time_int = []):
    if time_int == []:
        return df
    else:
        ad_indices = np.where(df['timestamp_s'] >= time_int[0] and df['timestamp_s'] <= time_int[1])[0]    
        return df[ad_indices[0]: ad_indices[-1]]

# arguments are seprately signal and times
def getCutSignal(sig_t, sig_p_x, time_int = []):
    if time_int == []:
        return sig_t, sig_p_x
    else:
        ad_indices = np.where((sig_t >= time_int[0]) & (sig_t <= time_int[1]))[0]        
        return sig_t[ad_indices[0]: ad_indices[-1]], sig_p_x[ad_indices[0]: ad_indices[-1]]

    
def getAvarageOfMeanStds(meanStd_df, mean_ColName, std_ColName):
    return np.mean(meanStd_df[mean_ColName]), np.mean(meanStd_df[std_ColName])

    
def getF1oF23(feat_vec):
    return feat_vec[0]/(feat_vec[1]+feat_vec[2])

def getF2oF3(feat_vec):
    return feat_vec[1]/feat_vec[2]

def getF1oF2(feat_vec):
    return feat_vec[0]/feat_vec[1]

def getFxoFy(feat_vec, x, y):
    dividing = 0
    divider = 0
    for i in x:
        dividing = dividing + feat_vec[i-1]
    for i in y:
        divider = divider + feat_vec[i-1]
        
    return dividing/divider


# @brief get peaks of a signal
# @arg sig_t time stamps of the signal
# @arg sig_x input signal
# @arg cut_f cutof frequency to smooth the signal
def get_peaks(sig_t, sig_x, cut_f):
    
    # Smooth signal
    sig_lp_x = lowpass_1D(sig_x, cut_f)    
    peaks, propos = ssig.find_peaks(sig_lp_x)
    
    return sig_t[peaks], propos


# @brief get monotone intervals of a signal
# @arg sig_t time stamps of the signal
# @arg sig_x input signal
# @arg cut_f cutof frequency to smooth the signal
# @arg monot_ints
def get_monotone_ints(sig_t, sig_x, cut_f):
    
    # Smooth signal
    sig_lp_x = lowpass_1D(sig_x, cut_f)
    
    # Get peaks
    peaks, propos = ssig.find_peaks(sig_lp_x)
    peaks_t = sig_t[peaks]
    
    # Compose intervals
    monot_ints = []
    start_t, end_t = sig_t[0], sig_t[-1]
    peaks_ss_t = np.insert(np.append(peaks_t, end_t), 0, start_t)
    for ind in range(1, len(peaks_ss_t)):
        monot_ints.append([peaks_ss_t[ind-1], peaks_ss_t[ind]])
    
    return monot_ints


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

#do this after low pass filtering

# @brief linear signal scaling 
# @arg mu mean after transformation
# @arg min minimum after transformation
# @arg max maximum after transformation
# @return transformed signal
def do_linear_transform(sig_t, sig_x, time_int =[], min_tr=0, max_tr=1):
    
    if time_int != []:
        sig_t, sig_x = getCutSignal(sig_t, sig_x, time_int)
        
    min_x, max_x = np.min(sig_x), np.max(sig_x) 
    if np.abs(max_x-min_x) < 0.001:
        k = 0
        print('k = 0000000')
    else:
        k = (max_tr - min_tr) / (max_x-min_x)
    n = min_tr - k*min_x

    return k*sig_x + n
    
#do this after low pass filtering first get scaling pasr and then use it in do_linear_transform
    
# @brief get scaling parameters
# @arg 
# @return min_x
# @return max_x 
def get_scaling_pars(sig_t, sig_x, time_int = [], feature_pars = [], code='standard', scaling_par=[]):
    
    #sig_t is not used just need for getting cut signals.
    if time_int != []:
        sig_t, sig_x = getCutSignal(sig_t, sig_x, time_int)
       
    if code == 'standard':
        
        min_percent = scaling_par[0]
        max_percent = 1.0 - min_percent
        size = len(sig_x)
        min_index = int(min_percent*size)
        max_index = int(max_percent*size)
        
        sorted_sig_x = np.sort(sig_x)
        min_val = np.mean(sorted_sig_x[0:min_index])
        max_val = np.mean(sorted_sig_x[max_index:size])
        print(min_val)
        print(max_val)
        # q = 0.05
        # min_x = np.min(sig_x) # Take average of 5% 
        # max_x = np.max(sig_x)
    
    return min_val, max_val


# @brief plot Gramian angular field
def plot_GramAF(X_gasf, X_gadf):
    
    fig = plt.figure(figsize=(8, 4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.3,
                     )
    images = [X_gasf[0], X_gadf[0]]
    titles = ['Summation', 'Difference']
    for image, title, ax in zip(images, titles, grid):
        im = ax.imshow(image, cmap='rainbow', origin='lower')
        ax.set_title(title, fontdict={'fontsize': 12})
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    plt.suptitle('Gramian Angular Fields', y=0.98, fontsize=16)
    plt.show()
    
    return 0

# @brief plot Reccurence matrix
def plot_ReccurM(X_rp):
    
    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.title('Recurrence Plot', fontsize=16)
    plt.tight_layout()
    plt.show()



# @brief git 2D time signal features
# @arg sig_t input signal timestamps
# @arg sig_x input signal
# @arg code: 
#   Gram_AF: Gramian Angular Field https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_gaf.html    
#   Recurr_M: recurrence matrix https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_rp.html
#   Markov_TF: Markov transition fields https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_mtf.html
#   Dyn_Time_W: Dynamic Time wrapping: https://pyts.readthedocs.io/en/stable/auto_examples/metrics/plot_dtw.html#sphx-glr-auto-examples-metrics-plot-dtw-py
# @return raw features as 2D numpy array    
def get_timesingal_2Dfeature(sig_t, sig_x, time_int = [], feature_pars = [], code='Gram_AF'): 
    
    if time_int != []:
        sig_t, sig_x = getCutSignal(sig_t, sig_x, time_int)
    
    if code == 'Gram_AF':
        X = sig_x.reshape(1, len(sig_x))
        gasf = GramianAngularField(image_size=24, method='summation')
        X_gasf = gasf.fit_transform(X)
        gadf = GramianAngularField(image_size=24, method='difference')
        X_gadf = gadf.fit_transform(X)
        plot_GramAF(X_gasf, X_gadf)
        return X_gasf, X_gadf
    
    
    if code == 'Recurr_M':
        rp = RecurrencePlot(threshold='point', percentage=20)
        X = sig_x.reshape(1, len(sig_x))
        X_rp = rp.fit_transform(X)
        plot_ReccurM(X_rp)
        return X_rp
    
    
    if code == 'Markov_TF':
        return []
    
    
    if code == 'Dyn_Time_W':
        return []
    

def get_derived_feature_(signal,feature, code, feature_function):
    
    if code =='num_of_peaks' and feature_function =='len':
        return len(feature[0])
    # list of peak times is feature[0]
    if code =='num_of_peaks' and feature_function =='avarage':
        return np.mean(feature[0][1:] - feature[0][:-1])


# @brief git 1D time signal features
# @arg sig_t input signal timestamps
# @arg sig_x input signal
# @arg code: 
#   std: standard deviation
#   slope: [slope, intercept] of the curve - linear fit
#   spec_amp: amplitude spectrum at bands given by feature_pars
#   spec_phs: amplitude spectrum at bands given by feature_pars
#   spec_pow: power spectrum at bands given by feature_pars
#   tot_var: total variation of the signal 
#   num_of_peaks: number of peaks of a prefitered signal 
#   exp_fit: exponential function fit
#   
# @return raw features as a 1D list      
def get_timesingal_feature(sig_t, sig_x, time_int = [], feature_pars = [], code='std'): 
    
    if time_int != []:
        sig_t, sig_x = getCutSignal(sig_t, sig_x, time_int)
    
    if code == 'std':
        return [np.std(sig_x)]
    
    if code == 'mean':
        return [np.mean(sig_x)]
    
    if code == 'kurtosis':
        return [sst.kurtosis(sig_x)]
    
    if code == 'slope':
        # print('--------------')
        # print(sig_t)        
        # print(sig_x)
        # print('--------------')
        lin_reg_mod = linregress(sig_t, sig_x)
        return [lin_reg_mod.slope, lin_reg_mod.intercept]
    
    if code == 'spec_comp':
        spec_comp = fft(sig_x)
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        
        return sig_f, spec_comp    
    
    if code == 'spec_amp':
        spec_comp = fft(sig_x)
        spec_amp = np.abs(spec_comp)
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        plt.plot(sig_f, spec_amp)
        
        # Define bands:
        # bands_lst = feature_pars
        # M = len(bands_lst)
        feat_vec = []
        # feat_vec.append(sum(spec_amp[sig_f <= bands_lst[0]]))
        # for ii in range(len(bands_lst)-1):
        #     feat_vec.append(sum(spec_amp[(bands_lst[ii] <= sig_f) & (sig_f < bands_lst[ii+1])]))
        # feat_vec.append(sum(spec_amp[bands_lst[-1] <= sig_f]))
        
        print(feat_vec)
        return feat_vec
    
    
    if code == 'spec_phs':
        spec_comp = fft(sig_x)
        spec_phs = np.angle(spec_comp)
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        plt.plot(sig_f, spec_phs)
        
        # Define bands:
        bands_lst = feature_pars
        M = len(bands_lst)
        feat_vec = []
        feat_vec.append(sum(spec_phs[sig_f <= bands_lst[0]]))
        for ii in range(len(bands_lst)-1):
            feat_vec.append(sum(spec_phs[(bands_lst[ii] <= sig_f) & (sig_f < bands_lst[ii+1])]))
        feat_vec.append(sum(spec_phs[bands_lst[-1] <= sig_f]))
        
        return feat_vec
    
    if code == 'spec_pow':
        # sig_x = sig_x - np.mean(sig_x)
        spec_comp = fft(sig_x)
        spec_pow = np.abs(spec_comp)**2
        
        N = len(sig_x)
        sample_rate = 30
        sig_f = fftfreq(N, 1 / sample_rate)
        #plt.plot(sig_f, spec_pow)
        
        # Define bands:
        bands_lst = feature_pars
        M = len(bands_lst)
        feat_vec = []
        feat_vec.append(sum(spec_pow[sig_f <= bands_lst[0]]))
        for ii in range(len(bands_lst)-1):
            feat_vec.append(sum(spec_pow[(bands_lst[ii] <= sig_f) & (sig_f < bands_lst[ii+1])]))
        feat_vec.append(sum(spec_pow[bands_lst[-1] <= sig_f]))
        
        return feat_vec


    if code == 'speed':
        
        dt = 1.0/30.0
        c_speed = (sig_x[1:]-sig_x[:-1]) / dt 
        speed = np.maximum(c_speed[1:], c_speed[:-1]) # Max of two speeds: see 2018-Kret        
        return speed


    if code == 'volume':
        
        volume = np.sum(sig_x)       
        return volume

    if code == 'periodogram':
        freqs, pow_dens = ssig.periodogram(sig_x)    
        return [freqs, pow_dens]
    
    if code == 'peaks':
        cut_f = feature_pars[0]
        peaks, props = get_peaks(sig_t, sig_x, cut_f)
        return [peaks, props]
    
    if code == 'num_of_peaks':
        cut_f = feature_pars[0]
        peaks, props = get_peaks(sig_t, sig_x, cut_f)
        return [len(peaks)]
    
    
    if code == 'monotone_ints':
        cut_f = feature_pars[0]
        monot_ints = get_monotone_ints(sig_t, sig_x, cut_f)
        return monot_ints
    
    if code == 'exp_fit':
        return [1,1,1]
    

    if code == 'total_var':
        total_var = tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(sig_x)
        return [total_var]  
    
'''
# ---------------------------------------
# Test it    
sig_x_df = pd.read_csv('test_df.csv')


# Load signals & test
signal_col = 'AccZ'
sig_x = np.array(sig_x_df[signal_col])
sig_t = np.array(sig_x_df['timestamp_s'])


# Check preprocessing
cut_f = 0.01
sig_p_x = lowpass_1D(sig_x, cut_f)


# Plot it
fig, axs = plt.subplots(2, 1)
axs[0].plot(sig_t, sig_x)
axs[1].plot(sig_t, sig_p_x)
plt.show()


cut_f = 0.01
codeIn = 'periodogram'
feat_vec = get_timesingal_feature(sig_t, sig_x, time_int = [], feature_pars = [], code=codeIn)
print ('Features: ', feat_vec)
'''

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
        effectSize = r**2
    if coeff_type == 'KendalTau':
        r, p = sst.kendalltau(x_data, y_data)
        effectSize = np.abs(r)
           
    
    return r, p, effectSize

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


# @brief gets generic effect size
# @arg x 1D array of data
# @arg y labels od groups, two classes only
def get_generic_es(x, y):
    
    labels = np.unique(y)
    M = len(labels)
    if M != 2:
        return 0
    
    x_G1 = x[y==labels[0]]
    x_G2 = x[y==labels[1]]
    
    n1, n2 = len(x_G1), len(x_G2)
    mu1, mu2 = np.mean(x_G1), np.mean(x_G2)
    sd1, sd2 = np.std(x_G1), np.std(x_G2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1*sd1 + (n2-1)*sd2*sd2) / (n1 + n2 - 2))

    gen_es = np.abs(mu2-mu1) / pooled_sd
    
    return gen_es