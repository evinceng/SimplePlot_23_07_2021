# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:33:13 2021

@author: Andrej Košir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Functions - can be set to another .py file
# @brief Get resampled signals and set it to users structure
# @arg users 
# @arg uID
# @arg sig_inds
# @arg common_times
# @arg sensor_lab_dc
# @arg signal_lab_dc
# @arg signal_tck_dc
# @return none
def get_and_set_resampled_sigs(users, uID, sig_inds, common_times, sensor_lab_dc, signal_lab_dc, signal_tck_dc):

    # Get resampled signals
    for sig_ind in sig_inds:
        users[uID][sensor_lab_dc[sig_ind] + '_' + signal_lab_dc[sig_ind] + '_res_times'] = common_times
        users[uID][sensor_lab_dc[sig_ind] + '_' + signal_lab_dc[sig_ind] + '_spl'] = signal_tck_dc[sig_ind]
        users[uID][sensor_lab_dc[sig_ind] + '_' + signal_lab_dc[sig_ind] + '_res_data'] = splev(common_times, signal_tck_dc[sig_ind])
        
        

# @brief Cut signals to common times
# @arg users
# @arg uID
# @arg sig_inds
# @arg common_time_delta
# @arg sensor_lab_dc
# @arg signal_ts_lab_dc
# @arg signal_df_dc
# @return common_times
# @return signal_times_dc
# @return signal_cut_data_np_dc
def signals_time_cut(users, uID, sig_inds, common_time_delta, sensor_lab_dc, signal_ts_lab_dc, signal_df_dc):

    start_cut_time_dc, stop_cut_time_dc = {}, {},
    sig_start_time_dc, sig_end_time_dc =  {}, {}
    t_S0_dc, freq_S_dc = {}, {}
    
    # Set times
    end_common_time = 10000000000 # BigM np.inf
    for sig_ind in sig_inds:
        t_S0_dc[sig_ind] = users[uID][sensor_lab_dc[sig_ind] + '_offset'] # Time ofset in seconds
        freq_S_dc[sig_ind] = sensors[sensor_lab_dc[sig_ind]]['sampl_fr'] # frequency 
        sig_start_time_dc[sig_ind] = np.min(signal_df_dc[sig_ind][signal_ts_lab_dc[sig_ind]]) # Start time
        sig_end_time_dc[sig_ind] = np.max(signal_df_dc[sig_ind][signal_ts_lab_dc[sig_ind]])  # End time
        end_common_time = np.minimum(end_common_time, sig_end_time_dc[sig_ind])
        
    # Cut data
    signal_cut_df, signal_cut_np, signal_cut_times_np, signal_cut_data_np_dc = {}, {}, {}, {}
    for sig_ind in sig_inds:
        start_cut_time_dc[sig_ind] = sig_start_time_dc[sig_ind] + t_S0_dc[sig_ind]  
        stop_cut_time_dc[sig_ind] =  sig_start_time_dc[sig_ind] + end_common_time 
        
        signal_cut_df[sig_ind] = signal_df_dc[sig_ind].loc[signal_df_dc[sig_ind][signal_ts_lab_dc[sig_ind]] > start_cut_time_dc[sig_ind]] 
        signal_cut_df[sig_ind] = signal_cut_df[sig_ind].loc[signal_cut_df[sig_ind][signal_ts_lab_dc[sig_ind]] < stop_cut_time_dc[sig_ind]]
    
        signal_cut_times_np[sig_ind] = signal_cut_df[sig_ind][signal_ts_lab_dc[sig_ind]].to_numpy()
        signal_cut_data_np_dc[sig_ind] = signal_cut_df[sig_ind][signal_data_lab_dc[sig_ind]].to_numpy()   
    
    # Transform raw times
    signal_raw_times_dc, signal_times_dc = {}, {}
    for sig_ind in sig_inds:
        signal_raw_times_dc[sig_ind] = signal_cut_df[sig_ind][signal_ts_lab_dc[sig_ind]].to_numpy()
        signal_times_dc[sig_ind] = (signal_raw_times_dc[sig_ind] - signal_raw_times_dc[sig_ind][0])
        
    
    # Get common times - at this times we resample  
    first_sig_ind = 1 # First user indeks
    time_int_len = (signal_times_dc[first_sig_ind][-1] - signal_times_dc[first_sig_ind][0]) # In Seconds
    time_int_num = int(time_int_len/common_time_delta)
    common_times = np.linspace(0, time_int_len, time_int_num)
    
    return common_times, signal_times_dc, signal_cut_data_np_dc


# @brief get spline represtntation of a given signal
# @arg sig_inds
# @arg Ts_mul
# @arg sensor_lab_dc
# @arg signal_times_dc
# @arg signal_cut_data_np_dc
# @return signal_tck_dc
def get_spline_rep(sig_inds, Ts_mul, sensor_lab_dc, signal_times_dc, signal_cut_data_np_dc):
    
    fCode = 1 # Filter code
    k = 3 # Spline order
    signal_tck_dc = {}
    for sig_ind in sig_inds:  
        print ('\n Splines: ', sensor_lab_dc[sig_ind])
        Ts = Ts_mul*np.mean([signal_times_dc[sig_ind][ii] for ii in range(100)])/10.0 # Step toward next index
        tMin, tMax = signal_times_dc[sig_ind][0], signal_times_dc[sig_ind][-1]
        signal_tck_dc[sig_ind] = interpolateBS(signal_times_dc[sig_ind], signal_cut_data_np_dc[sig_ind], tMin, tMax, k, Ts, fCode)    
    
    return signal_tck_dc



# @brief git 1D time signal features
def get_timesingal_feature(inp_sig, code):
    
    if code == 'std':
        return np.std(inp_sig)

#%% Data structure & Load data

# Users
users = {}
users[1] = {}
users[1]['real_time_offset'] = 1234.5
users[1]['shimmer_offset'] = 24.5
users[1]['empatica_offset'] = 20.5
users[1]['pupillabs_offset'] = 20.5
users[1]['hqvision_offset'] = 20.5
users[1]['shimmer_accX_raw_data'] = pd.DataFrame()
users[1]['empatica_accX_raw_data'] = pd.DataFrame()
users[1]['pupillabs_pupdiam_raw_data'] = pd.DataFrame()
users[1]['mmes_F1'] = 3.2 # Score
users[1]['mmes_F2'] = 5.7 # Score
users[1]['mmes_F3'] = 4.1 # Score
users[1]['mmes_F4'] = 1.2 # Score         

    
users[4] = {}
users[4]['real_time_offset'] = 1234.5
users[4]['shimmer_offset'] = 24.5
users[4]['empatica_offset'] = 20.5
users[4]['pupillabs_offset'] = 20.5
users[4]['hqvision_offset'] = 20.5
users[4]['shimmer_accX_raw_data'] = pd.DataFrame()
users[4]['empatica_acc_X_raw_data'] = pd.DataFrame()
users[4]['pupillabs_pupdiam_raw_data'] = pd.DataFrame()
users[4]['mme_F1'] = 3.2 # Score
users[4]['mme_F2'] = 5.7 # Score
users[4]['mme_F3'] = 4.1 # Score
users[4]['mme_F4'] = 1.2 # Score 



# Sensors
sensors = {}
sensors['shimmer'] = {}
sensors['shimmer']['sampl_fr'] = 256

sensors['empatica'] = {}
sensors['empatica']['sampl_fr'] = 32

sensors['pupillabs'] = {}
sensors['pupillabs']['sampl_fr'] = 128

sensors['hqvision'] = {}
sensors['hqvision']['sampl_fr'] = 25


# MM content
mm_contents = {}
mm_contents['72kg_dior'] = {}
mm_contents['72kg_dior']['real_time_offset'] = 1234.5
mm_contents['72kg_dior']['adstart_offset'] = 34.5
mm_contents['72kg_dior']['adend_offset'] = 134.5


# File names generation
fn = 'userID-02.??' # Everything of uID=2


# Save user
uID = 4
curr_file_name = 'userID-' + str(uID) + '.??'
#save_user(curr_file_name, users, uID)
#load_user(users, uID)
#sensor_lab_dc = 'empatica'
#load_users(users, uIDs, sensor_lab_dc)


uIDs = [1]
dataIDs = ['shimmer_accX', 'empatica_accX']
# @brief
def load_data(uIDs, dataIDs):
    
    # Generate file names!
    
    return 1


#%% Copy row data and save it to users format

'''
uID = 1
signal_code = 'empatica_accX'

curr_fn = ''
curr_df = pd.load(curr_fn)
curr_colname = ''
users[uID]['empatica_accX_raw_data'] = curr_df[curr_colname]


# Save data
'''

#%% Data integrity - plots

'''
uIDs = [1]
dataIDs = ['shimmer_accX_raw_data', 'empatica_accX_raw_data', 'pupillabs_pupdiam_raw_data']


#Load data
for uID in uIDs:
    for dataID in dataIDs:
        load_data(uIDs, dataIDs, users)


# @brief Plotd data to check the integrity
def plot_data(uIDs, dataIDs, users):
    
    return 1
'''



#%% Read data
def readAndPrintHeaders():
    timestampxLabel = 'timestamp_s'    
    shimmerylabel = "Accel_LN_X_m/(s^2)"    
    empaticaylabel = 'AccX'
    rootFolder = './'
    
    shimmerFileName = rootFolder + 'Shimmer.csv'
    empaticaFileName = rootFolder + 'Empatica.csv'
    
    shimmer_df = pd.read_csv(shimmerFileName, usecols = [timestampxLabel, shimmerylabel])
    empatica_df = pd.read_csv(empaticaFileName, usecols= [timestampxLabel, empaticaylabel])
   
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.1)
    axs = gs.subplots()
    fig.suptitle('figureTitle')
    axs[0].plot(shimmer_df[timestampxLabel], shimmer_df[shimmerylabel], label= 'Shimmer_' + shimmerylabel ) #+ '_b', color='b'
    axs[0].legend()
    axs[1].plot(empatica_df[timestampxLabel], empatica_df[empaticaylabel], label = 'Empatica_' + empaticaylabel + '_g', color='g')
    axs[1].legend()
    
    return shimmer_df, empatica_df
    
loadDataQ = True
if loadDataQ:
    shimmer_df, empatica_df = readAndPrintHeaders()


#%% Resample times


from scipy.interpolate import splev
from interpolate_funs import interpolateBS




# @brief get common times
def get_times(n, time_delta):
    return np.array()

# @brief resample signal
def resample_sig(times, inp_sig_spl, time_delta):
    return np.array()



# Do resampling

# Set users, signals, sensors and offsets
uID = 1

sensor_lab_dc, signal_lab_dc, signal_ts_lab_dc, signal_data_lab_dc, signal_df_dc = {}, {}, {}, {}, {}

sig_ind = 1 # Shimer
sensor_lab_dc[sig_ind], signal_lab_dc[sig_ind] = 'shimmer', 'accX' 
signal_ts_lab_dc[sig_ind], signal_data_lab_dc[sig_ind] = 'timestamp_s', 'Accel_LN_X_m/(s^2)' 
signal_df_dc[sig_ind] = shimmer_df

sig_ind = 2 # Empatica
sensor_lab_dc[sig_ind], signal_lab_dc[sig_ind] = 'empatica', 'accX'  # Shimer
signal_ts_lab_dc[sig_ind], signal_data_lab_dc[sig_ind] = 'timestamp_s', 'AccX' 
signal_df_dc[sig_ind] = empatica_df






# Cut data and get signal times
sig_inds = [1, 2]
common_time_delta = 1.0/24.0 # sampling frequency is 24 
common_times, signal_times_dc, signal_cut_data_np_dc = signals_time_cut(users, uID, sig_inds, common_time_delta, sensor_lab_dc, signal_ts_lab_dc, signal_df_dc)


# Step one of resempling: 
# Get spline representation
Ts_mul = 10.0 # Subsamping / filtering parameter
signal_tck_dc = get_spline_rep(sig_inds, Ts_mul, sensor_lab_dc, signal_times_dc, signal_cut_data_np_dc)


# Get resampled signals and set it to users structure - no return from this function
get_and_set_resampled_sigs(users, uID, sig_inds, common_times, sensor_lab_dc, signal_lab_dc, signal_tck_dc)    
    
    
# Plot results
fig, axs = plt.subplots(len(sig_inds), 1)
plt_ind = 0
for sig_ind in sig_inds:
    axs[plt_ind].plot(common_times, users[uID][sensor_lab_dc[sig_ind] + '_' + signal_lab_dc[sig_ind] + '_res_data'])
    axs[plt_ind].set_xlabel('Time in ms')
    axs[plt_ind].set_ylabel('Res: ' + sensor_lab_dc[sig_ind] + '_' + signal_lab_dc[sig_ind])
    axs[plt_ind].grid(True)
    plt_ind += 1
#fig.tight_layout()
plt.show()








#%% Get signal features

uID = 1
sig_ceature_code = 'shimmer_accX_featSD'
signal_code = 'shimmer_accX_res_data'
extr_code = 'std'

users[uID][sig_ceature_code] = get_timesingal_feature(users[uID][signal_code], extr_code)



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



