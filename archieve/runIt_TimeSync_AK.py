# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:46:09 2021

@author: TTTT
"""
# #Empatica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import Utils
from datetime import datetime

import dateutil
from tzlocal import get_localzone

import generateTimeStampedDFs

from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')


# Settings
root_folder = 'F:/LivingLabMeasurements/'
usersDataFileFolderNames = "Data/livinglabUsersFileFolderNames.xlsx"


def get_abs_acc(data_df, sensor_lab):
    
    if sensor_lab == 'shimmer':
        data_labs = ['Accel_LN_X_m/(s^2)', 'Accel_LN_Y_m/(s^2)', 'Accel_LN_Y_m/(s^2)']
       
        
    if sensor_lab == 'empatica':
        data_labs = ['AccX', 'AccY', 'AccZ']

        
    acc_df = pd.DataFrame((data_df[data_labs[0]]**2 + data_df[data_labs[1]]**2 + data_df[data_labs[2]]**2).apply(np.sqrt), columns=['abs_acc'])
    acc_df['timestamp_s'] = data_df['timestamp_s']
    return acc_df


#%%
# @brief cut or add zeros to time dependent signal where time label is 'timestamp_s'
# @arg data_df pandas dataframe 
# @arg time_diff_s time difference in seconds 
# @return dataframe with corrected times
def get_synced_time_df(data_df, time_diff_s):
    if time_diff_s == 0:
        return data_df
    
    if time_diff_s > 0: # Cut
        first_time_in = data_df['timestamp_s'][data_df.first_valid_index()]
        out_df = data_df[data_df['timestamp_s']>=time_diff_s+first_time_in].copy()
        first_time_synced = out_df['timestamp_s'][out_df.first_valid_index()]
        out_df['timestamp_s'] -= first_time_synced
        return out_df
    
    if time_diff_s < 0: # add zeros
        h,w = data_df.shape
        d_t = np.mean(np.array(data_df['timestamp_s'][1:10]) - np.array(data_df['timestamp_s'][0:9])) # sampling time to correct times of zeros
        was_start_time = first_time_in = data_df['timestamp_s'][data_df.first_valid_index()]
        num_of_z = len(data_df[data_df['timestamp_s']<=-time_diff_s+was_start_time])
        zeros_df = pd.DataFrame(data=np.zeros((num_of_z,w)), columns=data_df.columns)
        zeros_df['timestamp_s'] = np.linspace(was_start_time+time_diff_s-d_t, was_start_time-d_t, num_of_z)
        #data_df['timestamp_s'] += -time_diff_s # Correct times of original data
        out_df = pd.concat((zeros_df, data_df), axis=0, ignore_index=True)
        out_df['timestamp_s'] -= out_df['timestamp_s'][out_df.first_valid_index()]
        return out_df
'''    
t_s, t_e, f_s = 1.21, 5, 30
x_df = pd.DataFrame(data=np.vstack((np.linspace(t_s,t_e,int(f_s*(t_e-t_s))), np.array(range(1, int(f_s*(t_e-t_s)+1))), 10*np.array(range(1, int(f_s*(t_e-t_s)+1))))).T, columns=['timestamp_s', 'y_d', 'z_d'])
x_synced_df = get_synced_time_df(x_df, -1.5)

plt.figure()
n_o_p = len(x_synced_df['timestamp_s'])
plt.plot(range(n_o_p),x_synced_df['timestamp_s'])
plt.show()
''' 


# @brief Reads time stamps from file names and folder names. Dictionary
#   variable users is global
#   Times are in the form hh:mm:ss:ms. Can include - like '-00:00:01:919'
# @arg uID users id
# @ret time strings set to user dictionarry
def read_times_from_fdn(uID):
    
    users[uID]['empatica_abs_start_time'] = '12:57:01'
    users[uID]['shimmer_abs_start_time'] = '13:00:36'
    users[uID]['pupillabs_abs_start_time'] = '13:06:43'
    users[uID]['hqvision_abs_start_time'] = '13:04:55'
    users[uID]['kinect_abs_start_time'] = ''
    
    return 0


# @brief Reads time stamps from a predefined table. Dictionary
#   variable users is global. 
#   Times are in the form hh:mm:ss:ms. Can include - like '-00:00:01:919'
# @arg uID users id
# @ret time strings set to user dictionarry
def read_times_from_synctable(uID):

    # Open synctable file
    
    
    # read and fill times
    users[uID]['hqvision_abs_start_time'] = '13:04:55'
    users[uID]['pupillabs_abs_start_time'] = '13:06:43'
    users[uID]['kinect_abs_start_time'] = ''
    users[uID]['clap_time'] = '00:02:54:800'
    users[uID]['R1'] = ''
    users[uID]['R2'] = ''
    users[uID]['R3'] = ''
    users[uID]['R4'] = ''
    users[uID]['empatica_corr_time'] = '00:00:01:919'
    users[uID]['shimmer_corr_time'] = '00:00:01:712'

    
    return 0

#%% On MM

mm_seq = {}
mm_seq['Box1'] = ['C4', 'C3', 'C2', 'C1']
mm_seq['Box4'] = ['C1', 'C2', 'C3', 'C4']
mm_seq['Box2'] = ['C3', 'C1', 'C4', 'C2']
mm_seq['Box3'] = ['C2', 'C4', 'C1', 'C3']

video_len = {} 
video_len['C1'] = '00:04:04'
video_len['C2'] = '00:03:48'
video_len['C3'] = '00:03:54'
video_len['C4'] = '00:03:38'

ad_times = {}
ad_times['C1'] = ['00:00:50:060', '00:01:56:080']
ad_times['C2'] = ['00:01:57:230', '00:02:29:280']
ad_times['C3'] = ['00:02:08:010', '00:02:43:120']
ad_times['C4'] = ['00:00:43:090', '00:01:58:170']


#%% Step 1: file names --------------------------------------------------------

root_folder = 'F:/LivingLabMeasurements/'#'C:/OutOfDropbox/01-TestDBs/2021-KognitivnoVajenistvo/'

data_fn_dc = {}
data_fn_dc[1] = {}
data_fn_dc[1]['local_path'] = 'user1/' #'user1-Ursek/'
data_fn_dc[1]['shimmer_path'] = 'Shimmer/'
data_fn_dc[1]['shimmer_fn'] = '20210507130036 Device5E7F.csv'
data_fn_dc[1]['empatica_path'] = 'Empatica/1620385021_A0264A/'
data_fn_dc[1]['empatica_fn'] = 'ACC.csv'
data_fn_dc[1]['pupillabs_path'] = 'Pupillabs/'
data_fn_dc[1]['pupillabs_fn'] = 'pupil_positions.csv'

data_fn_dc[9] = {}
data_fn_dc[9]['local_path'] = 'user9/' #'user9-Andrej/'
data_fn_dc[9]['shimmer_path'] = 'Shimmer/'
data_fn_dc[9]['shimmer_fn'] = '20210514111855 Device5E7F.csv'
data_fn_dc[9]['empatica_path'] = 'Empatica/1620983849_A0264A/'
data_fn_dc[9]['empatica_fn'] = 'ACC.csv'
data_fn_dc[9]['pupillabs_path'] = 'Pupillabs/'
data_fn_dc[9]['pupillabs_fn'] = 'pupil_positions.csv'

data_fn_dc[36] = {}
data_fn_dc[36]['local_path'] = 'user36/' #'user36-Anja/'
data_fn_dc[36]['shimmer_path'] = 'Shimmer/'
data_fn_dc[36]['shimmer_fn'] = '20210506114816 Device5E7F.csv'
data_fn_dc[36]['empatica_path'] = 'Empatica/1620294325_A0264A/'
data_fn_dc[36]['empatica_fn'] = 'ACC.csv'
data_fn_dc[36]['pupillabs_path'] = 'Pupillabs/'
data_fn_dc[36]['pupillabs_fn'] = 'pupil_positions.csv'




#%% Step 2: Set time stamps from data except *_corr_time ------------------------------------------------------------------
# 


users = {}

uID = 1 # Uršek
users[uID] = {}
users[uID]['Box'] = 'Box1'
users[uID]['R1'] = '00:02:53:700' # Relative to what (EAO: relative to start of the world video)
users[uID]['R2'] = '00:10:50:233'
users[uID]['R3'] = '00:18:23:066'
users[uID]['R4'] = '00:25:32:133'
users[uID]['empatica_abs_start_time'] = '12:57:01'
users[uID]['empatica_corr_time'] = '00:00:01:919'
users[uID]['shimmer_abs_start_time'] = '13:00:36'
users[uID]['shimmer_corr_time'] = '00:00:01:712'
users[uID]['pupillabs_abs_start_time'] = '13:06:43'
users[uID]['clap_time'] = '00:02:54:800'
users[uID]['hqvision_abs_start_time'] = '13:04:55'
users[uID]['kinect_abs_start_time'] = ''


uID = 9 # Andrej
users[uID] = {}
users[uID]['Box'] = 'Box2'
users[uID]['R1'] = '00:04:58:400' # Relative to what (EAO: relative to hikvision)
users[uID]['R2'] = '00:18:39:633'
users[uID]['R3'] = '00:29:27:266'
users[uID]['R4'] = '00:36:44:700'
users[uID]['empatica_abs_start_time'] = '11:17:29'
users[uID]['empatica_corr_time'] = '00:00:03:542'
users[uID]['shimmer_abs_start_time'] = '11:18:55'
users[uID]['shimmer_corr_time'] = '00:00:08:605'
users[uID]['pupillabs_abs_start_time'] = '11:22:19'
users[uID]['clap_time'] = '00:27:54:833' # From Kinect video = '00:27:43:010', # From HK vision = '00:27:54:833' 
users[uID]['hqvision_abs_start_time'] = '11:22:39' # Checked
users[uID]['kinect_abs_start_time'] = '11:22:55' # Checked

uID = 36 # Anja
users[uID] = {}
users[uID]['Box'] = 'Box4'
users[uID]['R1'] = '00:06:44:800' # Relative to what (EAO: relative to start of the world video)
users[uID]['R2'] = '00:18:57:600'
users[uID]['R3'] = '00:34:56:166'
users[uID]['R4'] = '00:44:38:500'
users[uID]['empatica_abs_start_time'] = '11:45:25'
users[uID]['empatica_corr_time'] = '00:00:03:639'
users[uID]['shimmer_abs_start_time'] = '11:48:16'
users[uID]['shimmer_corr_time'] = '00:00:03:022'
users[uID]['pupillabs_abs_start_time'] = '11:56:42'
users[uID]['clap_time'] = '00:06:45:080' # Ref from video
users[uID]['hqvision_abs_start_time'] = '11:58:11'
users[uID]['kinect_abs_start_time'] = '' # Not available

# Save users
saveUsersQ = True
if saveUsersQ:
    json.dump( users, open( "users_1_9_36.json", 'w' ) )

# Read
#users = json.load( open( "users_1_9_36.json" ) )

# =============================================================================
# Settings
uID = 1
readDataQ = True
corr_times_setQ = False # Are correction times from acc entered?
sincTo_PupliLabsVideoQ = True # Sync to Pupillabs
sincTo_HKVisionVideoQ = False # Sync to HKvision


# Step 3: Set uID at Load data , run this script with corr_times_setQ = False to get time graphs
# Step 4: Read correction times (near red lines) from Python time graphs and write them to 
#  users[uID]['empatica_corr_time'] = 
#  users[uID]['shimmer_corr_time'] = 
# above. If red lines are off any spike, there is a problem. 
# Step 5: Set corr_times_setQ = True and run Step 5: Get synchronised files bellow  

#%% Load data
if readDataQ:
    shimmer_df, empatica_ACC_df, empatica_BVP_df, empatica_EDA_df, empatica_HR_df, empatica_IBI_df, empatica_TEMP_df, pupillabs_df = generateTimeStampedDFs.loadData(uID, root_folder, usersDataFileFolderNames) 
    

#%% Preprocess it

# Get absolute accelerations 
sensor_lab = 'shimmer'
shimmer_abs_acc_df = get_abs_acc(shimmer_df, sensor_lab)

sensor_lab = 'empatica'
empatica_abs_acc_df = get_abs_acc(empatica_ACC_df, sensor_lab)


# Convert times
empatica_abs_start_time_s = Utils.get_secs_from_str(users[uID]['empatica_abs_start_time'])
shimmer_abs_start_time_s = Utils.get_secs_from_str(users[uID]['shimmer_abs_start_time'])
pupillabs_abs_start_time_s = Utils.get_secs_from_str(users[uID]['pupillabs_abs_start_time'])
hkvision_abs_start_time_s = Utils.get_secs_from_str(users[uID]['hqvision_abs_start_time'])
rel_clap_time_s = Utils.get_secs_from_str(users[uID]['clap_time'])
start_times_lst = [empatica_abs_start_time_s, shimmer_abs_start_time_s, pupillabs_abs_start_time_s]







if sincTo_PupliLabsVideoQ:
    empatica_synced_abs_acc_df = get_synced_time_df(empatica_abs_acc_df, pupillabs_abs_start_time_s - empatica_abs_start_time_s)
    shimmer_synced_abs_acc_df = get_synced_time_df(shimmer_abs_acc_df, pupillabs_abs_start_time_s - shimmer_abs_start_time_s)
if sincTo_HKVisionVideoQ:
    empatica_synced_abs_acc_df = get_synced_time_df(empatica_abs_acc_df, hkvision_abs_start_time_s - empatica_abs_start_time_s)
    shimmer_synced_abs_acc_df = get_synced_time_df(shimmer_abs_acc_df, hkvision_abs_start_time_s - shimmer_abs_start_time_s)    


# Plot accelerations
fig, ax = plt.subplots(2, 1)

plt.title('uID =' + str(uID))

ax[0].plot(empatica_synced_abs_acc_df['timestamp_s'], empatica_synced_abs_acc_df['abs_acc'])
ax[0].axvline(x=rel_clap_time_s, c='r')
ax[0].set_xlabel('t [s]')
ax[0].set_ylabel('empatica abs acc')
ax[0].grid(True)

ax[1].plot(shimmer_synced_abs_acc_df['timestamp_s'], shimmer_synced_abs_acc_df['abs_acc'])
ax[1].axvline(x=rel_clap_time_s, c='r')
ax[1].set_xlabel('t [s]')
ax[1].set_ylabel('shimmer abs acc')
ax[1].grid(True)

plt.show()



#%% Step 5: Get synchronised files
if corr_times_setQ:
    #uID = 36
    
    # Cut dataframes 
    if sincTo_PupliLabsVideoQ:
        empatica_corr_time = pupillabs_abs_start_time_s - empatica_abs_start_time_s + Utils.get_secs_from_str(users[uID]['empatica_corr_time'])
        shimmer_corr_time = pupillabs_abs_start_time_s - shimmer_abs_start_time_s + Utils.get_secs_from_str(users[uID]['shimmer_corr_time'])
    if sincTo_HKVisionVideoQ:
        empatica_corr_time = hkvision_abs_start_time_s - empatica_abs_start_time_s + Utils.get_secs_from_str(users[uID]['empatica_corr_time'])
        shimmer_corr_time = hkvision_abs_start_time_s - shimmer_abs_start_time_s + Utils.get_secs_from_str(users[uID]['shimmer_corr_time'])
        
    empatica_ACC_synced_df = get_synced_time_df(empatica_ACC_df, empatica_corr_time)
    empatica_BVP_synced_df = get_synced_time_df(empatica_BVP_df, empatica_corr_time)
    empatica_EDA_synced_df = get_synced_time_df(empatica_EDA_df, empatica_corr_time)
    empatica_HR_synced_df = get_synced_time_df(empatica_HR_df, empatica_corr_time)
    empatica_TEMP_synced_df = get_synced_time_df(empatica_TEMP_df, empatica_corr_time)
    empatica_IBI_synced_df = empatica_IBI_df    
    empatica_IBI_synced_df['IBI_time'] += empatica_corr_time  # Just move by correction time 
    
    shimmer_synced_df = get_synced_time_df(shimmer_df, shimmer_corr_time)
    pupillabs_synced_df = pupillabs_df # Zero time is from pupil labs if vailable

    # Save files
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_ACC_synced_df.csv'
    empatica_ACC_synced_df.to_csv(empatica_synced_fn)
    
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_BVP_synced_df.csv'
    empatica_BVP_synced_df.to_csv(empatica_synced_fn)
    
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_EDA_synced_df.csv'
    empatica_EDA_synced_df.to_csv(empatica_synced_fn)
    
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_HR_synced_df.csv'
    empatica_HR_synced_df.to_csv(empatica_synced_fn)
    
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_IBI_synced_df.csv'
    empatica_IBI_synced_df.to_csv(empatica_synced_fn)
    
    empatica_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'empatica_TEMP_synced_df.csv'
    empatica_TEMP_synced_df.to_csv(empatica_synced_fn)
    
    shimmer_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'shimmer_synced_df.csv'
    shimmer_synced_df.to_csv(shimmer_synced_fn)
    
    pupillabs_synced_fn = root_folder + data_fn_dc[uID]['local_path'] + 'uID-' + str(uID) + '_' + 'pupillabs_synced_df.csv'
    pupillabs_synced_df.to_csv(pupillabs_synced_fn)
    
    

#%% Resample and Join 
# - times above are generated from start time and different sampling frequency. 
#- after cutting / synchronizing, all times start at 0 secs and we can resample them 
#  to the highest sampling rate which is 30 per second. 

