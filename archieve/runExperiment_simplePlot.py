# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:39:14 2021

@author: Andrej Košir
"""

import pandas as pd
import matplotlib.pyplot as plt
import Utils


from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')

# Set user ID here adn run, the figure will be generated and saved in the output folder.
# I saved the figures should be obtained by EAO_ prefix.
uID = 36  # 1, 9


# Presettings
data_fn_dc = {}
data_fn_dc[1] = {}
data_fn_dc[1]['local_path'] = 'users/user1-Ursek/'
data_fn_dc[1]['shimmer'] = data_fn_dc[1]['local_path'] + 'uID-1_shimmer_synced_df.csv'
data_fn_dc[1]['empatica_ACC'] = data_fn_dc[1]['local_path'] + 'uID-1_empatica_ACC_synced_df.csv'


data_fn_dc[9] = {}
data_fn_dc[9]['local_path'] = 'users/user9-Andrej/'
data_fn_dc[9]['shimmer'] = data_fn_dc[9]['local_path'] + 'uID-9_shimmer_synced_df.csv'
data_fn_dc[9]['empatica_ACC'] = data_fn_dc[9]['local_path'] + 'uID-9_empatica_ACC_synced_df.csv'


data_fn_dc[36] = {}
data_fn_dc[36]['local_path'] = 'users/user36-Anja/'
data_fn_dc[36]['shimmer'] =  data_fn_dc[36]['local_path'] + 'uID-36_shimmer_synced_df.csv'
data_fn_dc[36]['empatica_ACC'] =  data_fn_dc[36]['local_path'] + 'uID-36_empatica_ACC_synced_df.csv'



# Load data
empatica_cut_df = pd.read_csv(data_fn_dc[uID]['empatica_ACC'])
shimmer_cut_df = pd.read_csv(data_fn_dc[uID]['shimmer'])


# Select column
empatica_label = 'AccX'
shimmer_label = 'Accel_LN_X_m/(s^2)'


# Get times
usersJsonFileName = 'users_1_9_36.json'
out_times_lst = Utils.get_video_and_ad_times_fromJsonFile(usersJsonFileName, uID) # Defined in Utils.py


# Plot it
fig, ax = plt.subplots(3, 1)
 

ax[0].plot(empatica_cut_df['timestamp_s'], empatica_cut_df[empatica_label])
ax[0].set_xlabel('t [s]')
ax[0].set_ylabel('empatica: ' + empatica_label)
ax[0].grid(True)
    
ax[1].plot(shimmer_cut_df['timestamp_s'], shimmer_cut_df['Mag_Y_local_flux'], c='m')
ax[1].set_xlabel('t [s]')
ax[1].set_ylabel('shimmer ' + 'Mag Y')
ax[1].grid(True)


ax[2].plot(shimmer_cut_df['timestamp_s'], shimmer_cut_df[shimmer_label])
ax[2].set_xlabel('t [s]')
ax[2].set_ylabel('shimmer ' + shimmer_label)
ax[2].grid(True)

#draw vertical lines for each of the video-ad 
for axis in ax:
    for ad in out_times_lst:
        axis.axvline(x=ad[0], c='g')
        axis.axvline(x=ad[1], c='r')
        axis.axvline(x=ad[2], c='r')
        axis.axvline(x=ad[3], c='g')
        
plt.savefig('output/Start and end of 4 videos ads with horizontal lines user' + str(uID) + '.pdf')
plt.show()