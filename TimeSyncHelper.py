# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:21:43 2021

@author: Evin Aslan Oguz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Utils

def get_abs_acc(data_df, sensor_lab):
    
    if sensor_lab == 'shimmer':
        data_labs = ['Accel_LN_X_m/(s^2)', 'Accel_LN_Y_m/(s^2)', 'Accel_LN_Y_m/(s^2)']
       
        
    if sensor_lab == 'empatica':
        data_labs = ['AccX', 'AccY', 'AccZ']

        
    acc_df = pd.DataFrame((data_df[data_labs[0]]**2 + data_df[data_labs[1]]**2 + data_df[data_labs[2]]**2).apply(np.sqrt), columns=['abs_acc'])
    acc_df['timestamp_s'] = data_df['timestamp_s']
    return acc_df

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
def saveSynchedDFToCsvFile(data_df, rootFolder, uID, fileNameKeyword):      
    synced_fn = rootFolder + uID + '/uID-' + str(uID) + '_' + fileNameKeyword + '_synced_df.csv'
    data_df.to_csv(synced_fn)
    
    

def generateSynchedAbsoluteAccPlots(usersDictFileName, uID, shimmer_df, empatica_ACC_df):
    
    users = Utils.readDict(usersDictFileName)
    
    if users[uID]['isClap0Pupil1Hikvision2Tobii'] == 0:
        refAbsStartTime = users[uID]['pupillabsAbsStartTime_sec']
    elif users[uID]['isClap0Pupil1Hikvision2Tobii'] == 1:
        refAbsStartTime = users[uID]['hikVisionAbsStartTime_sec']
    else: #users[uID]['isClap0Pupil1Hikvision2Tobii'] == 2:
        refAbsStartTime = users[uID]['tobiiAbsStartTime_sec']
    
    # Get absolute accelerations 
    if not shimmer_df.empty:
        shimmer_abs_acc_df = get_abs_acc(shimmer_df, 'shimmer')
    empatica_abs_acc_df = get_abs_acc(empatica_ACC_df, 'empatica')
    
    
    # Convert times
    # empatica_abs_start_time_s = users[uID]['empaticaAbsStartTime_sec']
    # shimmer_abs_start_time_s = users[uID]['shimmerAbsStartTime_sec']
    rel_clap_time_s = Utils.get_secs_from_str(users[uID]['clapTime'])
       
    
    empatica_synced_abs_acc_df = get_synced_time_df(empatica_abs_acc_df, refAbsStartTime - users[uID]['empaticaAbsStartTime_sec'])
    if not shimmer_df.empty:
        shimmer_synced_abs_acc_df = get_synced_time_df(shimmer_abs_acc_df, refAbsStartTime - users[uID]['shimmerAbsStartTime_sec'])
   
    
    # Plot accelerations
    fig, ax = plt.subplots(2, 1)
    
    plt.title('uID =' + str(uID))
    
    ax[0].plot(empatica_synced_abs_acc_df['timestamp_s'], empatica_synced_abs_acc_df['abs_acc'])
    ax[0].axvline(x=rel_clap_time_s, c='r')
    ax[0].set_xlabel('t [s]')
    ax[0].set_ylabel('empatica abs acc')
    ax[0].grid(True)
    
    if not shimmer_df.empty:
        ax[1].plot(shimmer_synced_abs_acc_df['timestamp_s'], shimmer_synced_abs_acc_df['abs_acc'])
        ax[1].axvline(x=rel_clap_time_s, c='r')
        ax[1].set_xlabel('t [s]')
        ax[1].set_ylabel('shimmer abs acc')
        ax[1].grid(True)
    
    plt.show()


# Cut dataframes
def generateCutDfAfterClapSync(usersDictFileName, uID, dfDict, rootFolder):
        
    users = Utils.readDict(usersDictFileName)
    
    
    if users[uID]['isClap0Pupil1Hikvision2Tobii'] == 0:
        refAbsStartTime = users[uID]['pupillabsAbsStartTime_sec']
    elif users[uID]['isClap0Pupil1Hikvision2Tobii'] == 1:
        refAbsStartTime = users[uID]['hikVisionAbsStartTime_sec']
    else: #users[uID]['isClap0Pupil1Hikvision2Tobii'] == 2:
        refAbsStartTime = users[uID]['tobiiAbsStartTime_sec']
        
    # if users[uID]['isClapTimeHikvision']:
    #     refAbsStartTime = users[uID]['hikVisionAbsStartTime_sec']
    # else:
    #     refAbsStartTime = users[uID]['pupillabsAbsStartTime_sec']
    
    empatica_corr_time = refAbsStartTime - users[uID]['empaticaAbsStartTime_sec'] + users[uID]['empaticaCorrTime_sec']
    
    print(users[uID]['empaticaCorrTime_sec'])
    
    for currItemKey in dfDict.keys():
        if 'tobii' in currItemKey:
            corrTime = refAbsStartTime - Utils.get_secs_from_str(dfDict[currItemKey]["utc_time"].iloc[0])
            synchedDf = get_synced_time_df(dfDict[currItemKey], corrTime)            
        elif currItemKey == 'shimmer_df':
            if dfDict[currItemKey].empty:
                continue
            
            print(users[uID]['shimmerCorrTime_sec'])
            shimmer_corr_time = refAbsStartTime - users[uID]['shimmerAbsStartTime_sec'] + users[uID]['shimmerCorrTime_sec']
            synchedDf = get_synced_time_df(dfDict[currItemKey], shimmer_corr_time)
            
        elif currItemKey == 'pupillabs_df' or currItemKey == 'pupillabs_gaze_df':
            if users[uID]['isClap0Pupil1Hikvision2Tobii'] == 1 and not pd.isna( users[uID]['pupillabsCorrTime_sec']):
                pupillabs_corr_time =  users[uID]['pupillabsCorrTime_sec'] #refAbsStartTime - users[uID]['pupillabsAbsStartTime_sec'] +
                synchedDf = get_synced_time_df(dfDict[currItemKey], pupillabs_corr_time)
                
                print("The pupil labs corrections time is ")
                print(pupillabs_corr_time)
            else:
                synchedDf = dfDict[currItemKey] # Zero time is from pupil labs if available
        elif currItemKey == 'empatica_IBI_df':
            synchedDf = dfDict[currItemKey]
            synchedDf['IBI_time'] += empatica_corr_time  # Just move by correction time
        else: # Empatica other files except IBI
            synchedDf = get_synced_time_df(dfDict[currItemKey], empatica_corr_time)
        # save to file    
        synced_fn = rootFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + currItemKey + '_synced.csv'
        synchedDf.to_csv(synced_fn)
    

    