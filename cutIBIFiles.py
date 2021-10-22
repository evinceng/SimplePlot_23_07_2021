# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:52:57 2021

@author: evinao
"""

import Tools.signal_analysis_tools as sat
import pandas as pd
import numpy as np

contentList = ['C1', 'C2', 'C3', 'C4'] #
# uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
#               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user"
outputFolder = 'C:/Users/evinao/Documents/Paper2Data/IBIFiles_Synced_Cut/'
cutDfFolder = 'CutDF/'

userSensorContentFileName =  "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/Data/userContentSensorDict.csv"

timestamp_column = 'IBI_time'
signalName = 'IBI_dur'

sensors = {1:['empatica', {'IBI':[signalName]}]}

# arguments are seprately signal and times
def getCutSignal(sig_t, sig_p_x, time_int = []):
    if time_int == []:
        return sig_t, sig_p_x
    else:
        ad_indices = np.where((sig_t >= time_int[0]) & (sig_t <= time_int[1]))[0]
        if len(ad_indices) > 0:
            return True, sig_t[ad_indices[0]: ad_indices[-1]], sig_p_x[ad_indices[0]: ad_indices[-1]]
        else:
            return False, None, None
    

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df['uID']

for selectedContent in contentList:
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, 1, selectedContent)
    for cuID in filteredUserList:
        df = pd.read_csv(rootFolder+str(cuID)+'/uID-'+str(cuID)+'_empatica_IBI_df_synced.csv')
        df[timestamp_column] -=df[timestamp_column][df.first_valid_index()]
        df.to_csv(outputFolder+'/uID-'+str(cuID)+'_empatica_IBI_df_synced_startTime_0.csv',index=False)
        
        time_int = sat.getAdTimeInterval("Data/" + selectedContent + "_usersAdStartEndTimes.csv", cuID)
        sig_x = np.array(df[signalName])
        sig_t = np.array(df[timestamp_column])
        isNotEmpty, cut_sig_t, cut_sig_x = getCutSignal(sig_t, sig_x, time_int)
        
        #if they are both None, save them
        if isNotEmpty:
            cut_sig_t_df = pd.DataFrame(data=cut_sig_t, columns=[timestamp_column])
            cut_sig_x_df = pd.DataFrame(data=cut_sig_x, columns=[signalName])
            cut_df = pd.concat([cut_sig_t_df,cut_sig_x_df], axis=1)
            cut_df.to_csv(outputFolder+cutDfFolder+ selectedContent +'_uID-'+str(cuID)+'_empatica_IBI_df_synced_cut.csv',index=False)
        else:
            pass