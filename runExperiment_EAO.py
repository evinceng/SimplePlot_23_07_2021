# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:40:50 2021

@author: Evin Aslan Oguz
"""

import generateTimeStampedDFs
import Utils
import TimeSyncHelper
import generateSubPlots
import pandas as pd
import os.path
from pathlib import Path

# Step1 Run the 0 cell to load the uID etc. into memory

# User Settings
uID = 29
# readDataQ = True
# corr_times_setQ = False # Are correction times from acc entered?

# Folder and File Settings
rootFolder = 'D:/LivingLabMeasurements/'
usersDictFileName = "Data/usersDict.xlsx"
filesFoldersDictFileName = "Data/livinglabUsersFileFolderNames.xlsx"


# # Should be run only once in order to get the abs start time of sensors for each user.
# The generated file is already in Data folder usersAbsStartTimes.xlsx
# Utils.fillAbsStartTimesOfSensors(rootFolder, filesFoldersDictFileName)

#the output dataframes list
dfDict = {}


# %%

#Step2
#First fill the usersDict.xlsx with the clap time, start of the first video R1, start of the second video R2 ... and R4
# can convert via print(get_secs_from_str("00:06:45:080")) in the Utils.py file. Change the string and run the file, the output will be generated in sec.
#Run this cell first, read the seconds of the claps from the generated plots and write in the usersDict.xlsx
#After that go to Step3 and run the cell

# if readDataQ:
dfDict = generateTimeStampedDFs.loadData(uID, rootFolder, filesFoldersDictFileName)
# print(dfDict['shimmer_df'])
print(dfDict['empatica_ACC_df'])
# print(dfDict['shimmer_df'].head())
# print(dfDict['empatica_ACC_df'].head())
# print(dfDict['empatica_BVP_df'].head())
# print(dfDict['empatica_EDA_df'].head())
# print(dfDict['empatica_HR_df'].head())
# print(dfDict['empatica_IBI_df'].head())
# print(dfDict['empatica_TEMP_df'].head())
# print(dfDict['pupillabs_df'].head())

#plot empatica shimmer plots  for synconozation 
TimeSyncHelper.generateSynchedAbsoluteAccPlots(usersDictFileName, uID, dfDict['shimmer_df'] if "shimmer_df" in dfDict.keys() else pd.empty , dfDict['empatica_ACC_df'])

print('First cell finished... Read clap peak times from the figures and put them into usersDict.xlsx for the identified user empaticaClapPeakTime_sec, and shimmerClapPeakTime_sec.')

# subtract the peak from the red line, it is the corr time !!!!!!!!!!
print('Substract the peak times from clap time and get empaticaCorrTime_sec, and shimmerCorrTime_sec values save the file and run the second cell to produce cut dataframes.')

# %%

#Step3
outputFolder = 'output/'
Path(outputFolder + "user" + str(uID)).mkdir(parents=True, exist_ok=True)

TimeSyncHelper.generateCutDfAfterClapSync(usersDictFileName, uID, dfDict, outputFolder)

print('Second cell finished...')

#%%

# Step4 run to see subplots and video ad times.. 

out_times_lst = Utils.get_video_and_ad_times(usersDictFileName, uID)


# Select columns
empatica_name = 'empatica_ACC'
empatica_label = 'AccX'
empaticaFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + empatica_name + '_df_synced.csv'

#===============================================

shimmer_name = 'shimmer'
shimmer_label = 'Accel_LN_X_m/(s^2)'
shimmer_label2 = 'Mag_Y_local_flux'
shimmerFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + shimmer_name + '_df_synced.csv'

#first try if there is tobii data
eyeData = True

#tobii files 
pupil_name = 'tobii_pupilDim'
pupil_label = 'diameter'
pupil_gaze_name = 'tobii_gazePosition'
pupil_gaze_label = 'gp_x'

#===============================================

pupilFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + pupil_name + '_df_synced.csv'
    
#===============================================

pupilGazeFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + pupil_gaze_name + '_df_synced.csv'

#if tobii file does not exist test if tehre is pupillabs data
if not os.path.exists(pupilFilePath):
    pupil_name = 'pupillabs'
    pupil_label = 'diameter'
    pupil_gaze_name = 'pupillabs_gaze'
    pupil_gaze_label = 'gaze_point_3d_x'
    #===============================================

    pupilFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + pupil_name + '_df_synced.csv'
        
    #===============================================
    
    pupilGazeFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID) + '_' + pupil_gaze_name + '_df_synced.csv'
    
    #if tehre is no pupillabs data as well, there is no eyedata
    if not os.path.exists(pupilFilePath):
        eyeData = False

#end select comuns

yLabelDfDict = {}
index = 0

yLabelDfDict[index] = {}
yLabelDfDict[index]['sensorName'] = empatica_name
yLabelDfDict[index]['signalName'] = empatica_label
yLabelDfDict[index]['df'] = pd.read_csv(empaticaFilePath)
index += 1
print(index)

if os.path.exists(shimmerFilePath):
    shimmer_df_read = pd.read_csv(shimmerFilePath)
    yLabelDfDict[index] = {}
    yLabelDfDict[index]['sensorName'] = shimmer_name
    yLabelDfDict[index]['signalName'] = shimmer_label
    yLabelDfDict[index]['df'] = shimmer_df_read
    index += 1

    yLabelDfDict[index] = {}
    yLabelDfDict[index]['sensorName'] = shimmer_name
    yLabelDfDict[index]['signalName'] = shimmer_label2
    yLabelDfDict[index]['df'] = shimmer_df_read
    index += 1

if eyeData:    
    yLabelDfDict[index] = {}
    yLabelDfDict[index]['sensorName'] = pupil_name
    yLabelDfDict[index]['signalName'] = pupil_label
    yLabelDfDict[index]['df'] = pd.read_csv(pupilFilePath)
    index += 1
    
    yLabelDfDict[index] = {}
    yLabelDfDict[index]['sensorName'] = pupil_gaze_name
    yLabelDfDict[index]['signalName'] = pupil_gaze_label
    yLabelDfDict[index]['df'] = pd.read_csv(pupilGazeFilePath)
    index += 1

print(yLabelDfDict.keys())
saveFigFilePath = outputFolder + "user" + str(uID) + '/uID-' + str(uID)  + '_subplots.pdf'

generateSubPlots.generateSubPlotsWithAdVideoTimesOfOneUser(uID, out_times_lst, yLabelDfDict, saveFigFilePath)

