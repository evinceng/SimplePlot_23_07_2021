# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:04:36 2021

@author: evinao
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:23:54 2021

@author: evinao
"""
import os.path
from pathlib import Path
import pandas as pd
import Tools.signal_analysis_tools as sat
import matplotlib.pyplot as plt
import pickle
import Utils
import numpy as np
import scipy.stats as sst
import tabulate
from tabulate import Texttable

outputFolder = "SignalCorrelation/"

factorScoresFileName = "Data/MMEM_Scores_FNames/mmem_C1_argmin_F_df.csv"
selectedFactor = 'AE'
selectedContent = 'C1'

rootFolder = "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user"
usersDictFileName = "Data/usersDict.xlsx"


userSensorContentFileName =  "Data/userContentSensorDict.csv"

empaticaSignals = { 'ACC':['AccX', 'AccY', 'AccZ'], 'EDA':['EDA'], 'BVP':['BVP'], 'HR':['HR']}

shimmerSignals = {'':['Accel_LN_X_m/(s^2)', 'Accel_LN_Y_m/(s^2)',
           'Accel_LN_Z_m/(s^2)', #'GSR_Range_no_units.1', #I don't have it since not resampled includes nounits 
           'GSR_Skin_Conductance_microSiemens', 'GSR_Skin_Conductance_uS.1',
           'Pressure_BMP280_kPa', 'PPG_A12_mV',
           'Temperature_BMP280_Degrees Celsius', 'GSR_Skin_Resistance_kOhms',
           'Gyro_Z_deg/s', 'Gyro_X_deg/s', 'Gyro_Y_deg/s',
           'Accel_WR_Z_m/(s^2)', 'Accel_WR_Y_m/(s^2)', 'Accel_WR_X_m/(s^2)']}

pupillabsSignals = {'left_eye_2d':['norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle'],
                    'left_eye_3d':['norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle',
           'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z'],
                    'right_eye_2d':['norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle'],
                    'right_eye_3d':['norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle',
           'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z'],
                    'gaze': ['norm_pos_x', 'norm_pos_y', 'gaze_point_3d_x', 'gaze_point_3d_y',
           'gaze_point_3d_z', 'gaze_normal0_x', 'gaze_normal0_y',
           'gaze_normal0_z', 'gaze_normal1_x', 'gaze_normal1_y',
           'gaze_normal1_z']}
           
tobiiSignals = {'accelerometer': ['ac_x', 'ac_y', 'ac_z'], 
                'gazeDirection_left_eye' : ['gd_x', 'gd_y', 'gd_z'],
                'gazeDirection_right_eye' : ['gd_x', 'gd_y', 'gd_z'],
           'gazePosition': ['gp_x', 'gp_y', 'gp_latency'],
           'gazePosition3D': ['gp3d_x', 'gp3d_y', 'gp3d_z'],
           'gyroscope' :['gy_x', 'gy_y', 'gy_z'], 
           'pupilCenter_left_eye' :['pc_x', 'pc_y', 'pc_z'],
           'pupilCenter_right_eye' :['pc_x', 'pc_y', 'pc_z'],
           'pupilDim_left_eye':['diameter'],
           'pupilDim_right_eye':['diameter']}

# sensors = {1:['empatica', empaticaSignals] , 2:['shimmer', shimmerSignals],
#            3:['tobii',tobiiSignals],  4:['pupillabs', pupillabsSignals]}

signalName = 'EDA'
sensorID = 1

sensors = {1:['empatica', {'EDA':['EDA']}]}

# Settings

lowpassQ = True
scatterQ = True
pVals = True

plotRawQ = False
plotScatteQ = True
plot3D = False

# Configuration

preproc_meth = 'lowpass'
cut_f = 5 # Hertz

feature_codes = ['std', 'slope', 'spec_amp', 'spec_phs']
feature_code = 'std'
feature_pars = []
coeff_type = 'Pearson'


isOnlyLowHighScoreUsers = False

#%% Functions

def getAllUsersScores(userIDList, factorScoresFileName, selectedFactor):
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['userID'].isin(userIDList)] 
    return scores_df[['userID', selectedFactor]]

def getAllFilteredUserScores(outputFolder, selectedFactor, contentID, sensors, sensorID, userSensorContentFileName, factorScoresFileName):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    allUsers = getAllUsersScores(filteredUserList, factorScoresFileName, selectedFactor)
    return allUsers
    
def getLowAndHighFactorUserIDS(userIDList, factorScoresFileName, selectedFactor, numberOfUsers=4):
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['userID'].isin(userIDList)] 
    sorted_df = scores_df.sort_values(selectedFactor)
    
    return sorted_df[['userID', selectedFactor]][:numberOfUsers], sorted_df[['userID', selectedFactor]][-numberOfUsers:]
    

# print(fileNameGenerator(1, 2))

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df['userID']


def getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, contentID, sensors, sensorID, userSensorContentFileName, factorScoresFileName, numberOfUsers=6):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, factorScoresFileName, selectedFactor, numberOfUsers)
    return lowFactorUserIDs, highFactorUserIDs
        

def fileNameGenerator(userID, sensorID, sensorFileNameExt = ""):
    fileNameStr = rootFolder + str(userID) + "/Resampled/uID-" + str(userID) + "_" + sensors[sensorID][0]
    if sensorFileNameExt:
        fileNameStr = fileNameStr + '_'  + sensorFileNameExt + '_resampled.csv'
    else:
        fileNameStr = fileNameStr + '_resampled.csv'
        
    return fileNameStr
    
def readSignal(fileName, signalName):
    df = pd.read_csv(fileName)
    
    return df[['timestamp_s',signalName]]

def getSignalDf(userID, sensorID, signalName, sensorFileNameExt = ""):
    fileName = fileNameGenerator(userID, sensorID, sensorFileNameExt)
    return readSignal(fileName, signalName)

def writeToLatexTable(rows):
    print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))
    table = Texttable()
    table.set_cols_align(["c"] * 4)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    print('\nTexttable Table:')
    print(table.draw())
      
#%% Load MMAES scores
if isOnlyLowHighScoreUsers:
    lowFactorUserIDs, highFactorUserIDs = getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, selectedContent, sensors, sensorID, userSensorContentFileName, factorScoresFileName)  
    print(lowFactorUserIDs)
    print(highFactorUserIDs)
    
    # Build dictionary
    users = {}
    for cuID in lowFactorUserIDs['userID']:
        users[cuID] = {}
        users[cuID][signalName] = {}
        users[cuID][selectedFactor] = lowFactorUserIDs.loc[lowFactorUserIDs['userID']==cuID][selectedFactor]
        
    for cuID in highFactorUserIDs['userID']:
        users[cuID] = {}
        users[cuID][signalName] = {}
        users[cuID][selectedFactor] = highFactorUserIDs.loc[highFactorUserIDs['userID']==cuID][selectedFactor]


else:
    allusers = getAllFilteredUserScores(outputFolder, selectedFactor, selectedContent, sensors, sensorID, userSensorContentFileName, factorScoresFileName)

    users = {}
    for cuID in allusers['userID']:
        users[cuID] = {}
        users[cuID][signalName] = {}
        users[cuID][selectedFactor] = allusers.loc[allusers['userID']==cuID][selectedFactor]
        
#%% Load data
# uIDs = lowFactorUserIDs['userID'].tolist() + highFactorUserIDs['userID'].tolist()

if isOnlyLowHighScoreUsers:
    uIDs = lowFactorUserIDs['userID'].tolist() + highFactorUserIDs['userID'].tolist()
else:
    uIDs = allusers['userID']

for cuID in uIDs:
    signal_x_df = getSignalDf(cuID, sensorID, signalName, list(sensors[sensorID][1].keys())[0])
    sig_x = np.array(signal_x_df[signalName])
    sig_t = np.array(signal_x_df['timestamp_s'])
    # Preprocessing
    if lowpassQ:
        sig_p_x = sat.lowpass_1D(sig_x, cut_f)
    else:
        sig_p_x = sig_x
    
    if plotRawQ:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(sig_t, sig_x, label='Original')
        axs[1].plot(sig_t, sig_p_x, label='Preprocessed', color='r')
        fig.legend()
        fig.tight_layout()
        # plt.show()
        fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/uID_" + str(cuID) + "_" + signalName + "_preprocessed"
        # plt.savefig(fName + ".jpg")
        # pickle.dump(fig, open(fName +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
        plt.close(fig)
        
    # Feature extraction 
    time_int = sat.getAdTimeInterval("Data/" + selectedContent + "_usersAdStartEndTimes.csv", cuID)
    users[cuID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)

print(users)

#for one user
# uID =32
# signal_x_df = getSignalDf(uID, sensorID, signalName, list(sensors[sensorID][1].keys())[0])

# Load signals
# sig_x = np.array(signal_x_df[signalName])
# sig_t = np.array(signal_x_df['timestamp_s'])

# #%% Preprocessing
# if lowpassQ:
#     sig_p_x = sat.lowpass_1D(sig_x, cut_f)
# else:
#     sig_p_x = sig_x
    

# # Plot raw data
# if plotRawQ:
#     fig, axs = plt.subplots(2, 1)
#     axs[0].plot(sig_t, sig_x, label='Original')
#     axs[1].plot(sig_t, sig_p_x, label='Preprocessed', color='r')
#     fig.legend()
#     fig.tight_layout()
#     plt.show()
    
# #%% Feature extraction 

# time_int = sat.getAdTimeInterval("Data/" + selectedContent + "_usersAdStartEndTimes.csv", uID)
# users[uID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, code=feature_code)

# print(users[uID][signalName][feature_code])
#%% Load pickle file


Utils.loadFigFromPickleFile('C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/SignalCorrelation/C1/empatica/uID_5_EDA_preprocessed.pickle')

#%% Visual inspection 

if pVals:
    x_data, y_data = [k for k,v in users.items()], [v[signalName][feature_code][0] for k,v in users.items()]
    r, p = sat.correlate_sigs_MME_OnlyData(x_data, y_data, coeff_type)
    print(r)
    print(p)
    data = {'Sensor': [sensors[sensorID][0]],
            'Signal': [signalName],
            'Coeffcient Type':[coeff_type],
            'r Val':[r],
            'p Val':[p]}
    df = pd.DataFrame(data)
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + feature_code + ' ' + sensors[sensorID][0] + ' ' + signalName
    df.to_latex(fName + '_pval.tex', index=False)
    # # m = len(uIDs)

    # # Collect data
    
    
    # x_data = np.array(x_data)
    # y_data = np.array(y_data)
    # # for uID in uIDs: 
    # #     x_data, y_data = users[signalName][feature_code],  users[uID][selectedFactor]
        
    # # if plotQ:
    # #     plt.scatter(x_data, y_data)


    # # Compute correlation
    # if coeff_type == 'Pearson':
    #     r, p = sst.pearsonr(x_data, y_data)
    # if coeff_type == 'KendalTau':
    #     r, p = sst.kendalltau(x_data, y_data)
    # r, p = sat.correlate_sigs_MME(uIDs, users, signalName, feature_code, selectedFactor, coeff_type, False)

if plotScatteQ: # 2D scatter plot
    # sat.scatter_sigs_MME(uIDs, users, signalName, feature_code)
    for uID in uIDs:
        if feature_code == 'std':
            plt.scatter(users[uID][signalName][feature_code],users[uID][selectedFactor])
        elif feature_code == 'slope':
            plt.scatter(users[uID][signalName][feature_code][0],users[uID][selectedFactor])

    plt.title(selectedFactor + ' ' + feature_code + ' vs '  + sensors[sensorID][0] + ' ' + signalName )
    plt.xlabel(feature_code)
    plt.ylabel(signalName)
    plt.legend()
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + feature_code + ' ' + sensors[sensorID][0] + ' ' + signalName
    plt.savefig(fName + ".jpg")
    pickle.dump(fig, open(fName +'.pickle', 'wb'))
    plt.show()

if plot3D:
    feature_codes = ['std', 'slope', 'spec_low']
    sat.plot_features_MME_3D(uIDs, users, signalName, feature_codes)

    