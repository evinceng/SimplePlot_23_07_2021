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
import Tools.signal_visualization_tool as svt
import matplotlib.pyplot as plt
import pickle
import Utils
import numpy as np
import scipy.stats as sst
import seaborn as sns

outputFolder = "SignalCorrelation/"

factorScoresFileName = "Data/MMEM_Scores_FNames/mmem_C1_argmin_F_df.csv"
selectedFactor = 'AE'
selectedContent = 'C1'

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user"
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


sensorID = 1

# #pupillabs
# signalName = 'diameter'
# sensors = {1:['pupillabs', {'left_eye_2d':['diameter']}]}
# sensorsFeaturePars = {1:['pupillabs', {'diameter':[0, 1.6, 4]}]}
# cut_f = 0.27 # Hertz

# tobii
# signalName = 'diameter'
# sensors = {1:['tobii', {'pupilDim_left_eye':['diameter']}]}
# sensorsFeaturePars = {1:['tobii', {'diameter':[0, 1.6, 4]}]}
# cut_f = 0.27 # 4Hertz = 4/15

# signalName = 'EDA'
# sensors = {1:['empatica', {'EDA':['EDA']}]}
# sensorsFeaturePars = {1:['empatica', {'EDA':[0, 0.015]}]} # 0.02
# cut_f = 0.12 # Hertz

# signalName = 'HR'
# sensors = {1:['empatica', {'HR':['HR']}]}
# sensorsFeaturePars = {1:['empatica', {'HR':[0, 0.06, 0.12, 0.20]}]} #[0, 0.06, 0.12, 0.20] #0, 0.015, 0.030, 0.045
# cut_f = 1.8 # Hertz

signalName = 'AccX'
sensors = {1:['empatica', {'ACC':['AccX']}]}
sensorsFeaturePars = {1:['empatica', {'AccX':[0, 0.075, 0.147, 0.223, 0.280]}]}
cut_f = 0.3 # Hertz

# signalName = 'GSR_Skin_Conductance_microSiemens'
# sensors = {1:['shimmer', {'':['GSR_Skin_Conductance_microSiemens']}]}
# sensorsFeaturePars = {1:['shimmer', {'GSR_Skin_Conductance_microSiemens':[0, 0.015, 0.045, 0.075]}]} #[0, 0.015, 0.045, 0.060, 0.075]
# cut_f = 0.12 # Hertz

# signalName = 'Temperature_BMP280_Degrees Celsius'
# sensors = {1:['shimmer', {'':['Temperature_BMP280_Degrees Celsius']}]}
# sensorsFeaturePars = {1:['shimmer', {'Temperature_BMP280_Degrees Celsius':[0, 0.075, 0.147, 0.223, 0.280]}]}
# cut_f = 0.02 # Hertz


# signalName = 'BVP'
# sensors = {1:['empatica', {'BVP':['BVP']}]}
# sensorsFeaturePars = {1:['empatica', {'BVP':[0, 0.8, 1.8, 2.6, 3.6, 7]}]} #{'BVP':[0, 1, 2, 3.5, 7]}
# cut_f = 5 # Hertz
#fetures
# Number of peaks – you have to lowpass filter spectrum and count
# f1/(f2+f3+f4)
# f2/(f2+f3+f4)




# Settings
lowpassQ = True
scatterQ = True
pVals = True
pValuesFileName = outputFolder + '/AllPValues.csv'  #+ selectedContent 

plotRawQ = False
plotScatteQ = True
plotNormParsQ = True
plot3D = False

# Configuration

preproc_meth = 'lowpass'

feature_codes = ['std', 'slope', 'spec_comp', 'spec_amp', 'spec_phs', 'periodogram', 'num_of_peaks', 'monotone_ints', 'total_var', 'exp_fit', # till index 9
                 'Gram_AF', 'Recurr_M', 'Markov_TF', 'Dyn_Time_W']

# # slope
# feature_code = feature_codes[1]
# feature_pars = sensorsFeaturePars[sensorID][1][signalName]
# feature_function = ''


# num_of_peaks
feature_code = feature_codes[6]
feature_pars = [cut_f]#sensorsFeaturePars[sensorID][1][signalName]
feature_function = ''

# # Gram_AF
# feature_code = feature_codes[10]
# feature_pars = sensorsFeaturePars[sensorID][1][signalName]
# feature_function = ''

# # total_var
# feature_code = feature_codes[7]
# feature_pars = []
# feature_function = ''

# # monotone_ints
# feature_code = feature_codes[6]
# feature_pars = [5]
# feature_function = 'len'

# # num_of_peaks
# feature_code = feature_codes[5]
# feature_pars = [5]
# feature_function = ''

# #'periodogram'
# feature_code = feature_codes[4]
# feature_pars = []
# feature_function = ''

# # spec_phs
# feature_code = feature_codes[3]
# feature_pars = [1,2]
# feature_function = 'getF2oF3'

# # spec_comp
# feature_code = feature_codes[2]
# feature_pars = sensorsFeaturePars[sensorID][1][signalName]
# feature_function = ''

# #spec_amp
# feature_code = feature_codes[3]
# feature_pars = sensorsFeaturePars[sensorID][1][signalName]
# feature_function = ''

isOnlyLowHighScoreUsers = True #True 
isOneUser = False #True

doTransformation = True #True
doNormalizationQ = True
cutSignals = True

predef_pupildiameter_mean = 3.0

#%% Functions

def getAllUsersScores(userIDList, factorScoresFileName, selectedFactor):
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['uID'].isin(userIDList)] 
    return scores_df[['uID', selectedFactor]]

def getAllFilteredUserScores(outputFolder, selectedFactor, contentID, sensors, sensorID, userSensorContentFileName, factorScoresFileName):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0] + '/' + feature_code).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    allUsers = getAllUsersScores(filteredUserList, factorScoresFileName, selectedFactor)
    return allUsers
    
def getLowAndHighFactorUserIDS(userIDList, contentID, factorScoresFileName, selectedFactor, numberOfUsers=4):
    Path(outputFolder + contentID + '/' + sensors[sensorID][0] + '/' + feature_code).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['uID'].isin(userIDList)] 
    sorted_df = scores_df.sort_values(selectedFactor)
    
    return sorted_df[['uID', selectedFactor]][:numberOfUsers], sorted_df[['uID', selectedFactor]][-numberOfUsers:]
    

# print(fileNameGenerator(1, 2))

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df['uID']


def getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, contentID, sensors, sensorID, userSensorContentFileName, factorScoresFileName, numberOfUsers=6):
   
    Path(outputFolder + contentID + '/' + sensors[sensorID][0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
    print(filteredUserList)
    lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, contentID, factorScoresFileName, selectedFactor, numberOfUsers)
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

def writeFeatureToDf(users, outputFolder, selectedContent, sensorID, signalName, feature_code, isOnlyLowHighScoreUsers, cutSignals):
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/FeatureVals_"  + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code 
    
    if isOnlyLowHighScoreUsers:
        fName = fName + ' low-high_users'
    else:
        fName = fName + ' all_users'
    
    if cutSignals:
        fName = fName + '_cutSignal'
    else:
        fName = fName + '_allSignal'
        
    fName = fName+ '_df.csv' 
    
    # user_df = pd.DataFrame(users.items(), columns=['uID', feature_code, selectedFactor])
    
    userList, featureValList, selectedFactorScores = [k for k,v in users.items()], [v[sensorID][signalName][feature_code][0] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()]  # userID 
    
    user_arr = np.transpose(np.array([userList, featureValList, selectedFactorScores]))
    user_df = pd.DataFrame(data= user_arr, columns=['uID', feature_code, selectedFactor])
    
    user_df.to_csv(fName, index=False)
    
def readFeatureDf_FromFileName(fileName):
    return pd.read_csv(fileName)
    
def readFeatureDf(outputFolder, selectedContent, sensorID, signalName, feature_code, isOnlyLowHighScoreUsers, cutSignals):
    fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/FeatureVals_"  + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code 
    
    if isOnlyLowHighScoreUsers:
        fName = fName + ' low-high_users'
    else:
        fName = fName + ' all_users'
    
    if cutSignals:
        fName = fName + '_cutSignal'
    else:
        fName = fName + '_allSignal'
        
    fName = fName+ '_df.csv'
    
    return pd.read_csv(fName)
      
#%% Load MMAES scores
if isOnlyLowHighScoreUsers:
    lowFactorUserIDs, highFactorUserIDs = getFilteredListOfLowHighScoreUserIDS(outputFolder, selectedFactor, selectedContent, sensors, sensorID, userSensorContentFileName, factorScoresFileName)  
    low_high_Fname= outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + sensors[sensorID][0] + '_'
    lowFactorUserIDs.to_csv(low_high_Fname+"lowFactorUserIDs.csv", index= False)
    highFactorUserIDs.to_csv(low_high_Fname+"highFactorUserIDs.csv", index= False)
    print(lowFactorUserIDs)
    print(highFactorUserIDs)
    
    # Build dictionary
    users = {}
    for cuID in lowFactorUserIDs['uID']:
        users[cuID] = {}   
        users[cuID][sensorID] = {}
        users[cuID][sensorID][signalName] = {}
        users[cuID][selectedFactor] = lowFactorUserIDs.loc[lowFactorUserIDs['uID']==cuID][selectedFactor].iloc[0]
        
    for cuID in highFactorUserIDs['uID']:
        users[cuID] = {}   
        users[cuID][sensorID] = {}
        users[cuID][sensorID][signalName] = {}
        users[cuID][selectedFactor] = highFactorUserIDs.loc[highFactorUserIDs['uID']==cuID][selectedFactor].iloc[0]


else:
    allusers = getAllFilteredUserScores(outputFolder, selectedFactor, selectedContent, sensors, sensorID, userSensorContentFileName, factorScoresFileName)

    users = {}
    for cuID in allusers['uID']:
        users[cuID] = {}   
        users[cuID][sensorID] = {}
        users[cuID][sensorID][signalName] = {}
        users[cuID][selectedFactor] = allusers.loc[allusers['uID']==cuID][selectedFactor].iloc[0]
        
#%% Load data
# uIDs = lowFactorUserIDs['uID'].tolist() + highFactorUserIDs['uID'].tolist()

# import Tools.signal_analysis_tools as sat



lowFactorUsersMeanStd = []
highFactorUsersMeanStd = []
allUsersMeanStd = []

if isOnlyLowHighScoreUsers:
    uIDs = lowFactorUserIDs['uID'].tolist() + highFactorUserIDs['uID'].tolist()
else:
    uIDs = allusers['uID']

#store normalization parameters ['mean']

for cuID in uIDs:
    signal_x_df = getSignalDf(cuID, sensorID, signalName, list(sensors[sensorID][1].keys())[0])
    sig_x = np.array(signal_x_df[signalName])
    sig_t = np.array(signal_x_df['timestamp_s'])
    # Preprocessing
    if lowpassQ:
        sig_lp_x = sat.lowpass_1D(sig_x, cut_f)
    else:
        sig_lp_x = sig_x
    
    
    # normalization: take the mean of signals while in room, and substract mean from each user/signal 
    if doNormalizationQ:
        #covert pixel to mm for pupillabs
        # if sensors[cuID] == 'pupillabs':
        #     sig_lp_x = sig_lp_x * (3.5/32) # 3.5 non pupil labs mean, 32 is pupillabs mean
        time_int_inside_room = sat.getTimeIntervalofR1andR4(usersDictFileName, cuID)
        print(time_int_inside_room)
        inside_room_mean = sat.get_timesingal_feature(sig_t, sig_x, time_int = time_int_inside_room, feature_pars = [], code='mean')
        inside_room_std = sat.get_timesingal_feature(sig_t, sig_x, time_int = time_int_inside_room, feature_pars = [], code='std')
        inside_room_kurtosis = sat.get_timesingal_feature(sig_t, sig_x, time_int = time_int_inside_room, feature_pars = [], code='kurtosis')
        #add raw features to users dict
        users[cuID][sensorID][signalName]['mean'] = inside_room_mean
        users[cuID][sensorID][signalName]['std'] = inside_room_std               
        users[cuID][sensorID][signalName]['kurtosis'] = inside_room_kurtosis
        # add lists kurtosis...
        low_factor_userID_list = lowFactorUserIDs['uID'].tolist()
        if isOnlyLowHighScoreUsers:
            if cuID in low_factor_userID_list:
                lowFactorUsersMeanStd.append([cuID, inside_room_mean[0], inside_room_std[0], inside_room_kurtosis[0], users[cuID][selectedFactor]])
            else:
                highFactorUsersMeanStd.append([cuID, inside_room_mean[0], inside_room_std[0], inside_room_kurtosis[0],users[cuID][selectedFactor]])
        else:
            allUsersMeanStd.append([cuID, inside_room_mean[0], inside_room_std[0], inside_room_kurtosis[0],users[cuID][selectedFactor]])
        #subtract mean from the signal value
        # do the multiplication for pupillabs from pixels to mm
        #get hardcoded 
        
       
        # Note: we killed the volume feature
        
        sig_lp_x -= (inside_room_mean[0] - predef_pupildiameter_mean)
    
        
    time_int = []
    if doTransformation:
        #get min and max from the all signal
        print(cuID)
        min_tr, max_tr = sat.get_scaling_pars(sig_t, sig_lp_x, time_int, feature_pars, code='standard', scaling_par=[0.03])
        print('min is ' + str(min_tr))
        print('max is ' + str(max_tr))
        sig_p_x = sat.do_linear_transform(sig_t, sig_lp_x, time_int, min_tr, max_tr)
    
    #todo: Evin add ad video lines
    if plotRawQ:
        fig, axs = plt.subplots(2, 1, sharex = True)
        axs[0].plot(sig_t, sig_x, label='Original')
        axs[1].plot(sig_t, sig_p_x, label='Preprocessed', color='r')
        fig.legend()
        fig.tight_layout()
        plt.title("uID_" + str(cuID) +  "_"  + sensors[sensorID][0] +  "_" + signalName)
        # plt.show()
        Path(outputFolder + selectedContent + '/' + sensors[sensorID][0] + '/preprocessed').mkdir(parents=True, exist_ok=True)
        fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/preprocessed/uID_" + str(cuID) + "_" + signalName + "_preprocessed"
        plt.savefig(fName + ".jpg")
        pickle.dump(fig, open(fName +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
        plt.close(fig)
        
        
    # Feature extraction
    
    if cutSignals:
        time_int = sat.getAdTimeInterval("Data/" + selectedContent + "_usersAdStartEndTimes.csv", cuID)
        
    if feature_code == 'Gram_AF' or feature_code == 'Recurr_M' or feature_code == 'Markov_TF'or feature_code == 'Dyn_Time_W':
        print('2d feature')
        users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_2Dfeature(sig_t, sig_p_x, time_int, feature_pars, feature_code)
    
    elif feature_function == 'getF1oF23':
        print('getF1oF23')
        users[cuID][sensorID][signalName][feature_code] = sat.getF1oF23(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
    elif feature_function == 'getF2oF3':
        print('getF2oF3')
        users[cuID][sensorID][signalName][feature_code] = sat.getF2oF3(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
    elif feature_function == 'getF1oF2':
       print('getF1oF2')
       users[cuID][sensorID][signalName][feature_code] = sat.getF1oF2(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))

    elif feature_function == 'feature1':
       print('feature1')
       users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)[0]

    elif feature_function == 'len':
       print('len')
       users[cuID][sensorID][signalName][feature_code] = len(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code))
    
    elif feature_function == 'getFxoFy':
       print('getFxoFy')
       x = [2]
       y = [1]
       users[cuID][sensorID][signalName][feature_code] = sat.getFxoFy(sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code), x , y)
    else:
        users[cuID][sensorID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, feature_pars, feature_code)
    
    if isOneUser:
        break

writeFeatureToDf(users, outputFolder, selectedContent, sensorID, signalName, feature_code, isOnlyLowHighScoreUsers, cutSignals)
# df  = readFeatureDf(outputFolder, selectedContent, sensorID, signalName, feature_code, isOnlyLowHighScoreUsers, cutSignals)

if doNormalizationQ:
    writeFeatureToDf(users, outputFolder, selectedContent, sensorID, signalName, 'mean', isOnlyLowHighScoreUsers, cutSignals)
    writeFeatureToDf(users, outputFolder, selectedContent, sensorID, signalName, 'std', isOnlyLowHighScoreUsers, cutSignals)
    writeFeatureToDf(users, outputFolder, selectedContent, sensorID, signalName, 'kurtosis', isOnlyLowHighScoreUsers, cutSignals)

#write mean and std to files
meanColNames = ['uID','inside_room_Mean', 'inside_room_Std', 'inside_room_Kurtosis',selectedFactor]
meanFileNameBase = outputFolder + selectedContent + '/' + sensors[sensorID][0] +'/Mean_Std_Kurtosis_inside_room_' + sensors[sensorID][0]+ '_' + signalName + '_' + selectedFactor 
if isOnlyLowHighScoreUsers:
    low_mean_df = pd.DataFrame(data=lowFactorUsersMeanStd, columns= meanColNames)
    low_mean_df.to_csv(meanFileNameBase + '_low.csv', index=False)

    high_mean_df = pd.DataFrame(data=highFactorUsersMeanStd, columns=meanColNames)
    high_mean_df.to_csv(meanFileNameBase +'_high.csv', index=False)
else:
    mean_df = pd.DataFrame(data=allUsersMeanStd, columns=meanColNames)
    mean_df.to_csv(meanFileNameBase +'_all.csv', index=False)
        
#plot a grapgh horizontal:uID, vertical: normalization parameter (mean) values(a dot grpah with points: colored by the exposure) 
if plotNormParsQ:
                   
    if isOnlyLowHighScoreUsers:
        low_high_df = pd.concat([low_mean_df,high_mean_df])
        ax = sns.scatterplot(data=low_high_df, x='uID', y='inside_room_Mean', hue=selectedFactor)
        
        low_mean, low_std = sat.getAvarageOfMeanStds(low_mean_df, 'inside_room_Mean', 'inside_room_Std')
        plt.axhline(y=low_mean, color='r', linestyle='-', label='low_mean')
        plt.axhline(y=low_mean-low_std, color='m', linestyle='--', label='low_mean-low_std')
        plt.axhline(y=low_mean+low_std, color='m', linestyle='--', label='low_mean+low_std')
   
        high_mean, high_std = sat.getAvarageOfMeanStds(high_mean_df, 'inside_room_Mean', 'inside_room_Std')
        plt.axhline(y=high_mean, color='g', linestyle='-', label='high_mean')
        plt.axhline(y=high_mean-high_std, color='b', linestyle='--', label='high_mean-high_std')
        plt.axhline(y=high_mean+high_std, color='b', linestyle='--', label='high_mean+high_std')
    else:
        ax = sns.scatterplot(data=mean_df, x='uID', y='inside_room_Mean', hue=selectedFactor)
        mean, std = sat.getAvarageOfMeanStds(mean_df, 'inside_room_Mean', 'inside_room_Std')
        plt.axhline(y=mean, color='g', linestyle='-', label='mean')
        plt.axhline(y=mean-std, color='b', linestyle='--', label='mean-std')
        plt.axhline(y=mean+std, color='b', linestyle='--', label='mean+std')
        
    fig = ax.get_figure()
    plt.legend()
    plt.title('uID' + ' vs '  + signalName + ' inside_room_Mean')
    # plt.xlabel(sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code + ' ' + )
    plt.ylabel(sensors[sensorID][0] + ' ' + signalName + 'inside_room_Mean')
    normFname =  outputFolder + selectedContent + '/' + sensors[sensorID][0] +'/Normalization_' + sensors[sensorID][0]+ '_' + signalName + '_' + selectedFactor 
    plt.savefig(normFname + ".jpg")
    pickle.dump(fig, open(normFname +'.pickle', 'wb'))
    plt.show()

if isOnlyLowHighScoreUsers and feature_code == 'spec_comp':
    svt.generateSubPlotsofOneSignalOfMultipleUsers(users, sensors, selectedContent, selectedFactor, outputFolder,
                                                    lowFactorUserIDs['uID'],
                                                    highFactorUserIDs['uID'],
                                                    sensorID,
                                                    signalName,
                                                    feature_code)
    
    
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
# users[uID][sensorID][signalName][feature_code] = sat.get_timesingal_feature(sig_t, sig_p_x, time_int, code=feature_code)

# print(users[uID][sensorID][signalName][feature_code])

#%% P value and Visual inspection 

coeff_types = ['Pearson', 'KendalTau']
# coeff_type = coeff_types[0]

if doNormalizationQ:
    feature_types = [feature_code, 'mean', 'std', 'kurtosis']
else:
    feature_types = [feature_code]

for feat in feature_types:
    for coeff_type in coeff_types:
    
        fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + feat + '/' + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code + ' ' + coeff_type  
        Path(outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + feat).mkdir(parents=True, exist_ok=True)
        # fName = outputFolder + selectedContent + '/' + sensors[sensorID][0] + "/" + feat + ' ' + coeff_type + ' '  + sensors[sensorID][0] + ' ' + signalName
            
        if isOnlyLowHighScoreUsers:
            fName = fName + ' low-high_users'
        else:
            fName = fName + ' all_users'
        
        if cutSignals:
            fName = fName + '_cutSignal'
        else:
            fName = fName + '_allSignal'
            
        if pVals:
            if feat == 'slope' or feat == 'total_var' or feat == 'num_of_peaks' or feat == 'mean' or feat == 'std' or feat == 'kurtosis': #or feature_code == 'spec_amp'   
                x_data, y_data = [v[sensorID][signalName][feat][0] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()]  # userID [k for k,v in users.items()], 
            else:
                x_data, y_data = [v[sensorID][signalName][feat] for k,v in users.items()], [v[selectedFactor] for k,v in users.items()] # userID [k for k,v in users.items()], 
            
            r, p, es = sat.correlate_sigs_MME_OnlyData(x_data, y_data, coeff_type) #np.array(y_data, dtype='float'
            
            #generic effect size works with 2 classes
            if isOnlyLowHighScoreUsers:  
                low_factor_userID_list = lowFactorUserIDs['uID'].tolist()
                if feat == 'slope' or feat == 'total_var' or feat == 'num_of_peaks' or feat == 'mean' or feat == 'std' or feat == 'kurtosis':
                    gen_x_data, gen_y_data = [v[sensorID][signalName][feat][0] for k,v in users.items()], [0 if k in low_factor_userID_list else 1  for k,v in users.items()]
                else:
                    gen_x_data, gen_y_data = [v[sensorID][signalName][feat] for k,v in users.items()], [0 if k in low_factor_userID_list else 1  for k,v in users.items()]
                
                genericEffectSize = sat.get_generic_es(np.array(gen_x_data), np.array(gen_y_data))
            else: 
                genericEffectSize = 0
            print(signalName + ' ' + feat)
            print("r = " + str(r))
            print("p = "+ str(p))
            print("effectsize = "+ str(es))
            print("Generic effectsize = "+ str(genericEffectSize))
            data = { 'Content': [selectedContent],
                    'MMAES Subscale': [selectedFactor],
                    'Sensor': [sensors[sensorID][0]],
                    'Signal': [signalName],
                    'Feature Code': [feat],
                    'Coeffcient Type':[coeff_type],
                    'r Val':[r],
                    'p Val':[p],
                    'Effect Size':[es],
                    'Generic Effect Size':[genericEffectSize],
                    'OnlyLowHighUsers':[isOnlyLowHighScoreUsers],
                    'CutSignal':[cutSignals]}
            df = pd.DataFrame(data)
            df.to_latex(fName + '_pval.tex', index=False)
            
            if not os.path.isfile(pValuesFileName):
                df.to_csv(pValuesFileName, index=False)
            else:
                #skip duplicates if possible
                df.to_csv(pValuesFileName, mode='a', header=False, index=False)
    

# plot the mean and std dev of features, vertical lines
if plotScatteQ: # 2D scatter plot
    # sat.scatter_sigs_MME(uIDs, users, signalName, feature_code)
    fig, ax = plt.subplots()
    for uID in uIDs:
        if feature_code == 'slope'  or feature_code == 'periodogram': #or feature_code == 'spec_amp'
            ax.scatter(users[uID][sensorID][signalName][feature_code][0],users[uID][selectedFactor])
        else:
            ax.scatter(users[uID][sensorID][signalName][feature_code],users[uID][selectedFactor])
        if isOneUser:
            break      
    plt.title(selectedFactor + ' vs '  + sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code)
    plt.xlabel(sensors[sensorID][0] + ' ' + signalName + ' ' + feature_code)
    plt.ylabel(selectedFactor + '_Score')
    plt.legend()    
    plt.savefig(fName + ".jpg")
    pickle.dump(fig, open(fName +'.pickle', 'wb'))
    plt.show()
    

if plot3D:
    feature_codes = ['std', 'slope', 'spec_low']
    sat.plot_features_MME_3D(uIDs, users, signalName, feature_codes)

#%% Load pickle file


# Utils.loadFigFromPickleFile('C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/SignalCorrelation/C1/empatica/uID_5_EDA_preprocessed.pickle')
# Utils.loadFigFromPickleFile('C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/SignalCorrelation/C1/empatica/AE_empatica_EDA_spec_comp sig_f vs sig_comp.pickle')
