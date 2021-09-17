# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:16:04 2021

@author: evinao
"""
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from datetime import datetime
import os.path
from pathlib import Path

factorScoresFileName = "Data/MMEM_Scores/mmem_C1_argmin_F_df.csv"
selectedFactor = 'F1'
selectedContent = 'C1'

userSensorContentFileName =  "Data/userContentSensorDict.csv"

# lowFactorUserIDs = [14,16,17,10] #tobii [14,16,17,10] # empatica, shimmer pupillabs [1,2,3,4]
# highFactorUserIDs = [50,51,52,53] # empattica, shimmer [5,6,7,8] # pupillabs [33,34,35,36] #tobii

sensorID = 1
sensorFileNameExt = 'ACC'# 'pupilCenter_left_eye'#'left_eye_2d' #EDA
signalName = 'AccX' #'pc_x' #'diameter' # 'EDA'

outputFolderName = "PictorialProof/"


rootFolder = "C:/Users/evinao/Dropbox (Lucami)/Lucami Team Folder/MMEM_Data/output/user"

usersDictFileName = "Data/usersDict.xlsx"

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

sensors = {1:['empatica', empaticaSignals] , 2:['shimmer', shimmerSignals],
           3:['tobii',tobiiSignals],  4:['pupillabs', pupillabsSignals]}



# sensors = {1:'empatica', 2:'shimmer', 3:'tobii', 4:'pupilLabs'}

# signals = ['AccX', 'AccY', 'AccZ', 'EDA', 'BVP', 'HR',
#            'Accel_LN_X_m/(s^2)', 'Accel_LN_Y_m/(s^2)',
#            'Accel_LN_Z_m/(s^2)', 'GSR_Range_no_units.1',
#            'GSR_Skin_Conductance_microSiemens', 'GSR_Skin_Conductance_uS.1',
#            'Pressure_BMP280_kPa', 'PPG_A12_mV',
#            'Temperature_BMP280_Degrees Celsius', 'GSR_Skin_Resistance_kOhms',
#            'Gyro_Z_deg/s', 'Gyro_X_deg/s', 'Gyro_Y_deg/s',
#            'Accel_WR_Z_m/(s^2)', 'Accel_WR_Y_m/(s^2)', 'Accel_WR_X_m/(s^2)',
#            'norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle',
#            'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z',
#            'norm_pos_x', 'norm_pos_y', 'gaze_point_3d_x', 'gaze_point_3d_y',
#            'gaze_point_3d_z', 'gaze_normal0_x', 'gaze_normal0_y',
#            'gaze_normal0_z', 'gaze_normal1_x', 'gaze_normal1_y',
#            'gaze_normal1_z', 'ac_x', 'ac_y', 'ac_z', 'gd_x', 'gd_y', 'gd_z',
#            'gp_x', 'gp_y', 'gp_latency', 'gp3d_x', 'gp3d_y', 'gp3d_z',
#            'gy_x', 'gy_y', 'gy_z', 'pc_x', 'pc_y', 'pc_z', 'diameter']

# mmaesFactors = {1:'AE', 2:'RE', 3:'AA', 4:'PI'}

# print(mmaesFactors[1])

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


def generateSignalSubplots(ax, axIndex, lowOrHighFactorUserIDs, sensorID, signalName, sensorFileNameExt):
    subPlotId = 0    
    for userID in lowOrHighFactorUserIDs:
        # if (key % 2) == 0:
        #     color = '#1f77b4'
        # else:
        #     color =  'm'
          
        df = getSignalDf(userID, sensorID, signalName, sensorFileNameExt)
        print(df.head())
        ax[axIndex][subPlotId].plot(df['timestamp_s'], df[signalName]) #, c=color
        ax[axIndex][subPlotId].set_xlabel('t [s]')
        ax[axIndex][subPlotId].set_ylabel('user ' + str(userID))
        ax[axIndex][subPlotId].grid(True)
        subPlotId = subPlotId + 1

def generateAdVidLines(ax, axIndex, usersDictFileName,lowOrHighFactorUserIDs,selectedContent):
    
    out_times_lst = Utils.get_video_and_ad_times_userslist_one_content(usersDictFileName,
                                                                       lowOrHighFactorUserIDs,
                                                                       selectedContent)
    subPlotId = 0  
    #draw vertical lines for one of the video-ad 
    for ad in out_times_lst:
        ax[axIndex][subPlotId].axvline(x=ad[0], c='g')
        ax[axIndex][subPlotId].axvline(x=ad[1], c='r')
        ax[axIndex][subPlotId].axvline(x=ad[2], c='r')
        ax[axIndex][subPlotId].axvline(x=ad[3], c='g')
        subPlotId = subPlotId + 1

def generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(contentID, 
                                                               lowFactorUserIDs,
                                                               highFactorUserIDs,
                                                               sensorID,
                                                               signalName,
                                                               sensorFileNameExt = ""):
    
    subPlotCount = max(len(lowFactorUserIDs), len(highFactorUserIDs))
    #take min and max 
    fig, ax = plt.subplots(2, subPlotCount) #, sharey=True
    fig.suptitle(sensors[sensorID][0] +' ' + sensorFileNameExt +' ' + signalName + ' ' + selectedContent)
    
    ax[0][0].set_title('Low score ' + selectedFactor)
    ax[1][0].set_title('High score ' + selectedFactor)
    
    # plot the first row of subplots
    generateSignalSubplots(ax, 0, lowFactorUserIDs, sensorID, signalName, sensorFileNameExt)    
    # plot the second row of subplots
    generateSignalSubplots(ax, 1, highFactorUserIDs, sensorID, signalName, sensorFileNameExt)
    
    # draw the first row of video ad lines
    generateAdVidLines(ax, 0, usersDictFileName, lowFactorUserIDs, selectedContent)    
    # draw the second row of video ad lines
    generateAdVidLines(ax, 1, usersDictFileName, highFactorUserIDs, selectedContent) 
    
    now = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    if '/' in signalName:
        signalName = signalName.replace('/','-')
    elif '.' in signalName:
        signalName = signalName.replace('.','-')
    saveFileFig = outputFolderName + contentID + '/' + sensors[sensorID][0] + '/' + selectedFactor + '_' + sensors[sensorID][0] + '_' + sensorFileNameExt + '_' + signalName + '_' + now
    plt.savefig(saveFileFig +'.jpg')
    Utils.writePickleFile(fig, saveFileFig)
    # pickle.dump(fig, open(saveFileFig +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
    plt.close(fig)
    # plt.show()
    
    


def getLowAndHighFactorUserIDS(userIDList, factorScoresFileName, selectedFactor, numberOfUsers=4):
    scores_df = pd.read_csv(factorScoresFileName)
    scores_df = scores_df.loc[scores_df['userID'].isin(userIDList)] 
    sorted_df = scores_df.sort_values(selectedFactor)
    
    # print(sorted_df['userID'][:numberOfUsers])
    # print(sorted_df['userID'][-numberOfUsers:])
    return sorted_df['userID'][:numberOfUsers], sorted_df['userID'][-numberOfUsers:]
    

# print(fileNameGenerator(1, 2))

def getUsersSignalsOfOneContent(fileName, sensors, sensorID, contentID):
    usersContent_df = pd.read_csv(fileName, encoding = "utf-8")
    sensorContentStr = sensors[sensorID][0] + '_' + str(contentID)
    usersContent_df = usersContent_df.loc[usersContent_df[sensorContentStr] == 1]
    # print(usersContent_df.head())
    return usersContent_df['userID']


# create the figures for one sensor-signal 

# filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, 1, 1)
# print(filteredUserList)
# lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, factorScoresFileName, selectedFactor)
# generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(lowFactorUserIDs,
#                                                             highFactorUserIDs,
#                                                             sensorID,
#                                                             signalName,
#                                                             sensorFileNameExt)

pictOutputFolder = "PictorialProof/"

def createFiguresForAll(pictOutputFolder, selectedFactor, contentID, sensors, userSensorContentFileName, factorScoresFileName, numberOfUsers=6):
    for sensorID, sensor in sensors.items():
        Path(pictOutputFolder + contentID + '/' + sensor[0]).mkdir(parents=True, exist_ok=True) # will not change directory if exists
        filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, contentID)
        print(filteredUserList)
        lowFactorUserIDs, highFactorUserIDs = getLowAndHighFactorUserIDS(filteredUserList, factorScoresFileName, selectedFactor, numberOfUsers)
        
        for sensorFileNameExt, signalList in sensor[1].items():
            for signalName in signalList:
                generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(contentID, lowFactorUserIDs,
                                                            highFactorUserIDs,
                                                            sensorID,
                                                            signalName,
                                                            sensorFileNameExt)        
        

# createFiguresForAll(pictOutputFolder, selectedFactor, selectedContent, sensors, userSensorContentFileName, factorScoresFileName)  

# 6 users  
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccX_2021_09_07-12-32-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccY_2021_09_07-12-32-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_ACC_AccZ_2021_09_07-12-32-08.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_BVP_BVP_2021_09_07-12-32-14.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_EDA_EDA_2021_09_07-12-32-11.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/empatica/F1_empatica_HR_HR_2021_09_07-12-32-17.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_X_m-(s^2)_2021_09_07-12-32-26.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_Y_m-(s^2)_2021_09_07-12-32-35.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_LN_Z_m-(s^2)_2021_09_07-12-32-44.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_X_m-(s^2)_2021_09_07-12-34-35.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Y_m-(s^2)_2021_09_07-12-34-25.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Z_m-(s^2)_2021_09_07-12-34-16.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Accel_WR_Z_m-(s^2)_2021_09_07-12-34-16.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Conductance_microSiemens_2021_09_07-12-32-53.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Conductance_uS-1_2021_09_07-12-33-03.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__GSR_Skin_Resistance_kOhms_2021_09_07-12-33-39.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_X_deg-s_2021_09_07-12-33-58.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_Y_deg-s_2021_09_07-12-34-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Gyro_Z_deg-s_2021_09_07-12-33-48.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__PPG_A12_mV_2021_09_07-12-33-21.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Pressure_BMP280_kPa_2021_09_07-12-33-12.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/shimmer/F1_shimmer__Temperature_BMP280_Degrees Celsius_2021_09_07-12-33-31.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_normal0_x_2021_09_07-12-40-10.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_normal1_x_2021_09_07-12-40-34.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_gaze_point_3d_x_2021_09_07-12-39-46.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_gaze_norm_pos_x_2021_09_07-12-39-30.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_2d_diameter_2021_09_07-12-36-10.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_2d_ellipse_angle_2021_09_07-12-36-15.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_3d_circle_3d_normal_x_2021_09_07-12-37-15.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/pupillabs/F1_pupillabs_left_eye_3d_diameter_2021_09_07-12-36-52.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_x_2021_09_07-12-34-38.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_y_2021_09_07-12-34-41.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_accelerometer_ac_z_2021_09_07-12-34-44.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazeDirection_left_eye_gd_x_2021_09_07-12-34-47.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition_gp_latency_2021_09_07-12-35-11.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition_gp_x_2021_09_07-12-35-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gazePosition3D_gp3d_x_2021_09_07-12-35-14.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_gyroscope_gy_x_2021_09_07-12-35-24.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_pupilCenter_left_eye_pc_x_2021_09_07-12-35-33.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1/tobii/F1_tobii_pupilDim_left_eye_diameter_2021_09_07-12-35-53.pickle')


#4 users                                             
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_AccX_2021_09_06-00-29-59.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_BVP_2021_09_06-00-30-06.pickle')   
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_EDA_2021_09_06-00-30-05.pickle') 
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_HR_2021_09_06-00-30-08.pickle')    
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/empatica/F1_empatica_HR_2021_09_06-00-30-08.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Conductance_microSiemens_2021_09_06-00-30-29.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Conductance_uS-1_2021_09_06-00-30-34.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_GSR_Skin_Resistance_kOhms_2021_09_06-00-30-56.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Pressure_BMP280_kPa_2021_09_06-00-30-40.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_PPG_A12_mV_2021_09_06-00-30-45.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Temperature_BMP280_Degrees Celsius_2021_09_06-00-30-51.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Gyro_X_deg-s_2021_09_06-00-31-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Gyro_Z_deg-s_2021_09_06-00-31-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/shimmer/F1_shimmer_Accel_WR_Z_m-(s^2)_2021_09_06-00-31-18.pickle')

# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_normal0_x_2021_09_06-00-42-46.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_normal1_x_2021_09_06-00-43-00.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_gaze_point_3d_x_2021_09_06-00-42-33.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_gaze_norm_pos_x_2021_09_06-00-42-24.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_2d_diameter_2021_09_06-00-40-39.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_2d_ellipse_angle_2021_09_06-00-40-41.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_3d_circle_3d_normal_x_2021_09_06-00-41-13.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/pupillabs/F1_pupillabs_left_eye_3d_diameter_2021_09_06-00-41-00.pickle')


# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_x_2021_09_06-00-39-45.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_y_2021_09_06-00-39-47.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_accelerometer_ac_z_2021_09_06-00-39-49.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazeDirection_left_eye_gd_x_2021_09_06-00-39-51.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition_gp_latency_2021_09_06-00-40-05.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition_gp_x_2021_09_06-00-40-02.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gazePosition3D_gp3d_x_2021_09_06-00-40-07.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_gyroscope_gy_x_2021_09_06-00-40-13.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_pupilCenter_left_eye_pc_x_2021_09_06-00-40-18.pickle')
# Utils.loadFigFromPickleFile(outputFolderName + 'C1-4 users/tobii/F1_tobii_pupilDim_left_eye_diameter_2021_09_06-00-40-29.pickle')








                                                  