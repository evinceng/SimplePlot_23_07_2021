# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:16:04 2021

@author: evinao
"""
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from datetime import datetime

selectedFactor = 4
selectedContent = 'C1'

lowFactorUserIDs = [14,16,17,10] #tobii [14,16,17,10] # empatica, shimmer pupillabs [1,2,3,4]
highFactorUserIDs = [50,51,52,53] # empattica, shimmer [5,6,7,8] # pupillabs [33,34,35,36] #tobii

sensorID = 3
signalName = 'pc_x' #'diameter' # 'EDA'
sensorFileNameExt = 'pupilCenter_left_eye'#'left_eye_2d' #EDA

outputFolderName = "PictorialProof/"

rootFolder = "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user"
usersDictFileName = "Data/usersDict.xlsx"

sensors = {1:'empatica', 2:'shimmer', 3:'tobii', 4:'pupilLabs'}

signals = ['AccX', 'AccY', 'AccZ', 'EDA', 'BVP', 'IBI', 'HR',
           'Accel_LN_X_m/(s^2)', 'Accel_LN_Y_m/(s^2)',
           'Accel_LN_Z_m/(s^2)', 'GSR_Range_no_units.1',
           'GSR_Skin_Conductance_microSiemens', 'GSR_Skin_Conductance_uS.1',
           'Pressure_BMP280_kPa', 'PPG_A12_mV',
           'Temperature_BMP280_Degrees Celsius', 'GSR_Skin_Resistance_kOhms',
           'Gyro_Z_deg/s', 'Gyro_X_deg/s', 'Gyro_Y_deg/s',
           'Accel_WR_Z_m/(s^2)', 'Accel_WR_Y_m/(s^2)', 'Accel_WR_X_m/(s^2)',
           'norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_angle',
           'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z',
           'norm_pos_x', 'norm_pos_y', 'gaze_point_3d_x', 'gaze_point_3d_y',
           'gaze_point_3d_z', 'gaze_normal0_x', 'gaze_normal0_y',
           'gaze_normal0_z', 'gaze_normal1_x', 'gaze_normal1_y',
           'gaze_normal1_z', 'ac_x', 'ac_y', 'ac_z', 'gd_x', 'gd_y', 'gd_z',
           'gp_x', 'gp_y', 'gp_latency', 'gp3d_x', 'gp3d_y', 'gp3d_z',
           'gy_x', 'gy_y', 'gy_z', 'pc_x', 'pc_y', 'pc_z', 'diameter']

mmaesFactors = {1:'AE', 2:'RE', 3:'AA', 4:'PI'}

# print(mmaesFactors[1])

def fileNameGenerator(userID, sensorID, sensorFileNameExt = ""):
    fileNameStr = rootFolder + str(userID) + "/Resampled/uID-" + str(userID) + "_" + sensors[sensorID]
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

def generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(lowFactorUserIDs,
                                                               highFactorUserIDs,
                                                               sensorID,
                                                               signalName,
                                                               sensorFileNameExt = ""):
    
    subPlotCount = max(len(lowFactorUserIDs), len(highFactorUserIDs))
    #take min and max 
    fig, ax = plt.subplots(2, subPlotCount) #, sharey=True
    fig.suptitle(sensors[sensorID] +' ' + sensorFileNameExt +' ' + signalName + ' ' + selectedContent)
    
    ax[0][0].set_title('Low score F' + str(selectedFactor))
    ax[1][0].set_title('High score F' + str(selectedFactor))
    
    # plot the first row of subplots
    generateSignalSubplots(ax, 0, lowFactorUserIDs, sensorID, signalName, sensorFileNameExt)    
    # plot the second row of subplots
    generateSignalSubplots(ax, 1, highFactorUserIDs, sensorID, signalName, sensorFileNameExt)
    
    # draw the first row of video ad lines
    generateAdVidLines(ax, 0, usersDictFileName, lowFactorUserIDs, selectedContent)    
    # draw the second row of video ad lines
    generateAdVidLines(ax, 1, usersDictFileName, highFactorUserIDs, selectedContent) 
    
    now = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    saveFileFig = outputFolderName + 'F' + str(selectedFactor) + '_' + sensors[sensorID] + '_' + signalName + '_' + now
    plt.savefig(saveFileFig +'.jpg')
    Utils.writePickleFile(fig, saveFileFig)
    # pickle.dump(fig, open(saveFileFig +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
    plt.show()
    

# print(fileNameGenerator(1, 2))

generateSubPlotsofOneSignalOfMultipleUsersWithAdVideoTimes(lowFactorUserIDs,
                                                            highFactorUserIDs,
                                                            sensorID,
                                                            signalName,
                                                            sensorFileNameExt)
                                                           
# Utils.loadFigFromPickleFile(outputFolderName + 'F1_empatica_EDA_2021_08_25-12-24-41.pickle')                                                          