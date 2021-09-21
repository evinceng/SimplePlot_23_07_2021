# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:57:13 2021

@author: evinao
"""
#resample the signals

import numpy as np
import Tools.interpolate_funs as intFuns
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import glob, os
import pandas as pd
from scipy.interpolate import CubicSpline
import json
import pathlib
from scipy import stats
from pandas.api.types import is_numeric_dtype
import time
from datetime import timedelta
import pickle
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')





freq = 30.0


#change the root folder to your local folder
# rootFolder = "C:/Users/evinao/Documents/user"
rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user"
# rootFolder = 'C:/Users/Andrej Košir/Dropbox (Lucami)/002-CurrPapers/000-ModelMME/00-Code/SplineTest/output/user'
# uID = 3

    
def writeOneTckDictToJson(tck, fileName):
    tckZip = zip(('t', 'c', 'k'), tck)
    #ndarray is not serializable error if not convert it to list
    tckDict = dict((x, y if isinstance(y, int) else y.tolist()) for x, y in tckZip)
    with open(fileName, 'w') as file:
        json.dump(tckDict, file)      
  
def readOneTckDictFromJson(tckJsonFileName):
     # Opening JSON file
    with open(tckJsonFileName) as json_file:
        tckDict = json.load(json_file)
    tck = (np.asarray(tckDict["t"]), np.asarray(tckDict["c"]), tckDict["k"])
    print(tck)
    return tck
      
def writeTckDictsToJson(tck, fileName):
    with open(fileName, 'w') as file:
        json.dump(tck, file)      

def readTckDictFromJson(tckJsonFileName):
     # Opening JSON file
    with open(tckJsonFileName) as json_file:
        tckDict = json.load(json_file)
    for sensor in tckDict.keys():
        for signal in tckDict[sensor].keys():
            tckDict[sensor][signal]["t"] = np.asarray(tckDict[sensor][signal]["t"])
            tckDict[sensor][signal]["c"] = np.asarray(tckDict[sensor][signal]["c"])
            tck = (np.asarray(tckDict[sensor][signal]["t"]), np.asarray(tckDict[sensor][signal]["c"]), tckDict[sensor][signal]["k"])
            print(tck)
    return tckDict


def plotFigure(xIn, yIn, new_y, title):
    plt.figure(1)
    plt.title(title) 
    plt.plot(xIn, yIn, color = 'b', label='Orig')
    plt.plot(xIn, new_y, color = 'r', label='Interp.')
    plt.legend()
    plt.show()
    


#for one file interpolate and plot

# fileNameList = ["_empatica_ACC", "_empatica_BVP", "_empatica_EDA","_empatica_HR",
#                 "_empatica_IBI", "_empatica_TEMP", "_pupillabs", "_pupillabs_gaze",
#                 "_shimmer_df", "_tobii_accelerometer", "_tobii_gazeDirection",
#                 "_tobii_gazePosition", "_tobii_gazePosition3D", "_tobii_gyroscope",
#                 "_tobii_pupilCenter", "_tobii_pupilDim"]
# # to test one file
# filePath = rootFolder + str(uID) + "/uID-" + str(uID) + fileNameList[0]

# df = pd.read_csv(filePath + "_df_synced.csv")
# signalName = 'AccX'
# print(df)

# xIn = np.array(df['timestamp_s'])#np.array([-0.1, 0.1, 1.1, 1.9, 6.1, 7.9, 8.1])
# yIn = np.array(df[signalName]) #np.array([2, 1.0, 1.8, 0.1, 6.5, 4.1, 2.3])
# tMin, tMax = df['timestamp_s'].iloc[0], df['timestamp_s'].iloc[-1]
# kIn = 3 #the order of each polynomial file in there, smoothness of the data 
# TsIn = 1.0/30.0
# fCode = 1
# tck = intFuns.interpolateBS(xIn, yIn, tMin, tMax, kIn, TsIn, fCode)

# writeTckDictToJson(tck, filePath + '_' + signalName + '_tck.json')

# readTckDictFromJson(filePath + '_' + signalName + '_tck.json')
# # Scipy
# # cs = CubicSpline(xIn, yIn)

# new_y = splev(xIn, tck)
# plt.figure(1)
# plt.title('interpolateBS ' + signalName + " plot") 
# plt.plot(xIn, yIn, color = 'b', label='Orig')
# plt.plot(xIn, new_y, color = 'r', label='Interp.')
# # plt.plot(xIn, cs(xIn), color = 'm', label='Scipy.')
# plt.legend()
# plt.show()


#for the files in folder s

#not used anymore, pandas bfill and ffill methods are used
def getPrevAndNextHealthyIndexes2(df, outliers, loc):
    size = df.shape[0]
    prev_index = loc - 1
    next_index = loc + 1
    
    if prev_index < 0:
        prev_index = -1 
        # print("There were no values till the start of the df, take only next_index")
    else:    
        while prev_index in outliers:
            prev_index = prev_index - 1
            if prev_index <= 0:
                # print("There were no values till the start of the df, take only next_index")
                prev_index = -1
                break
    if next_index >= df.shape[0]:
        next_index = -1
        # print("There were no values till the end of the, going to the start of the df")
    else:
        while next_index in outliers:
            next_index = next_index + 1
            if next_index >= size - 1:
                # print("There were no values till the end of the, going to the start of the df")
                next_index = -1
                break
    return prev_index, next_index

def correctoutliers_pandasf(df):
    min_percent = 25
    max_percent = 75
    #remove outliers    
    if not is_numeric_dtype(df):
        return df
    # Identify index locations above cutoff
    # Identify index locations above cutoff
    
    print("mean before is ")
    print(df.mean())
    
    notnull_df = df[~np.isnan(df)]
    percentile_df = notnull_df[np.logical_and(notnull_df > np.percentile(notnull_df,min_percent),notnull_df < np.percentile(notnull_df,max_percent))]
    
    mean = percentile_df.mean()
    std = percentile_df.std()
    
    # (df < mean - std*3) | (df > mean + std*3)
    
    #careful df.where keeos the value if the condition is ok otherwise replace it by the value provided
    df.where((df <  mean + std*3)  & (df > mean - std*3), np.nan, inplace=True)

    # print(df)
    # value sin the middle
    df.where(df.notnull(), other=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2, inplace=True)
    # print(df)
    df.where(df.notnull(), other=(df.fillna(method='ffill')), inplace=True)
    # print(df)
    df.where(df.notnull(), other=(df.fillna(method='bfill')), inplace=True)
    
    print("Mean after is ")
    print(df.mean())
    return df

#not used anymore, instead correctoutliers_pandasf is used, which is 5 times faster
def correctOutliers(df):
    
    # #reset the index incase it is a df from filtered df such as pupil labs right and left eye dfs
    # df  = df.reset_index(drop=True)
    
    min_percent = 25
    max_percent = 75
    #remove outliers    
    if not is_numeric_dtype(df):
        return df
    # Identify index locations above cutoff
    # Identify index locations above cutoff
    
    print("mean before is ")
    print(df.mean())
    
    notnull_df = df[~np.isnan(df)]
    percentile_df = notnull_df[np.logical_and(notnull_df > np.percentile(notnull_df,min_percent),notnull_df < np.percentile(notnull_df,max_percent))]
    # quantile_df = df[df > np.percentile(df,min_percent) & df < np.percentile(df,max_percent)]
    
    # size = df.shape[0]
    # outlier_min_percent = int(min_percent/100*size)
    # outlier_max_percent = int(max_percent/100*size)    
    # sorted_df = df.sort_values().iloc[outlier_min_percent:outlier_max_percent]
    # sorted_df = sorted_df.iloc[outlier_min_percent:outlier_max_percent]
    
    mean = percentile_df.mean()
    std = percentile_df.std()
    # print(sorted_df)
    
    # or interquartile
    # Q1 = df[col].quantile(0.10)
    # Q9 = df[col].quantile(0.90)
    # IQR = Q9 - Q1
    #outliers are nan  values and the values above or less than 3 times the std deviation
    outliers = df[(np.isnan(df)) | (df < mean - std*3) | (df > mean + std*3)].index # (np.abs(df-mean) <= std*3)
    # outliers = df[(df < mean - std*3) | (df > mean + std*3)].index
    prev_outlier_index = -100
    new_val = 0
    # print(outliers)
    # Browse through outliers and average according to index location
    for loc in outliers:
        # Get index location
        # print(loc)
        if loc == prev_outlier_index + 1:
            df[loc] = new_val
            prev_outlier_index = loc
            continue
        else:
            prev_healthy_index, next_healthy_index = getPrevAndNextHealthyIndexes2(df, outliers, loc)
            new_val = -10000
            if prev_healthy_index == -1 and next_healthy_index == -1:
                print("!!!!!!!!!!Couldn't find healty values. Error !!!!!!!!!!!!!!")
            elif prev_healthy_index == -1:
                new_val = df[next_healthy_index]
            elif next_healthy_index == -1:
                new_val = df[prev_healthy_index]
            else:
                new_val = (df[prev_healthy_index]+df[next_healthy_index])/2.0
            # meanVal = (df[col][prev_healthy_index] + df[col][next_healthy_index]) / 2
            df[loc] = new_val
            # print(df)
                
            prev_outlier_index = loc
    print("Mean after is ")
    print(df.mean())
    return df

def resample(uID, file, df, tckDict, fileName, plotIndex):
    #reset the index incase it is a df from filtered df such as pupil labs right and left eye dfs
    # if plotIndex != 45:
    #     plotIndex = plotIndex + 1
    #     return plotIndex
    resampled_df = pd.DataFrame()
    
    tckDict[file] = {}
    
    #check if the time is decreasing, if yes sort it according to timestamp
    xIn = np.array(df['timestamp_s'])
    if np.min(xIn[1:] - xIn[:-1]) < 0:
        decreasingTimeText = "SORTING:The time is decreasing in some part " + fileName
        with open(rootFolder + str(uID) + "/Resampled/decreasingTimes.txt", 'w') as f:
            f.write(decreasingTimeText)
        print(decreasingTimeText)
        df.sort_values("timestamp_s", inplace = True)
        #assign  xIn as well
        # df = df.sort_values("timestamp_s", inplace = True)
        # xIn = df.sort_values("timestamp_s", inplace = True)
        # xIn = np.sort(xIn)
        
        # if the time doesn't start froma t least 0.01 ad copy of the first row with time 0
    if df['timestamp_s'].iloc[0] >= 0.01:
        newrow = pd.DataFrame(df.head(1))
        newrow['timestamp_s'].iloc[0] = 0
        df = pd.concat([newrow, df])
        
    df.index = pd.RangeIndex(len(df.index))
    xIn = np.array(df['timestamp_s'])
    # df.reset_index(drop=True)
    
    
    # assign new time_stamps for one df only once
    resampled_df['timestamp_s'] = np.arange(0, np.max(xIn), 1.0/freq).tolist()
    
    #convert resampled times to np array incase datfarame will make problems
    
    xIn_resampled = np.array(resampled_df['timestamp_s'])
    for signalName in df.columns:
        # if not (signalName == "diameter"):
        #     continue
        #do not resample the following signals either because they are strings (gd_eye)
        #or not needed resampling (confidence)
        if (signalName == "Unnamed: 0" or ("time" in signalName) or 
            ("Time" in signalName) or ("no_units" in signalName)
            or (signalName == "gd_eye") or (signalName == "eye")
            or (signalName == "eye_id") 
            or (signalName == "method") or (signalName == "world_index")
            or (signalName == "confidence") 
            or (signalName == "model_confidence") or (signalName == "model_id")
            or (signalName == "base_data")):
            continue
        print(signalName)
        # if signalName != 'eye_center0_3d_x':
        #     continue
        # df[signalName] = correctOutliers(df[signalName])
        correctoutliers_pandasf(df[signalName]) #df[signalName] = 
        
        title = 'interpolateBS '  + fileName +  ' ' + signalName + ' plot'
        #since tehre are signalnames  Accel_LN_X_m/(s^2) replace slash / with - 
        title = title.replace("/", "-")
        
        # #check if the time is decreasing, if yes sort it according to timestamp
        # xIn = np.array(df['timestamp_s'])
        # if np.min(xIn[1:] - xIn[:-1]) < 0:
        #     decreasingTimeText = "The time is decreasing in some part " + fileName + " " + signalName
        #     with open(rootFolder + str(uID) + "/Resampled/decreasingTimes.txt", 'w') as f:
        #         f.write(decreasingTimeText)
        #     print(decreasingTimeText)
        #     # df = df.sort_values("timestamp_s", inplace = True)
        #     # xIn = df.sort_values("timestamp_s", inplace = True)
        #     xIn = np.sort(xIn)
        
        #start interpolation
        # xIn = np.array(df['timestamp_s'])
        yIn = np.array(df[signalName])
        tMin, tMax = xIn[0], xIn[-1]
        kIn = 3
        TsIn = 1.0/freq
        fCode = 1
                
        tck = intFuns.interpolateBS(xIn, yIn, tMin, tMax, kIn, TsIn, fCode)
                    
        tckZip = zip(('t', 'c', 'k'), tck)
        #ndarray is not serializable error if not convert it to list
        tck_ZipDict = dict((x, y if isinstance(y, int) else y.tolist()) for x, y in tckZip)
        tckDict[file][signalName] = tck_ZipDict
        # print(tckDict[file][signalName])
        
        new_y = splev(xIn_resampled, tck)
        resampled_df[signalName] = new_y
        
        fig = plt.figure(plotIndex)
        plt.title('interpolateBS '  + fileName + " " + signalName + " plotIndex=" + str(plotIndex)) 
        plt.plot(xIn, yIn, color = 'b', label='Orig')
        plt.plot(xIn_resampled, new_y, color = 'r', label='Interp.')
        plt.legend()
        # plt.show()
        plt.savefig(rootFolder + str(uID) + "/Figs/" + title + str(plotIndex) + ".jpg")
        pickle.dump(fig, open(rootFolder + str(uID) + "/Figs/" + title + str(plotIndex) +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
        plt.close(fig)
        plotIndex += 1
        
    resampled_df.to_csv(rootFolder + str(uID) + "/Resampled/" + fileName + "_resampled.csv")
    return plotIndex

def loadFigFromPickleFile(filename):
    figx = pickle.load(open(filename, 'rb'))
    figx.show() # Show the figure, edit it, etc.!
    

def resampleuIDCsvFiles(rootFolder , uID):
    start_time = time.monotonic()
    os.chdir(rootFolder + str(uID))
    # print(glob.glob("*.csv"))
        
    tckDict = {}
    plotIndex = 1
    
    for file in glob.glob("*.csv"):
        #don't know what to do withIBI_time
        print(file)
        if "IBI" in file:
            print("IBI file don't know what to do")
            continue
       
        df = pd.read_csv(file)
        
        #create figs and resampled folders
        pathlib.Path(rootFolder + str(uID) + "/Figs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(rootFolder + str(uID) + "/Resampled").mkdir(parents=True, exist_ok=True)
                                    
        fileName = file.replace("_df_synced.csv", "")
        
        if ("tobii_gazeDirection" in file):
            right_eye_df = df[df["gd_eye"] == "right"]
            left_eye_df = df[df["gd_eye"] == "left"]
            print("Right eye resampling")
            plotIndex = resample(uID, file, right_eye_df, tckDict, fileName + "_right_eye", plotIndex)
            print("Left eye resampling")
            plotIndex = resample(uID, file, left_eye_df, tckDict, fileName + "_left_eye", plotIndex)
            
        elif ("tobii_pupilCenter" in file) or ("tobii_pupilDim" in file):
            right_eye_df = df[df["eye"] == "right"]
            left_eye_df = df[df["eye"] == "left"]
            print("Right eye resampling")
            plotIndex = resample(uID, file, right_eye_df, tckDict, fileName + "_right_eye", plotIndex)
            print("Left eye resampling")
            plotIndex = resample(uID, file, left_eye_df, tckDict, fileName + "_left_eye", plotIndex)
        elif ("pupillabs_df_synced" in file):
            right_eye_df = df[df["eye_id"] == 0]
            left_eye_df = df[df["eye_id"] == 1]
            # there are two measurements with differnet methods one is : "2d c++" the other is "pye3d 0.0.6 post-hoc"
            #the columns after the 7th are for 3d measurments
            right_eye_2d_df = right_eye_df[right_eye_df['method']== "2d c++"].loc[:,
            ['timestamp_s', 'norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_center_x',
              'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle']]
            
            right_eye_pye3d_df = right_eye_df[right_eye_df['method']== "pye3d 0.0.6 post-hoc"]
            print("Right eye method: 2d c++ resampling")
            plotIndex = resample(uID, file, right_eye_2d_df, tckDict, fileName + "_right_eye_2d", plotIndex)
            print("Right eye method:pye3d 0.0.6 post-hoc resampling")
            plotIndex = resample(uID, file, right_eye_pye3d_df, tckDict, fileName + "_right_eye_3d", plotIndex)
            #left eye
            left_eye_2d_df = left_eye_df[left_eye_df['method']== "2d c++"].loc[:,
            ['timestamp_s', 'norm_pos_x', 'norm_pos_y', 'diameter', 'ellipse_center_x',
             'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle']]
            left_eye_pye3d_df = left_eye_df[left_eye_df['method']== "pye3d 0.0.6 post-hoc"]
            print("Left eye method: 2d c++ resampling")
            plotIndex = resample(uID, file, left_eye_2d_df, tckDict, fileName + "_left_eye_2d", plotIndex)
            print("Left eye method: pye3d 0.0.6 post-hoc resampling")
            plotIndex = resample(uID, file, left_eye_pye3d_df, tckDict, fileName + "_left_eye_3d", plotIndex)
        else:
            #resample
            plotIndex = resample(uID, file, df, tckDict, fileName, plotIndex)
    
                    
    writeTckDictsToJson(tckDict, rootFolder + str(uID) + "/spline_tck.json")
    readTckDictFromJson(rootFolder + str(uID) + "/spline_tck.json")
    
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    with open(rootFolder + str(uID) + "/Resampled/ExecutionTime.txt", 'w') as f:
        f.write(str(timedelta(seconds=end_time - start_time)))
           

def resampleAllFolders(rootFolder):
    
    #1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
              # 37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60
              #2,3,4,5,6,7,8,
    uIDlist = [9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]
    
    for userid  in uIDlist:
        print(userid)
        resampleuIDCsvFiles(rootFolder, userid)
        
# resampleuIDCsvFiles(rootFolder, 4)

resampleAllFolders(rootFolder)

# loadFigFromPickleFile(rootFolder + '60/Figs/interpolateBS uID-60_empatica_ACC AccX plot1.pickle')
