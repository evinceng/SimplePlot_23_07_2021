# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:03:21 2021

@author: Evin Aslan Oguz
"""

#Empatica
import pandas as pd
import matplotlib.pyplot as plt
import Utils
import numpy as np

from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')

   
def createDFAndcomputeShimmerTimestamps(file_path, plotQ=False): #, cutAfter60minutes = False
    
    xlabel = "Timestamp_ms"
    ylabel = "Accel_LN_X_m/(s^2)" #"GSR_Skin_Conductance_uS"#
    timeStampRenameColumn = 'timestamp_s'

    # Header names are really interesting. it is like a tree structure
    # So eliminated the device name and units
    shimmer_df = pd.read_csv(file_path,  delimiter = '\t', header = [1,3]) #, skiprows = [2]
    
    #last column is empty, drop it
    shimmer_df = shimmer_df.iloc[:, :-1]
    
    #rename columns to have the values with units:CAL and without units:UNCAL
    shimmer_df.columns = shimmer_df.columns.map('_'.join)
    # print(shimmer_df.info())
    # print(shimmer_df.columns)
    
    #user26 has values 400, which is way more than typical value which is  25
    # outlier_T = 25
    # shimmer_df = shimmer_df.loc[shimmer_df[ylabel].abs() < outlier_T]
    
    
    #for user26
    # 3.90625
    # shimmer_df[timeStampRenameColumn] = np.arange(start=shimmer_df[xlabel].iloc[0], stop= shimmer_df.shape[0]*3.90625 + shimmer_df[xlabel].iloc[0],  step=3.90625)
    #convert ms to seconds
    shimmer_df[timeStampRenameColumn] = shimmer_df[xlabel]/1000
    # if(cutAfter60minutes):
    shimmer_df = shimmer_df.loc[shimmer_df[timeStampRenameColumn] < 3600]
    # print(shimmer_df[timeStampRenameColumn])
    # print(shimmer_df[xlabel])

    # Get data columns
    
    if plotQ: 
        fig = plt.figure()
        fig.suptitle("Shimmer")
        plt.plot(shimmer_df[timeStampRenameColumn],shimmer_df[ylabel]) #shimmer_df[timeStampRenameColumn],shimmer_df[ylabel]
        plt.grid(True)
        plt.xlabel('t [s]')
        plt.ylabel(ylabel)
        plt.show() 
    
    return shimmer_df.copy(), shimmer_df[timeStampRenameColumn][0]



def createDFAndComputeEmpaticaAccelTimestamps(folder_path, plotQ=False):
    
    
    file_name_lst = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']
    
    all_data_df = pd.read_csv(folder_path + 'ACC.csv')
    
    # Arrange data and time stamp
    timestamp_begin_utc = Utils.getFloatFromOneOrTwoDotsTimestamp(all_data_df.columns[0])
    startTime = Utils.utcToLocalTimeZone(timestamp_begin_utc)
    start_time_ms = Utils.get_secs_from_str(startTime)
    #print(startTime)
    
    #The second row is the sample rate expressed in Hz. so 1/4 we need to add 0.25 s to the starting time if we want the timestamps
    #float sample rate
    timestamp_increment = 1.0/float(all_data_df.iat[0,0])
    
    # drop first two rows
    all_data_df.drop(all_data_df.index[1])
    all_data_df.rename(columns={all_data_df.columns[0]: "AccX", all_data_df.columns[1]: "AccY", all_data_df.columns[2]: "AccZ"}, inplace=True)
    size = len(all_data_df.index)
    
    all_data_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+
    
    # ACC: 1620385021
    # BVP: 1620385021
    # IBI: 1620385021
    # EDA: 1620385021
    
    for curr_fn in file_name_lst:
        
        # Read file
        curr_fpn = folder_path + curr_fn
        curr_df = pd.read_csv(curr_fpn)
        
        # Arrange data and time stamp
        timestamp_begin_utc = Utils.getFloatFromOneOrTwoDotsTimestamp(curr_df.columns[0]) # Nonsense for IBI
        startTime = Utils.utcToLocalTimeZone(timestamp_begin_utc)
        start_time_ms = Utils.get_secs_from_str(startTime)
        #print(startTime)
        
        #float sample rate: read from the seconf 
        size = len(curr_df.index)
        timestamp_increment = 1.0/float(curr_df.iat[0,0])
        
        
        
        # Arrange data
        if curr_fn =='ACC.csv':
            
            # drop first two rows and name columns
            curr_df.drop(curr_df.index[1])
            curr_df.rename(columns={curr_df.columns[0]: "AccX", curr_df.columns[1]: "AccY", curr_df.columns[2]: "AccZ"}, inplace=True)
                    
            #  Generate time stamps
            curr_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+
            
            # Make a copy
            empatica_ACC_df = curr_df.copy()
        
        if curr_fn =='BVP.csv':
            timestamp_increment = 1.0/float(curr_df.iat[0,0])
            curr_df.drop(curr_df.index[1]) # drop first two rows
            curr_df.rename(columns={curr_df.columns[0]: 'BVP'}, inplace=True)
            
            #  Generate time stamps
            size = len(curr_df.index)
            curr_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+
            
            # Make a copy
            empatica_BVP_df = curr_df.copy()
        
        if curr_fn =='EDA.csv':
            timestamp_increment = 1.0/float(curr_df.iat[0,0])
            curr_df.drop(curr_df.index[1]) # drop first two rows
            curr_df.rename(columns={curr_df.columns[0]: 'EDA'}, inplace=True)
            
            #  Generate time stamps
            size = len(curr_df.index)
            curr_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+
            
            # Make a copy
            empatica_EDA_df = curr_df.copy()
            
        if curr_fn =='HR.csv':
            timestamp_increment = 1.0/float(curr_df.iat[0,0])
            curr_df.drop(curr_df.index[1]) # drop first two rows
            curr_df.rename(columns={curr_df.columns[0]: 'HR'}, inplace=True)
            
            #  Generate time stamps
            size = len(curr_df.index)
            curr_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+
            
            # Make a copy
            empatica_HR_df = curr_df.copy()
        
        if curr_fn =='IBI.csv':
            #timestamp_increment = 1.0/float(curr_df.iat[0,0])
            curr_df.drop(curr_df.index[0]) # drop first row
            curr_df.rename(columns={curr_df.columns[0]: 'IBI_time', curr_df.columns[1]: 'IBI_dur'}, inplace=True)

            # Make a copy
            empatica_IBI_df = curr_df.copy()
        
        if curr_fn =='TEMP.csv':
            timestamp_increment = 1.0/float(curr_df.iat[0,0])
            curr_df.drop(curr_df.index[1]) # drop first two rows
            curr_df.rename(columns={curr_df.columns[0]: 'TEMP'}, inplace=True)
            
            #  Generate time stamps
            size = len(curr_df.index)
            curr_df['timestamp_s'] = [start_time_ms + timestamp_increment*x for x in range(size)] #timestamp_begin_ms+                            
        
            # Make a copy
            empatica_TEMP_df = curr_df.copy()
             
    
    
    if plotQ:
        plt.figure("Empatica_ACC")
        plt.plot(empatica_ACC_df['timestamp_s'], empatica_ACC_df.iloc[:,0], color='g')
        plt.xlabel('t [s]')
        plt.show()
    
    # print(empatica_df.info())
    return empatica_ACC_df, empatica_BVP_df, empatica_EDA_df, empatica_HR_df, empatica_IBI_df, empatica_TEMP_df, startTime 



#need to add code for gaze file..
def createDFAndFilterPupilRelevantData(file_path, plotQ=False):
    x_label = 'pupil_timestamp'
    y_label =  'diameter'
    outlier_T = 100

    timeStampRenameColumn = 'timestamp_s'
        
    file_name_lst = ['pupil_positions.csv', 'gaze_positions.csv']
    # pupil positions
    df = pd.read_csv(file_path + file_name_lst[0])

    #filter the rows with low confidence 
    df = df.loc[df['confidence'] > 0.8]
    
    # Filter outliers
    df = df.loc[df[y_label] < outlier_T]
    
    #subtract the starting time to get absolute seconds 
    df[timeStampRenameColumn] = df[x_label] - df[x_label].iloc[0]
    
    # gaze data
    gaze_x_label = 'gaze_timestamp'
    gaze_y_label =  'gaze_point_3d_x'
    
    gaze_df = pd.read_csv(file_path + file_name_lst[1])
    if not gaze_df.empty:        
        #filter the rows with low confidence 
        gaze_df = gaze_df.loc[gaze_df['confidence'] > 0.8]
        
        # Filter outliers
        gaze_df = gaze_df.loc[gaze_df[gaze_y_label].abs() < 3*outlier_T]
        
        #subtract the starting time to get absolute seconds 
        gaze_df[timeStampRenameColumn] = gaze_df[gaze_x_label] - gaze_df[gaze_x_label].iloc[0]
    else: 
        gaze_df[timeStampRenameColumn] = gaze_df[gaze_x_label]
        print("Gaze file is empty!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        
    if plotQ:
        #plt.plot(df[x_label], df[y_label])
        plt.plot(df[timeStampRenameColumn], df[y_label])
        plt.xlabel('t [s]')
        plt.ylabel(y_label)
        plt.show()
    
    # print(df[timeStampRenameColumn])
    return df.copy(), gaze_df.copy() #, df[timeStampRenameColumn][0]

def createTobiiDf(tobiiFilePath, plotQ=False):
    #user5 only has pupilDim
        #
    fnList = ["accelerometer.csv", "gazeDirection.csv", "gazePosition.csv",
               "gazePosition3D.csv", "gyroscope.csv", "pupilCenter.csv", "pupilDim.csv"]
    
    dfList = []
    for fileName in fnList:
        df = pd.read_csv(tobiiFilePath + fileName)
        df["timestamp_s"] = df['utc_time'].apply(Utils.get_secs_from_str)
        df["timestamp_s"] =  df["timestamp_s"]-(df["timestamp_s"].iloc[0])
        # right_eye_df = df[df['eye'] == 'right']
        # left_eye_df = df[df['eye'] == 'left']
        # print(startime)
        # startsec = get_secs_from_str(startime)
        # print(startsec)
        # print(df)
        dfList.append(df)
        if fileName =="pupilDim.csv" and plotQ:
            xLabel = "timestamp_s"
            yLabel = "diameter"
            right_eye_df = df[df['eye'] == 'right']
            left_eye_df = df[df['eye'] == 'left']
            #plot
            plt.figure()
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.plot(right_eye_df[xLabel], right_eye_df[yLabel])
            plt.plot(left_eye_df[xLabel], left_eye_df[yLabel])
            plt.show()
            
    return dfList

def loadData(uID, root_folder, usersDictFilePath):
    data_fn_dict =  Utils.readDict(usersDictFilePath)
    
    shimmer_df = None
    if not pd.isnull(data_fn_dict[uID]['shimmerFilePath']):        
        shimmer_file_name = root_folder + data_fn_dict[uID]['shimmerFilePath']
        shimmer_df, start_time = createDFAndcomputeShimmerTimestamps(shimmer_file_name)
    
    empatica_file_path = root_folder + data_fn_dict[uID]['empaticaFolderPath']
    empatica_ACC_df, empatica_BVP_df, empatica_EDA_df, empatica_HR_df, empatica_IBI_df, empatica_TEMP_df, startTime = createDFAndComputeEmpaticaAccelTimestamps(empatica_file_path)
    
    
    dfDict = {}
    dfDict['shimmer_df'] = shimmer_df
    dfDict['empatica_ACC_df'] = empatica_ACC_df
    dfDict['empatica_BVP_df'] = empatica_BVP_df
    dfDict['empatica_EDA_df'] = empatica_EDA_df
    dfDict['empatica_HR_df'] = empatica_HR_df
    dfDict['empatica_IBI_df'] = empatica_IBI_df
    dfDict['empatica_TEMP_df'] = empatica_TEMP_df
    
    
    if not pd.isnull(data_fn_dict[uID]['pupilLabsFolderPath']):        
        pupillabs_file_path = root_folder + data_fn_dict[uID]['pupilLabsFolderPath']
        pupillabs_df, pupillabs_gaze_df = createDFAndFilterPupilRelevantData(pupillabs_file_path) #, start_time
        dfDict['pupillabs_df'] = pupillabs_df
        dfDict['pupillabs_gaze_df'] = pupillabs_gaze_df
    
    elif not pd.isnull(data_fn_dict[uID]['tobiiFolderPath_csharp']):        
        tobiiFilePath = root_folder + data_fn_dict[uID]['tobiiFolderPath_csharp']
        tobii_dfList = createTobiiDf(tobiiFilePath)
        # fnList = ["accelerometer.csv", "gazeDirection.csv", "gazePosition.csv",
        #       "gazePosition3D.csv", "gyroscope.csv", "pupilCenter.csv", "pupilDim.csv"]
        dfDict['tobii_accelerometer_df'] = tobii_dfList[0]
        dfDict['tobii_gazeDirection_df'] = tobii_dfList[1]
        dfDict['tobii_gazePosition_df'] = tobii_dfList[2]
        dfDict['tobii_gazePosition3D_df'] = tobii_dfList[3]
        dfDict['tobii_gyroscope_df'] = tobii_dfList[4]
        dfDict['tobii_pupilCenter_df'] = tobii_dfList[5]
        dfDict['tobii_pupilDim_df'] = tobii_dfList[6]
        #for user5
        # dfDict['tobii_pupilDim_df'] = tobii_dfList[0]
    
    
    return dfDict
    # return shimmer_df, empatica_ACC_df, empatica_BVP_df, empatica_EDA_df, empatica_HR_df, empatica_IBI_df, empatica_TEMP_df, pupillabs_df

# pupil, gaze = createDFAndFilterPupilRelevantData("D:/LivingLabMeasurements/user29/Pupillabs/20210518090526482/exports/000/", plotQ=True)
#run for testing
# shimmer_df, empatica_ACC_df, empatica_BVP_df, empatica_EDA_df, empatica_HR_df, empatica_IBI_df, empatica_TEMP_df, pupillabs_df = loadData(1, "Data/livinglabUsersFileFolderNames.xlsx")

# print(shimmer_df.head())
# print(empatica_ACC_df.head())
# print(empatica_BVP_df.head())
# print(empatica_EDA_df.head())
# print(empatica_HR_df.head())
# print(empatica_IBI_df.head())
# print(empatica_TEMP_df.head())
# print(pupillabs_df.head())

# df, timestampdf = createDFAndcomputeShimmerTimestamps("D:/LivingLabMeasurements/user26/Shimmer/20210519110048 Device5E7F.csv", plotQ=True)

# df, timestampdf = createDFAndcomputeShimmerTimestamps("D:/LivingLabMeasurements/user27/Shimmer/20210519091054 Device5E7F.csv", plotQ=True)
# createTobiiDf("D:/evin/SpyderProjects/SimplePlot/Data/users/user38/Tobii/pupilDim.csv", plotQ=True)