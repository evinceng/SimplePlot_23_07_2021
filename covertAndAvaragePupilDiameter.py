# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:22:21 2021

@author: evinao
"""

import glob, os
import os.path
from pathlib import Path
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')
import pandas as pd
import numpy as np

contents = ['C1','C2','C3','C4'] #, 'C2','C3','C4'

sensors = ['empatica', 'shimmer', 'pupillabs', 'tobii']

avarageColName = 'diameter'
newFileName = 'tobii_diameter'
meanColName = 'inside_room_Mean'

rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user"
sigCorrFolder = "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/SignalCorrelation/"


uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]

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


tobii_leftEye_FileName = "Mean_Std_Kurtosis_inside_room_tobii_pupilDim_left_eye_diameter_AE_all.csv"
pupillabs_leftEye_FileName = "Mean_Std_Kurtosis_inside_room_pupillabs_left_eye_2d_diameter_AE_all.csv"

def getAvarageAndThenMean(leftEyeFileName):
    left_eye_df = pd.read_csv(leftEyeFileName)
    rightFileName = leftEyeFileName.replace('left', 'right')
    right_eye_df = pd.read_csv(rightFileName)

    mean = np.mean((left_eye_df[meanColName] + right_eye_df[meanColName])/2.0)
    return mean

def getMeanDiameterMeanValues(tobiiLeftEyeFileName, pupillabsLeftEyeFileName):
    tobii_mean = getAvarageAndThenMean(tobiiLeftEyeFileName)
    pupillabs_mean = getAvarageAndThenMean(pupillabsLeftEyeFileName)
    
    return tobii_mean, pupillabs_mean


# tobii_diameter_mean, pupilLabs_diamter_mean =  getMeanDiameterMeanValues(sigCorrFolder+'C1/tobii/'+tobii_leftEye_FileName, sigCorrFolder+'C1/pupillabs/'+pupillabs_leftEye_FileName)

#these values are return of the above function 
tobii_diameter_mean = 4.3201111540620225
pupilLabs_diamter_mean = 37.236853739611774

print(tobii_diameter_mean )
print(pupilLabs_diamter_mean )

for uID in uIDlist:        
    os.chdir(rootFolder + str(uID) +"/Resampled")
    # get all .csv files in the user folder
    for file in glob.glob("*.csv"):        
        #tobi or pupillabs
        if "tobii_pupilDim_left_eye" in file or "pupillabs_left_eye_2d" in file:
            print(file)
            left_eye_df = pd.read_csv(file)
            right_eye_df_filename = file.replace('left', 'right')
            right_eye_df = pd.read_csv(right_eye_df_filename)
            
            avarage_df = pd.DataFrame()
            
            avarage_df['timestamp_s'] = left_eye_df['timestamp_s']
            avarage_df[avarageColName] = (left_eye_df[avarageColName]+right_eye_df[avarageColName])/2.0
            avarage_df = avarage_df.dropna(axis='rows')
            if 'pupillabs' in file:
                avarage_df[avarageColName] = avarage_df[avarageColName] *(tobii_diameter_mean/pupilLabs_diamter_mean)
                #     sig_lp_x = sig_lp_x * (3.5/32) # 3.5 non pupil labs mean, 32 is pupillabs mean
            
            if "tobii_pupilDim_left_eye" in file:
                avarageFileName = file.replace('tobii_pupilDim_left_eye', newFileName)
            else:
                avarageFileName = file.replace('pupillabs_left_eye_2d', newFileName)
                 
            avarage_df.to_csv(avarageFileName)
                        
            
   


