# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:22:33 2021

@author: evinao
"""

import glob, os
import os.path
from pathlib import Path
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')
import pandas as pd
from shutil import copyfile


contents = ['C1','C2','C3','C4'] #, 'C2','C3','C4'

sensors = ['empatica', 'shimmer', 'tobii']

rootFolder = "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/SignalCorrelationR3_12users_RE/"

outputFile = "C:/Users/evinao/Documents/Paper2Data/UserFeaturesR3.csv"

uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]

columnNameList = ['uID', 'AE', 'RE', 'AA', 'PI']

selectedFactor = ['AE']

allFeatures_df = pd.DataFrame(data=uIDlist, columns=['uID'])

features_dfs_list = [allFeatures_df]

strList = []
name = ''

result = pd.DataFrame()


# def getAllUsersScores(userIDList, factorScoresFileName, selectedFactor):
#     scores_df = pd.read_csv(factorScoresFileName)
#     scores_df = scores_df.loc[scores_df['userID'].isin(userIDList)] 
#     return scores_df[['userID', selectedFactor]]

def copyFeatureFiles(rootFolder, content, sensor, allFeatures_df):
    os.chdir(rootFolder + content + '/' +sensor)
    
    for file in glob.glob("*.csv"):
        print(file)
        if "FeatureVals_" in file and 'all_users_cutSignal' in file:
            df = pd.read_csv(file)
            strList = file.split()
            name = content + ' ' + sensor + ' ' + strList[1] + ' ' + strList[2] #+ ' ' + strList[3].replace('_cutSignal_df.csv', '')
            
            cols = df.columns
            for col in cols:
                if col not in columnNameList:
                    print(col)
                    df.rename({col:name}, axis=1, inplace =True)            
            # features_dfs_list.append(df)
            allFeatures_df = pd.merge(allFeatures_df, df[['uID', name]], on='uID', how='left')
    return allFeatures_df    
def copyFromAllContentFolders(rootFolder, allFeatures_df):
   
    for content  in contents:        
        print(content)
        for sensor in sensors:
            print(sensor)
            allFeatures_df = copyFeatureFiles(rootFolder, content, sensor, allFeatures_df)
    
    # result = pd.concat(features_dfs_list, axis=1, join="outer").drop_duplicates().reset_index(drop=True)
    allFeatures_df.to_csv(outputFile, index=False)

copyFromAllContentFolders(rootFolder, allFeatures_df)