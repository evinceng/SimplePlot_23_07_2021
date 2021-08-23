# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:22:02 2021

@author: evinao
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from scipy import stats

df = pd.DataFrame([np.nan, np.nan, 3.5,4,5,2, 2015,2050, 2010, 3,4.2,0,5,6,1,2.4,3,np.nan,np.nan,np.nan,np.nan,np.nan,2,3,4,3,4,np.nan], columns=['Signal'])

sdf = pd.DataFrame({'Signal' : [3.5,4,5,2, 2015,2050, 2010, 3,4.2,0,5,6,1,2.4,32,3,4,3,4], 'S2': [1.5,4,5,2, 2015,2050, 2010, 3,4.2,0,5,6,1,2.4,32,3,4,3,4]})

def getPrevAndNextHealthyIndexes2(df, outliers, loc):
    size = df.shape[0]
    prev_index = loc - 1
    next_index = loc + 1
    
    if prev_index < 0:
        prev_index = -1 
        print("There were no values till the start of the df, take only next_index")
    else:    
        while prev_index in outliers:
            prev_index = prev_index - 1
            if prev_index <= 0:
                print("There were no values till the start of the df, take only next_index")
                prev_index = -1
                break
    if next_index >= df.shape[0]:
        next_index = -1
        print("There were no values till the end of the, going to the start of the df")
    else:
        while next_index in outliers:
            next_index = next_index + 1
            if next_index >= size - 1:
                print("There were no values till the end of the, going to the start of the df")
                next_index = -1
                break
    return prev_index, next_index


def correctOutliers(df):
    
    #reset the index incase it is a df from filtered df such as pupil labs right and left eye dfs
    df  = df.reset_index(drop=True)
    #remove outliers    
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue
        # Identify index locations above cutoff
        min_percent = 20
        max_percent = 80
        
        print(df[col].mean())
        size = df[col].shape[0]
        outlier_min_percent = int(min_percent/100*size)
        outlier_max_percent = int(max_percent/100*size)
        
        sorted_df = df[col].sort_values()
        sorted_df = sorted_df.iloc[outlier_min_percent:outlier_max_percent]
        mean = sorted_df.mean()
        std = sorted_df.std()
        print(sorted_df)
        
        
        # or interquartile
        # Q1 = df[col].quantile(0.10)
        # Q9 = df[col].quantile(0.90)
        # IQR = Q9 - Q1
        
        outliers = df[(df[col] < mean - std*3) | (df[col] > mean + std*3)].index
        # outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q9 + 1.5 * IQR))].index
        print(outliers)
        # Browse through outliers and average according to index location
        for loc in outliers:
            # Get index location
            # print(loc)
            prev_healthy_index, next_healthy_index = getPrevAndNextHealthyIndexes2(df[col], outliers, loc)
            new_val = -10000
            if prev_healthy_index == -1 and next_healthy_index == -1:
                print("!!!!!!!!!!Couldn't find healty values. Error !!!!!!!!!!!!!!")
            elif prev_healthy_index == -1:
                new_val = df[col][next_healthy_index]
            elif next_healthy_index == -1:
                new_val = df[col][prev_healthy_index]
            else:
                val1 = df[col][prev_healthy_index]
                val2 = df[col][next_healthy_index]
                new_val = (val1+val2)/2.0
            # meanVal = (df[col][prev_healthy_index] + df[col][next_healthy_index]) / 2
            df[col][loc] = new_val
            # print(df)
        print(df[col].mean())
    return df
            
# print(df.mean())
# df = correctOutliers(df)
# print(df.mean())     


def correctOutliers_fromresample(df):
    
    # #reset the index incase it is a df from filtered df such as pupil labs right and left eye dfs
    # df  = df.reset_index(drop=True)
    
    min_percent = 10
    max_percent = 90
    #remove outliers    
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue
        # Identify index locations above cutoff
        # Identify index locations above cutoff
        
        print("mean before is " + col)
        print(df[col].mean())
        size = df[col].shape[0]
        outlier_min_percent = int(min_percent/100*size)
        outlier_max_percent = int(max_percent/100*size)
        
        sorted_df = df[col].sort_values()
        sorted_df = sorted_df.iloc[outlier_min_percent:outlier_max_percent]
        mean = sorted_df.mean()
        std = sorted_df.std()
        # print(sorted_df)
        
        
        # or interquartile
        # Q1 = df[col].quantile(0.10)
        # Q9 = df[col].quantile(0.90)
        # IQR = Q9 - Q1
        
        outliers = df[(df[col] < mean - std*3) | (df[col] > mean + std*3)].index
        
        # print(outliers)
        # Browse through outliers and average according to index location
        for loc in outliers:
            # Get index location
            # print(loc)
            prev_healthy_index, next_healthy_index = getPrevAndNextHealthyIndexes2(df[col], outliers, loc)
            new_val = -10000
            if prev_healthy_index == -1 and next_healthy_index == -1:
                print("!!!!!!!!!!Couldn't find healty values. Error !!!!!!!!!!!!!!")
            elif prev_healthy_index == -1:
                new_val = df[col][next_healthy_index]
            elif next_healthy_index == -1:
                new_val = df[col][prev_healthy_index]
            else:
                val1 = df[col][prev_healthy_index]
                val2 = df[col][next_healthy_index]
                new_val = (val1+val2)/2.0
            # meanVal = (df[col][prev_healthy_index] + df[col][next_healthy_index]) / 2
            df[col][loc] = new_val
            # print(df)
        print("Mean after is " + col)
        print(df[col].mean())
    return df

# does ot correct the 0 at the start, the end no problem
# df.replace(to_replace=0, value=np.nan, inplace=True)
# df.interpolate(inplace=True)


#doen't work
# df.where(np.isnan(df.values), other=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2)

# df.where(df.notnull(), other=(df.fillna(method='ffill')+df.fillna(method='bfill'))/2)

df['Signal'].where((df['Signal'] <= 2000)  & (df['Signal'] > 2), np.nan, inplace=True)

print(df)
# value sin the middle
df.where(df.notnull(), other=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2, inplace=True)
# print(df)
df.where(df.notnull(), other=(df.fillna(method='ffill')), inplace=True)
# print(df)
df.where(df.notnull(), other=(df.fillna(method='bfill')), inplace=True)
# df = df.where(df.values >= 2000, other=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2)

print(df)

