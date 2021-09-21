# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:45:44 2021

@author: evinao
"""
import pandas as pd
import heartpy as hp
import neurokit2 as nk
from pyEDA.main import *
# data = [['a', 1], ['b', 2], ['c',3], ['d', 4], ['e', 5]] 
# df = pd.DataFrame(data, columns=['signal', 'timestamp_s'])

# time_int = [1,2,3]
# print(df)
# print(df[time_int[0]:time_int[-1]])



import numpy as np
a = [10, 2, 8, 4, 5, 6, 7, 3, 9, 1]
# print(np.percentile(a,10)) # gives the 95th percentile

min_percent = 20
max_percent = 80
size = len(a)
min_index = int(min_percent/size)
max_index = int(max_percent/size)
# print(a.mean())
# size = df[col].shape[0]
# outlier_min_percent = int(min_percent/100*size)
# outlier_max_percent = int(max_percent/100*size)

sorted_a = np.sort(a)
min_val = np.mean(sorted_a[0:min_index])
max_val = np.mean(sorted_a[max_index:size])
print(min_val)
print(max_val)