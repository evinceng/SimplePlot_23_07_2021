# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:45:44 2021

@author: evinao
"""
# import pandas as pd
# import heartpy as hp
# import neurokit2 as nk
# from pyEDA.main import *
# data = [['a', 1], ['b', 2], ['c',3], ['d', 4], ['e', 5]] 
# df = pd.DataFrame(data, columns=['signal', 'timestamp_s'])

# time_int = [1,2,3]
# print(df)
# print(df[time_int[0]:time_int[-1]])



# import numpy as np
# a = [10, 2, 8, 4, 5, 6, 7, 3, 9, 1]
# # print(np.percentile(a,10)) # gives the 95th percentile

# min_percent = 20
# max_percent = 80
# size = len(a)
# min_index = int(min_percent/size)
# max_index = int(max_percent/size)
# # print(a.mean())
# # size = df[col].shape[0]
# # outlier_min_percent = int(min_percent/100*size)
# # outlier_max_percent = int(max_percent/100*size)

# sorted_a = np.sort(a)
# min_val = np.mean(sorted_a[0:min_index])
# max_val = np.mean(sorted_a[max_index:size])
# print(min_val)
# print(max_val)



# lowList = [[2,1], [4,2], [8,4]]
# sum_mean = 0
# sum_std = 0
# for item in lowList:
#     sum_mean = sum_mean + item[0]
#     sum_std = sum_std + item[1]

# print(sum_mean/len(lowList))
# print(sum_std/len(lowList))

# # scatter plot
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# population = np.random.rand(100)
# Area = np.random.randint(100,600,100)
# continent =['North America','Europe', 'Asia', 'Australia']*25

# df = pd.DataFrame(dict(population=population, Area=Area, continent = continent))

# import seaborn as sns
# ax=sns.lmplot(x='population', y='Area', data=df, hue='continent', fit_reg=False)

# plt.axhline(y=10, color='r', linestyle='-')

# plt.show()

# # fig, ax = plt.subplots()

# # colors = {'North America':'red', 'Europe':'green', 'Asia':'blue', 'Australia':'yellow'}


# # ax.scatter(df['population'], df['Area'], c=df['continent'].map(colors))

# # plt.show()


# avarage of two dataframes
import pandas as pd

lowList = [[2,1], [4,2], [8,4], [10,6]]
highList = [[3,2], [5,3], [9,5]]

low_df = pd.DataFrame(lowList, columns=['a', 'b'])
high_df = pd.DataFrame(highList, columns=['a', 'c'])
new_df = pd.DataFrame()
new_df['a'] = (low_df['a'] + high_df['a'])/2.0
print(new_df)
new_df = new_df.dropna(axis='rows')
print(new_df)