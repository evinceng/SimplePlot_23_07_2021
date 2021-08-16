# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:57:13 2021

@author: evinao
"""
#resample the signals

import Tools.interpolate_funs as intFuns
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import glob, os
import pandas as pd
import numpy as np


def plotFigure(xIn, yIn, new_y, title):
    plt.figure(1)
    plt.title(title) 
    plt.plot(xIn, yIn, color = 'b', label='Orig')
    plt.plot(xIn, new_y, color = 'r', label='Interp.')
    plt.legend()
    plt.show()
    
def writeSplineTupleToFile(samplingTuple, filePath):
    np.savetxt(filePath + 't_vector_of_knots.txt', tck[0])
    np.savetxt(filePath + 'c_B_spline_coefficients.txt', tck[1])
    with open(filePath + 'k_degree_of_the_spline.txt', 'w') as f:
        f.write('%d' % tck[2])

def readSplineTuple(filePath):
    t_vector_of_knots = np.loadtxt(filePath + 't_vector_of_knots.txt')
    c_B_spline_coefficients = np.loadtxt(filePath + 'c_B_spline_coefficients.txt')
    with open(filePath + 'k_degree_of_the_spline.txt', 'r') as f:
        k_degree_of_the_spline = int(f.readline())
    samplingTuple = (t_vector_of_knots, c_B_spline_coefficients, k_degree_of_the_spline)
    print(samplingTuple)
    return samplingTuple

#change the root folder to your local folder
rootFolder = "C:/Users/evinao/Documents/GitHub/SimplePlot_23_07_2021/output/user"

uID = 28

#for one file interpolate and plot

fileNameList = ["_empatica_ACC", "_empatica_BVP", "_empatica_EDA","_empatica_HR",
                "_empatica_IBI", "_empatica_TEMP", "_pupillabs", "_pupillabs_gaze",
                "_shimmer_df", "_tobii_accelerometer", "_tobii_gazeDirection",
                "_tobii_gazePosition", "_tobii_gazePosition3D", "_tobii_gyroscope",
                "_tobii_pupilCenter", "_tobii_pupilDim"]

# to test one file
df = pd.read_csv(rootFolder + str(uID) + "/uID-" + str(uID) + fileNameList[3] + "_df_synced.csv")
signalName = 'HR'
print(df)

xIn = df['timestamp_s']#np.array([-0.1, 0.1, 1.1, 1.9, 6.1, 7.9, 8.1])
yIn = df[signalName] #np.array([2, 1.0, 1.8, 0.1, 6.5, 4.1, 2.3])
tMin, tMax = df['timestamp_s'].iloc[0], df['timestamp_s'].iloc[-1]
kIn = 3 #the order of each polynomial file in there, smoothness of the data 
TsIn = 0.2
fCode = 1
tck = intFuns.interpolateBS(xIn, yIn, tMin, tMax, kIn, TsIn, fCode)

tupleFilePath = rootFolder + str(uID) + "/uID" + str(uID)
writeSplineTupleToFile(tck, tupleFilePath)
# readSplineTuple(tupleFilePath)

# with open(rootFolder + str(uID) + "/uID" + str(uID) + "_samplingCoefficients.csv", "w") as the_file:
#     csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
#     writer = csv.writer(the_file, dialect="custom")
#     writer.writerow(tck)
        
# w = open(rootFolder + str(uID) + "/uID" + str(uID) + "_samplingCoefficients.json", "wb")#Open the file
# pickle.dump(tck, w)#Dump the dictionary bok, the first parameter into the file object w.
# w.close()
    
print(tck)
new_y = splev(xIn, tck)
plt.figure(1)
plt.title('interpolateBS ' + signalName + " plot") 
plt.plot(xIn, yIn, color = 'b', label='Orig')
plt.plot(xIn, new_y, color = 'r', label='Interp.')
plt.legend()
plt.show()

#==============================================================================
#for the files in folder 

# os.chdir(rootFolder + str(uID))
# print(glob.glob("*.csv"))

# plotIndex = 1

# os.chdir(rootFolder + str(uID))
# for file in glob.glob("*.csv"):
#     #don't know what to do withIBI_time
#     print(file)
#     if "IBI" in file:
#         print("IBI file don't know what to do")
#         continue
#     else:
#         df = pd.read_csv(file)
#         print(df['timestamp_s'])
#         for signalName in df.columns:
#             if signalName == "timestamp_s" or signalName == "Unnamed: 0":
#                 continue
#             else:
#                 xIn = df['timestamp_s']
#                 yIn = df[signalName] 
#                 tMin, tMax = df['timestamp_s'].iloc[0], df['timestamp_s'].iloc[-1]
#                 kIn = 3
#                 TsIn = 0.2
#                 fCode = 1
#                 tck = intFuns.interpolateBS(xIn, yIn, tMin, tMax, kIn, TsIn, fCode)
                
#                 new_y = splev(xIn, tck)
#                 plt.figure(plotIndex)
#                 plt.title('interpolateBS '  + file +  ' ' + signalName + ' plot') 
#                 plt.plot(xIn, yIn, color = 'b', label='Orig')
#                 plt.plot(xIn, new_y, color = 'r', label='Interp.')
#                 plt.legend()
#                 plt.show()
#                 plotIndex += 1
#                 print(signalName)