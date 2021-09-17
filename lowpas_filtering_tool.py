# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 23:47:19 2021

@author: Andrej Košir
"""

from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import butter,filtfilt

from runExperiment_PictExamination import getSignalDf
from runExperiment_PictExamination import  getUsersSignalsOfOneContent
import os.path
from pathlib import Path

# @brief lowpass filtering of the signal using Butterworth filter
# @arg sig_x signal data
# @arg cut_f cat-off frequency in Hz
def lowpass_1D(sig_x, cut_f):
    
    # Settings
    #T = 5.0         # Sample Period
    fs = 30.0       # sample rate, Hz
    cutoff = cut_f      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = len(sig_x) # total number of samples
        
    # Filter 
    normal_cutoff = cutoff / nyq # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    sig_lowpass_x = filtfilt(b, a, sig_x)    
    
    return sig_lowpass_x  

def update(val):
    cut_f = s_cut_f.val
    sig_lowpass_x = lowpass_1D(sig_x, cut_f)
    l.set_ydata(sig_lowpass_x)
    fig.canvas.draw_idle()
    
def reset(event):
    s_cut_f.reset()  

#pupilsize
# sensors = {1:['empatica', {'ACC':['AccX'], 'EDA':['EDA'], 'HR':['HR']}], 2:['shimmer', {'':['GSR_Skin_Conductance_microSiemens']}],
#            3:['tobii', {'pupilDim_left_eye':['diameter']}],  4:['pupillabs', {'left_eye_2d':['diameter']}]}


# sensors = {1:['empatica', {'ACC':['AccX']}]}

# sensors = {1:['empatica', {'EDA':['EDA']}]}

# sensors = {1:['empatica', {'HR':['HR']}]}

# sensors = {2:['shimmer', {'':['GSR_Skin_Conductance_microSiemens']}]}

# sensors = {3:['tobii', {'pupilDim_left_eye':['diameter']}]}

sensors = {4:['pupillabs', {'left_eye_2d':['diameter']}]}
                         
rootFolder = "C:/Users/evinao/Dropbox (Lucami)/Lucami Team Folder/MMEM_Data/output/user"


selectedFactor = 'F1'
selectedContent = 'C1'

userSensorContentFileName =  "Data/userContentSensorDict.csv"

for sensorID, sensor in sensors.items():
    filteredUserList = getUsersSignalsOfOneContent(userSensorContentFileName, sensors, sensorID, selectedContent)
    userID = filteredUserList.iloc[0] 
    for sensorFileNameExt, signalList in sensor[1].items():
        for signalName in signalList:
            sig_x_df = getSignalDf(userID, sensorID, signalName, sensorFileNameExt)


            # Load data & select signal
            # sig_x_df = pd.read_csv('Data/test_df.csv')
            # signal_col = 'AccZ'
            sig_x = np.array(sig_x_df[signalName])
            sig_t = np.array(sig_x_df['timestamp_s'])
            
            
            
            # Interactive plot
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            plt.title(str(userID) + ' ' +  sensors[sensorID][0]+ ' ' +  signalName )
            s = sig_x #a0 * np.sin(2 * np.pi * f0 * t)
            l, = plt.plot(sig_t, sig_x, lw=1)
            #plt.plot(sig_t, sig_x, c='r')
            ax.margins(x=0)
            
            axcolor = 'lightgoldenrodyellow'
            ax_cut_f = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            
            
            #sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
            cut_f0 = 1
            cut_f_step = 0.01
            cut_f_min, cut_f_max = 0.001, 3.0
            s_cut_f = Slider(ax_cut_f, 'Cut_f', cut_f_min, cut_f_max, valinit=cut_f0, valstep=cut_f_step)
            #samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
            
            
            s_cut_f.on_changed(update)
            
            resetax = plt.axes([0.8, 0.025, 0.1, 0.04]) 
            button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
            
               
            button.on_clicked(reset)
            
            plt.show()