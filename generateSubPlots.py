# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:52:30 2021

@author: Evin Aslan Oguz
"""

import matplotlib.pyplot as plt


def generateSubPlotsWithAdVideoTimesOfOneUser(uID, out_times_lst, yLabelDfDict, saveFigFilePath):
    
    subPlotCount = len(yLabelDfDict.keys())
    fig, ax = plt.subplots(subPlotCount, 1)
    
    for key in yLabelDfDict.keys():
        if (key % 2) == 0:
            color = '#1f77b4'
        else:
           color =  'm'
          
        signalName = yLabelDfDict[key]['signalName']
        df = yLabelDfDict[key]['df']
        print(df.head())
        ax[key].plot(df['timestamp_s'], df[signalName], c=color)
        ax[key].set_xlabel('t [s]')
        ax[key].set_ylabel(yLabelDfDict[key]['sensorName'] +': ' + signalName)
        ax[key].grid(True)
        
    #draw vertical lines for each of the video-ad 
    for axis in ax:
        for ad in out_times_lst:
            axis.axvline(x=ad[0], c='g')
            axis.axvline(x=ad[1], c='r')
            axis.axvline(x=ad[2], c='r')
            axis.axvline(x=ad[3], c='g')
                        
    plt.savefig(saveFigFilePath)
    plt.show()


    