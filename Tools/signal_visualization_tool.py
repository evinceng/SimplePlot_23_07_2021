# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:31:50 2021

@author: evinao
"""

import matplotlib.pyplot as plt
import pandas as pd
import Utils
import numpy as np

def generateSignalSubplots_spec_amp(users, ax, axIndex, lowOrHighFactorUserIDs, sensorID, signalName, feature_code):
    subPlotId = 0    
    for userID in lowOrHighFactorUserIDs:
        # if (key % 2) == 0:
        #     color = '#1f77b4'
        # else:
        #     color =  'm'
        
        # x, x2 = np.array_split(users[userID][sensorID][signalName][feature_code][0],2)
        # y, y2 = np.array_split(users[userID][sensorID][signalName][feature_code][1],2)
        
        x = users[userID][sensorID][signalName][feature_code][0]
        y = users[userID][sensorID][signalName][feature_code][1]
        nonzero_x_indices = np.where(x > 0)
                
        ax[axIndex][subPlotId].plot(x[nonzero_x_indices], np.abs(y[nonzero_x_indices])) #, c=color
        # ax[axIndex][subPlotId].set_xlabel('sig_f')
        # ax[axIndex][subPlotId].set_ylabel('spec_amp')
        ax[axIndex][subPlotId].grid(True)
        # ax[axIndex][subPlotId].set_yscale('log')
        subPlotId = subPlotId + 1
        
        
# def generateSignalSubplots_Gram_AF(users, ax, axIndex, lowOrHighFactorUserIDs, sensorID, signalName, feature_code):
#     subPlotId = 0    
#     for userID in lowOrHighFactorUserIDs:
#         # if (key % 2) == 0:
#         #     color = '#1f77b4'
#         # else:
#         #     color =  'm'
        
#         # x, x2 = np.array_split(users[userID][sensorID][signalName][feature_code][0],2)
#         # y, y2 = np.array_split(users[userID][sensorID][signalName][feature_code][1],2)
        
#         x = users[userID][sensorID][signalName][feature_code][0]
#         y = users[userID][sensorID][signalName][feature_code][1]
#         nonzero_x_indices = np.where(x > 0)
                
#         ax[axIndex][subPlotId].plot(x[nonzero_x_indices], np.abs(y[nonzero_x_indices])) #, c=color
#         # ax[axIndex][subPlotId].set_xlabel('sig_f')
#         # ax[axIndex][subPlotId].set_ylabel('spec_amp')
#         ax[axIndex][subPlotId].grid(True)
#         # ax[axIndex][subPlotId].set_yscale('log')
#         subPlotId = subPlotId + 1
        
def generateSubPlotsofOneSignalOfMultipleUsers(usersDict, sensors, selectedContent, selectedFactor, outputFolderName,
                                                    lowFactorUserIDs,
                                                    highFactorUserIDs,
                                                    sensorID,
                                                    signalName,
                                                    feature_code):
    
    subPlotCount = max(len(lowFactorUserIDs), len(highFactorUserIDs))
    #take min and max 
    fig, ax = plt.subplots(2, subPlotCount, sharey = True, sharex=True) #, sharey=True
    fig.suptitle(sensors[sensorID][0] +' ' + signalName + ' ' + selectedContent + ' sig_f vs sig_comp')
    #labels
    # ax[0][0].set_xlabel('sig_f')
    # ax[0][0].set_ylabel('spec_amp')
    
    ax[0][0].set_title('Low score ' + selectedFactor)
    ax[1][0].set_title('High score ' + selectedFactor)
    
    # if feature_code == 'spec_amp':
    # plot the first row of subplots
    generateSignalSubplots_spec_amp(usersDict, ax, 0, lowFactorUserIDs, sensorID, signalName, feature_code)    
    # plot the second row of subplots
    generateSignalSubplots_spec_amp(usersDict, ax, 1, highFactorUserIDs, sensorID, signalName, feature_code)
    # elif feature_code == 'Gram_AF':
    #     # plot the first row of subplots
    #     generateSignalSubplots_Gram_AF(usersDict, ax, 0, lowFactorUserIDs, sensorID, signalName, feature_code) 
        
    # now = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    if '/' in signalName:
        signalName = signalName.replace('/','-')
    elif '.' in signalName:
        signalName = signalName.replace('.','-')
    saveFileFig = outputFolderName + selectedContent + '/' + sensors[sensorID][0] + '/' + selectedFactor + '_' + sensors[sensorID][0]  + '_' + signalName + '_' + feature_code + ' sig_f vs sig_comp'#+ '_' + now
    plt.savefig(saveFileFig +'.jpg')
    Utils.writePickleFile(fig, saveFileFig)
    # pickle.dump(fig, open(saveFileFig +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
    # plt.close(fig)
    
    fig.tight_layout()
    plt.show()