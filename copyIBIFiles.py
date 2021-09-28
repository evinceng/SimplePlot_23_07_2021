# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:04:18 2021

@author: evinao
"""

import glob, os
import os.path
from pathlib import Path
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

from shutil import copyfile


rootFolder = "C:/Users/evinao/Dropbox (Lucami)/LivingLab MMEM data/output/user"

copyFolder = "C:/Users/evinao/Documents/Paper2Data/IBIFiles/"


def copyIBIFile(rootFolder , uID):
    os.chdir(rootFolder + str(uID))
    # print(glob.glob("*.csv"))
    Path(copyFolder).mkdir(parents=True, exist_ok=True) # will not change directory if exists
    
    for file in glob.glob("*.csv"):
        #don't know what to do withIBI_time
        print(file)
        if "IBI" in file:
            print("IBI file don't know what to do")
            copyfile(rootFolder + str(uID) +'/'+file, copyFolder+file)
            break

            
       
           
def copyFromAllUsersFolders(rootFolder):

    uIDlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,
               37,38,39,46,47,48,49,50,51,52,53,54,55,56,57,58,60]
    
    for userid  in uIDlist:
        print(userid)
        copyIBIFile(rootFolder, userid)
        

copyFromAllUsersFolders(rootFolder)