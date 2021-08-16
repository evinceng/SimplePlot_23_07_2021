# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:43:35 2021

@author: Evin Aslan Oguz
"""
import pandas as pd

fileName = "D:/LivingLabMeasurements/user18/Tobii/recordings/lww5e2a/segments/1/livedata.json"

df = pd.read_json(fileName, lines=True, orient='records')

print(df) 

df.to_csv('output/user18.csv')
