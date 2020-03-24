# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:22:23 2018

@author: KOGANTI
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import pyflux as pf

my_data = np.genfromtxt('pollution.csv',delimiter=",")[1:,1:]
def nan_helper(y):# from stack overflow
        return np.isnan(y), lambda z: z.nonzero()[0]
def fill_na(data):
    nans, x= nan_helper(data)
    data[nans]= np.interp(x(nans), x(~nans), data[~nans])
    return data
for i in range(4,8):
    modified_pm = fill_na(my_data[:,i])
    my_data[:,i] = np.reshape(modified_pm,modified_pm.size)


stop = 15000
start = 12000
interval = 20
nmbr_predictions = 2
allpredicted_values = np.array([0,0,0,0]).reshape(1,4)
original_values = np.array([0,0,0,0]).reshape(1,4)

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    VAR_data = pd.DataFrame({'pm25':my_data[:,4][:i],'dewp':my_data[:,5][:i],'temp':my_data[:,6][:i],'pres':my_data[:,7][:i]})
    VAR_data = VAR_data[['pm25','dewp','temp','pres']]
    model = pf.VAR(data=VAR_data,lags=4,integ=1)
    x = model.fit()
    print(model.predict_is(nmbr_predictions).values)
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[i-3,4:8].reshape(1,4),model.predict_is(nmbr_predictions).values,axis=0),axis=0)[1:,:],axis=0)
    original_values = np.append(original_values,my_data[i-2:i,4:8],axis=0)



allpredicted_values = allpredicted_values[1:,]
original_values = original_values[1:,]
np.save("predicted_values_short_term_for_error",allpredicted_values)
np.save("original_values_short_term_for_error",original_values)



