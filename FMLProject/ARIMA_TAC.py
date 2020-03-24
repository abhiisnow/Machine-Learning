# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:47:18 2018

@author: z003v0nd
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from pandas import datetime
import pyflux as pf
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score


tac_data = []
tac_dates = []
with open('tacHKG_EUR.csv') as tac_csv:
    reader = csv.DictReader(tac_csv)
    for row in reader:
        tac_dates.append(row['Date'])
        tac_data.append(row['TAC'])
tac_data = np.array(tac_data).astype('float64')*10

plt.plot(tac_data[:1000])
plt.plot(np.diff(tac_data))
adf_result = adfuller(np.diff(tac_data))
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
	print('\t%s: %.3f' % (key, value))
plot_acf(np.diff(tac_data),lags=10)
plot_pacf(np.diff(tac_data),lags=10)
data = pd.DataFrame({'tac_data':tac_data[:1000]})
model = pf.ARIMA(data=data, ar=2, ma=1, integ=1, target='tac_data', family=pf.Normal())
x = model.fit("MLE")
model.plot_fit()
long_term = model.predict_is(58).values
long_term = np.insert(long_term,0,tac_data[-58])
long_term = np.cumsum(long_term)[1:]
allpredicted_values = long_term 
original_values = tac_data[1000:1058]
error_arima1 =  mean_absolute_error(original_values,allpredicted_values[:np.size(original_values)])
error_arima2 = r2_score(original_values,allpredicted_values[:np.size(original_values)])


plt.plot(np.arange(1000,1058),long_term)
plt.plot(np.arange(1000,1058),tac_data[1000:1058])
plt.plot(1010,29,'bo')
plt.text(1012,29,"forecasted values")
plt.plot(1010,28.4,'go')
plt.text(1012,28.4,"true values")
#long-term
stop = 1058
start = 1005
interval = 5
nmbr_predictions = 5
allpredicted_values = np.array([])
original_values = np.array([])

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    data = pd.DataFrame({'tac_data':tac_data[:i]})
    model = pf.ARIMA(data=data, ar=2, ma=1, integ=1, target='tac_data', family=pf.Normal())
    x = model.fit("MLE")
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(tac_data[i-(nmbr_predictions+1)],model.predict_is(nmbr_predictions).values))[1:])
    original_values = np.append(original_values,tac_data[i-nmbr_predictions:i])
    
#sampleforcasting graph
#plootign the forecst part
x_values = []
end_value = 0

for i in range(0,stop-start,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+1000
plt.plot(np.arange(1000,1058),tac_data[1000:1058],'r')
plt.plot(x_formattedvalue,allpredicted_values,'bo')
plt.plot(1010,29.10,'bo')
plt.text(1012,29.5,"true values")
plt.plot(1010,29.6,'ro')
plt.text(1012,29.1,"forecasted values")
error_arima1 =  mean_absolute_error(original_values,allpredicted_values[:np.size(original_values)])
error_arima2 = r2_score(original_values,allpredicted_values[:np.size(original_values)])


#short term forecasting for ARIMA
stop = 1058
start = 1002
interval = 5
nmbr_predictions = 2
allpredicted_values = np.array([])
original_values = np.array([])

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    data = pd.DataFrame({'tac_data':tac_data[:i]})
    model = pf.ARIMA(data=data, ar=2, ma=1, integ=1, target='tac_data', family=pf.Normal())
    x = model.fit("MLE")
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(tac_data[i-(nmbr_predictions+1)],model.predict_is(nmbr_predictions).values))[1:])
    original_values = np.append(original_values,tac_data[i-nmbr_predictions:i])
    
#sampleforcasting graph
#plootign the forecst part
x_values = []
end_value = 0

for i in range(0,stop-start,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+1000
plt.plot(np.arange(1000,1058),tac_data[1000:1058],'r')
plt.plot(x_formattedvalue,allpredicted_values,'bo')
plt.plot(1010,29.10,'bo')
plt.text(1012,29.5,"true values")
plt.plot(1010,29.6,'ro')
plt.text(1012,29.1,"forecasted values")
error_arima1 =  mean_absolute_error(original_values,allpredicted_values[:np.size(original_values)])
error_arima2 = r2_score(original_values,allpredicted_values[:np.size(original_values)])
