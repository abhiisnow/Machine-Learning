# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:54:45 2018

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
from io import BytesIO


def nan_helper(y):# from stack overflow
        return np.isnan(y), lambda z: z.nonzero()[0]
def fill_na(data):
    nans, x= nan_helper(data)
    data[nans]= np.interp(x(nans), x(~nans), data[~nans])
    return data

plt.plot(my_data[:,4][:12000])
plt.plot(my_data[:,5][:12000])
plt.plot(my_data[:,6][:12000])
plt.plot(my_data[:,7][:12000])
plt.plot(my_data[:,9][:12000])

#loading data and attanching day
my_data = np.genfromtxt('pollution2.csv',delimiter=",")[1:,1:]
modified_pm = fill_na(my_data[:,4])
my_data[:,4] = modified_pm


"""day_week = np.zeros(my_data.shape[0])
for i in range(0,day_week.size):
    date = datetime.date(int(my_data[i,0]),int(my_data[i,1]),int(my_data[i,2]))
    day_week[i] = date.weekday()

my_data = my_data[:,0:]
my_data = np.column_stack((day_week,my_data))

#converting data into daily data
mnth_values_date_year = []
for year in range(2010,2015):
    for month in range(1,13):
        for date in range(1,31):
            avg_values = np.mean(my_data[np.intersect1d(np.intersect1d(np.where(my_data[:,1]==year)[0],np.where(my_data[:,2]==month)[0]),np.where(my_data[:,3]==date)[0]),:][:,5],axis=0)
            mnth_values_date_year.append(np.append([year,month,date],avg_values))
my_datadaily = np.array(mnth_values_date_year)
my_datadaily = my_datadaily[~np.isnan(my_datadaily).any(axis=1)]"""

# test for unit root(non stationary process)
#more negative than all signinficant levels so the series is stationary process
"""plt.plot(my_datadaily[:,3])
adf_result = adfuller(my_datadaily[:,3])
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
	print('\t%s: %.3f' % (key, value))
#plots
#looks like AR2 an MA2 model PACF cuts at lag2 and ACF cuts at lag2
#plot ACF
plot_acf(my_datadaily[:,3],lags=10)

#plot PACF
plot_pacf(my_datadaily[:,3],lags=10)

#fitting arima model
model = ARIMA(my_datadaily[:,3][:-7],order = (2,0,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#simple forecast for next day
forecast = model_fit.forecast()[0]
forecast = model_fit.forecast(steps=10)[0]"""


"""#use dynamic linear regresion to model unseen factors
#using full data
plt.plot(np.diff(my_data[:,4]))
adf_result = adfuller(np.diff(my_data[:,4]))
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
	print('\t%s: %.3f' % (key, value))
#plots
#looks like AR2 an MA2 model PACF cuts at lag2 and ACF cuts at lag2
#plot ACF
plot_acf(np.diff(my_data[:,4]),lags=20)

#plot PACF
plot_pacf(np.diff(my_data[:,4]),lags=10)

#fitting arima model
model = ARIMA(my_data[:,4],order = (1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#simple forecast for next day
forecast = model_fit.forecast()[0]
forecast = model_fit.forecast(steps=10)[0]
forecast[0] = my_data[:,5][-1]+forecast[0]
forecast = np.cumsum(forecast)"""

#ARIMA model in pyflux
data = pd.DataFrame({'pm25data':my_data[:,4][:12100]})
model = pf.ARIMA(data=data, ar=1, ma=1, integ=0, target='pm25data', family=pf.Normal())
x = model.fit("MLE")
model.plot_fit()
model.plot_predict(300)

plt.plot(my_data[12000:12100,4])
stop = 15000
start = 14902
interval = 10
nmbr_predictions = 2
allpredicted_values = np.array([])
original_values = np.array([])

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    data = pd.DataFrame({'pm25data':my_data[:,4][:i]})
    model = pf.ARIMA(data=data, ar=1, ma=1, integ=1, target='pm25data', family=pf.Normal())
    x = model.fit("MLE")
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[:,4][i-3],model.predict_is(nmbr_predictions).values))[1:])
    original_values = np.append(original_values,my_data[:,4][i-2:i])
#sampleforcasting graph
#plootign the forecst part
x_values = []
end_value = 0

for i in range(0,stop-start,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+14900
plt.plot(np.arange(14900,15000),my_data[:,4][14900:15000],'r')
plt.plot(x_formattedvalue,allpredicted_values,'bo')
plt.plot(14967,165,'bo')
plt.text(14970,160,"forecasted values")
plt.plot(14967,155,'ro')
plt.text(14970,150,"true values")
#long term plotting using ARIMA
data = pd.DataFrame({'pm25data':my_data[:,4][:15000]})
model = pf.ARIMA(data=data, ar=1, ma=1, integ=1, target='pm25data', family=pf.Normal())
long_term = model.predict_is(100).values
long_term = np.insert(long_term,0,my_data[:,4][:15000][-100])
long_term = np.cumsum(long_term)[1:]
allpredicted_values = long_term
original_values = my_data[:,4][14900:15000]
plt.plot(np.arange(14900,15000),long_term)
plt.plot(np.arange(14900,15000),my_data[:,4][14900:15000])
plt.plot(14970,175,'bo')
plt.text(14972,175,"forecasted values")
plt.plot(14970,165,'go')
plt.text(14972,165,"true values")
#errors
error_arima1 =  mean_absolute_error(original_values,allpredicted_values)
error_arima2 = r2_score(original_values,allpredicted_values)


#Vector auto regression(VAR)
my_data = np.genfromtxt('pollution2.csv',delimiter=",")[1:,1:]
for i in range(4,12):
    modified_pm = fill_na(my_data[:,i])
    my_data[:,i] = np.reshape(modified_pm,modified_pm.size)
VAR_data = VAR_data[['pm25','dewp','temp','pres','cbwd','lws','ls','lr']]
VAR_data = pd.DataFrame({'pm25':my_data[:,4][:15000],'dewp':my_data[:,5][:15000],'temp':my_data[:,6][:15000],'pres':my_data[:,7][:15000],'cbwd':my_data[:,8][:15000],'lws':my_data[:,9][:15000],'ls':my_data[:,10][:15000],'lr':my_data[:,11][:15000]})
VAR_data = VAR_data[['pm25','dewp','temp','pres']]
model = pf.VAR(data=VAR_data,lags=1,integ=1)
x = model.fit()

forecast_values = np.cumsum(np.append(my_data[14900,4:8].reshape(1,4),model.predict_is(100).values,axis=0),axis=0)[1:]
allpredicted_values = forecast_values
original_values = my_data[14900:15000,4]
x_values = []
end_value = 0
for i in range(0,stop-start+2,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+14900
plt.plot(np.arange(14900,15000),my_data[14900:15000,4],'r')
plt.plot(np.arange(14900,15000),forecast_values[:,0],'bo')
plt.plot(14970,145,'bo')
plt.text(14972,145,"true values",clip_on=True)
plt.plot(14970,157,'ro')
plt.text(14972,157,"forecasted values",clip_on=True)
stop = 15000
start = 14902
interval = 20
nmbr_predictions = 2
allpredicted_values = np.array([0,0,0,0]).reshape(1,4)
original_values = np.array([0,0,0,0]).reshape(1,4)

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    
    VAR_data = pd.DataFrame({'pm25':my_data[:,4][14700:i],'dewp':my_data[:,5][14700:i],'temp':my_data[:,6][14700:i],'pres':my_data[:,7][14700:i],'cbwd':my_data[:,8][14700:i],'lws':my_data[:,9][14700:i],'ls':my_data[:,10][14700:i],'lr':my_data[:,11][14700:i]})
    VAR_data = VAR_data[['pm25','dewp','temp','pres']]
    model = pf.VAR(data=VAR_data,lags=1,integ=1)
    x = model.fit("MLE")
    print(model.predict_is(nmbr_predictions).values)
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[i-3,4:8].reshape(1,4),model.predict_is(nmbr_predictions).values,axis=0),axis=0)[1:,:],axis=0)
    original_values = np.append(original_values,my_data[i-2:i,4:8],axis=0)
allpredicted_values = allpredicted_values[1:,]
original_values = original_values[1:,]
error_arima1 =  mean_absolute_error(original_values,allpredicted_values[:,0])
error_arima2 = r2_score(original_values,allpredicted_values[:,0])
#sampleforcasting graph
#plotting the forecst part

x_values = []
end_value = 0

for i in range(0,stop-start+2,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+14900
plt.plot(np.arange(14900,15000),my_data[14900:15000,4],'r')
plt.plot(x_formattedvalue,allpredicted_values[:,0],'bo')
plt.plot(14970,145,'bo')
plt.text(14972,145,"true values",clip_on=True)
plt.plot(14970,157,'ro')
plt.text(14972,157,"forecasted values",clip_on=True)


plt.plot(np.diff(my_data[:,4][start:stop]),'r')
plt.plot(x_values,allpredicted_values[:,0],'bo')
plt.plot(np.diff(my_data[:,5][start:stop]),'r')
plt.plot(x_values,allpredicted_values[:,1],'bo')
plt.plot(np.diff(my_data[:,6][start:stop]),'r')
plt.plot(x_values,allpredicted_values[:,2],'bo')
plt.plot(np.diff(my_data[:,7][start:stop]),'r')
plt.plot(x_values,allpredicted_values[:,3],'bo')


#vector

stop = 15000
start = 14902
interval = 20
nmbr_predictions = 5
allpredicted_values = np.array([0,0,0,0]).reshape(1,4)
original_values = np.array([0,0,0,0]).reshape(1,4)

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    
    VAR_data = pd.DataFrame({'pm25':my_data[:,4][14700:i],'dewp':my_data[:,5][14700:i],'temp':my_data[:,6][14700:i],'pres':my_data[:,7][14700:i],'cbwd':my_data[:,8][14700:i],'lws':my_data[:,9][14700:i],'ls':my_data[:,10][14700:i],'lr':my_data[:,11][14700:i]})
    VAR_data = VAR_data[['pm25','dewp','temp','pres']]
    model = pf.VAR(data=VAR_data,lags=1,integ=1)
    x = model.fit("MLE")
    print(model.predict_is(nmbr_predictions).values)
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[i-3,4:8].reshape(1,4),model.predict_is(nmbr_predictions).values,axis=0),axis=0)[1:,:],axis=0)
    original_values = np.append(original_values,my_data[i-5:i,4:8],axis=0)
allpredicted_values = allpredicted_values[1:,]
original_values = original_values[1:,]
#sampleforcasting graph
#plotting the forecst part

x_values = []
end_value = 0

for i in range(0,stop-start+2,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+14900
plt.plot(np.arange(14900,15000),my_data[14900:15000,4],'r')
plt.plot(x_formattedvalue,allpredicted_values[:,0],'bo')
plt.plot(14970,145,'bo')
plt.text(14972,145,"true values",clip_on=True)
plt.plot(14970,157,'ro')
plt.text(14972,157,"forecasted values",clip_on=True)

















#errors
error_var1 =  mean_absolute_error(original_values[:,0],allpredicted_values[:,0])
error_var2 = r2_score(original_values[:,0],allpredicted_values[:,0])


#GARCH
data = pd.DataFrame({'pm25data':np.diff(my_data[:,4][:12000])})
model = pf.GARCH(data=data, p=1, q=1,  target='pm25data')
x = model.fit("MLE")
model.plot_fit()

stop = 15000
start = 14900
interval = 20
nmbr_predictions = 2
allpredicted_values = np.array([])
original_values = np.array([])

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    data = pd.DataFrame({'pm25data':np.diff(my_data[:,4][:i])})
    model = pf.GARCH(data=data, p=0, q=1,  target='pm25data')
    x = model.fit("MLE")
    print(model.predict_is(nmbr_predictions).values)
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[:,4][i-3],model.predict_is(nmbr_predictions).values))[1:])
    original_values = np.append(original_values,my_data[:,4][i-2:i])
#sampleforcasting graph
#plootign the forecst part
x_values = []
end_value = 0
for i in range(0,stop-start,interval): 
    x_values.append(i)
    x_values.append(i+1)
x_values = np.array(x_values)
plt.plot(my_data[:,4][14900:15000],'r')
plt.plot(x_values,allpredicted_values,'bo')
#long term plotting using ARIMA
data = pd.DataFrame({'pm25data':my_data[:,4][:15000]})
model = pf.GARCH(data=data, p=1, q=1,  target='pm25data')
model.plot_predict_is(50)


#ARIMA model in pyflux
data = pd.DataFrame({'pm25data':my_data[:,4][:12000]})
model = pf.ARIMA(data=data, ar=1, ma=1, integ=1, target='pm25data', family=pf.Normal())
x = model.fit("MLE")
model.plot_fit()

stop = 15000
start = 14900
interval = 10
nmbr_predictions = 5
allpredicted_values = np.array([])
original_values = np.array([])

for i in range(start,stop,interval):
    i = i+nmbr_predictions
    data = pd.DataFrame({'pm25data':my_data[:,4][:i]})
    model = pf.ARIMA(data=data, ar=1, ma=1, integ=1, target='pm25data', family=pf.Normal())
    x = model.fit("MLE")
    allpredicted_values = np.append(allpredicted_values,np.cumsum(np.append(my_data[:,4][i-3],model.predict_is(nmbr_predictions).values))[1:])
    original_values = np.append(original_values,my_data[:,4][i-2:i])
#sampleforcasting graph
#plootign the forecst part
x_values = []
end_value = 0

for i in range(0,stop-start,interval):
    for j in range(0,nmbr_predictions):
        k = i+j
        x_values.append(k)
        
x_values = np.array(x_values)
x_formattedvalue = x_values+14900
plt.plot(np.arange(14900,15000),my_data[:,4][14900:15000],'r')
plt.plot(x_formattedvalue,allpredicted_values,'bo')
plt.plot(14967,165,'bo')
plt.text(14970,160,"forecasted values")
plt.plot(14967,155,'ro')
plt.text(14970,150,"true values")
#long term plotting using ARIMA
data = pd.DataFrame({'pm25data':my_data[:,4][:15000]})
model = pf.ARIMA(data=data, ar=1, ma=1, integ=1, target='pm25data', family=pf.Normal())
model.plot_predict_is(50)
#errors
error_arima1 =  mean_absolute_error(original_values,allpredicted_values)
error_arima2 = r2_score(y_true,y_pred)




