# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:35:55 2021

@author: MRITYUNJAY
Data Processing
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
%matplotlib inline

df=pd.read_excel(r'Combined Sales.xlsx',parse_dates=['Month'], index_col=1)
df.drop(df.columns[[0, 1, 2, 3, 5]], axis=1,inplace=True)
df.rename(columns={'Fees Total':'sales'}, inplace=True)
df
sales_data=df['sales'].resample('MS').sum()
df.groupby('Month').agg(sum)['sales']
sales_data.head()


#Plot
fig, ax = plt.subplots(figsize = (12, 8))
fig = plt.plot(sales_data, color = "blue")
ax.set(xlabel="Month",
       ylabel="Sales",
       title="Sales for individual months")
date_form = DateFormatter("%b-%y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

#Testing whether data is stationary - 2 Methods
#For the data to be stationary the mean and std must be constant which is
#clearly not the case here
#Rolling Statistics
def RollingStatistics(Sales):
    rolmean = Sales.rolling(window = 3).mean()
    rolstd = Sales.rolling(window = 3).std()
    print("Mean & Standard Deviation: ")
    print(rolmean, rolstd, sep="\n")
    orig = plt.plot(Sales, color = 'blue', label='Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    dtFmt = DateFormatter('%b-%y')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.show(block = False)
    

RollingStatistics(sales_data)
#Taking sqrt for larger Data Handling
sales_data_sqrt= np.sqrt(sales_data)
RollingStatistics(sales_data_sqrt)


#Since we have a data for only one year, we don't have any seasonality present
#as described by Standard deviation
#Trend component is primarily the reason for the non-stationarity of the data

#Augmented Dicky-Fuller test
#Null Hypothesis : Sales is not Stationary
#Alternate Hypothesis : Sales is Stationary
from statsmodels.tsa.stattools import adfuller
def adfuller_test(sales):
    print("Results of Dicky-Fuller Test")
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(sales_data)
adfuller_test(sales_data_sqrt)
sales_data_sqrt.head(10)
    
    
#Trend 
exponentialDecayWeightedAverage = sales_data_sqrt.ewm(halflife = 12, min_periods = 0, adjust = True).mean()
plt.plot(sales_data_sqrt)
plt.plot(exponentialDecayWeightedAverage, color = 'red')
dtFmt = DateFormatter('%b-%y')
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.legend(['Original', 'Trend'])
plt.show(block = False)


#Without std deviation 
#sales_data_sqrt1 = sales_data_sqrt - exponentialDecayWeightedAverage
#adfuller_test(sales_data_sqrt1)
#RollingStatistics(sales_data_sqrt1)

#Differencing
sales_data_sqrt_diff = sales_data_sqrt - sales_data_sqrt.shift(1)
adfuller_test(sales_data_sqrt_diff.dropna())
plt.ylim(-500, 2000) #To set to original range of earlier plot Sales
RollingStatistics(sales_data_sqrt_diff.dropna())
#The null Hypothesis is Rejected and Now we have stationary Data at our hand
#Now we can move on with actual Modelling

#Components of Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(sales_data_sqrt.dropna(), period = 2)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
ax1 = plt.plot(sales_data_sqrt, label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
ax2 = plt.plot(trend, label = "Trend")
plt.legend(loc = 'best')
plt.subplot(413)
ax3 =plt.plot(seasonal, label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
ax4 = plt.plot(residual, label = 'Irreguarity')
plt.legend(loc = 'best')
plt.tight_layout()

RollingStatistics(residual)

#Auto Regrssive Model
from pandas.plotting import autocorrelation_plot 
autocorrelation_plot(sales_data_sqrt_diff.dropna())

#ACF And PACF plots
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(sales_data_sqrt_diff.dropna(), nlags = 6)
lag_pacf = pacf(sales_data_sqrt_diff.dropna(), nlags = 6, method = 'ols')
#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle = '--', color = 'grey')
plt.axhline(y = -1.96/np.sqrt(len(sales_data_sqrt_diff.dropna())), linestyle = "--", color = 'grey')
plt.axhline(y = 1.96/np.sqrt(len(sales_data_sqrt_diff.dropna())), linestyle = "--", color = 'grey')
plt.title('Autocorrelation Function')
#Plot PACF :
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle = '--', color = 'grey')
plt.axhline(y = -1.96/np.sqrt(len(sales_data_sqrt_diff.dropna())), linestyle = "--", color = 'grey')
plt.axhline(y = 1.96/np.sqrt(len(sales_data_sqrt_diff.dropna())), linestyle = "--", color = 'grey')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#version 2
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sales_data_sqrt_diff.dropna(), lags = 6, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sales_data_sqrt_diff.dropna(), lags = 6, ax = ax2, method = "ols")
fig.show()
#Hence q=0.5

import warnings
warnings.filterwarnings('ignore')

#From earlier p=0, d=1, q=7
#Using ARIMA Model for Forecasting
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(sales_data_sqrt ,order =(1, 1, 0))
result = model.fit()
result.summary()

sales_data_sqrt_diff_forecast = result.predict()
sales_data_sqrt_diff.dropna().plot(figsize = (12, 8))
sales_data_sqrt_diff_forecast.plot(figsize=(12, 8))
plt.legend(["Original", "Predictions"])

sales_data_sqrt_diff_forecast_cumsum =sales_data_sqrt_diff_forecast.cumsum()
predictions_ARIMA_sqrt = pd.Series(sales_data_sqrt.iloc[1], index = sales_data_sqrt.index)
predictions_ARIMA_sqrt = predictions_ARIMA_sqrt.add(sales_data_sqrt_diff_forecast_cumsum, fill_value = 0)
sales_data_sqrt.plot(figsize = (12, 8), legend = True)
predictions_ARIMA_sqrt.plot(figsize = (12, 8), legend = True)
plt.legend(["Original", "Predictions"])

final_predictions = predictions_ARIMA_sqrt**2
sales_data.plot(figsize = (12, 8), legend = True)
final_predictions.plot(figsize = (12, 8), legend = True)
plt.legend(["Original", "Predictions"])

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(sales_data, final_predictions))
print(rmse)

EPSILON =  1e-10 # Yes, Python is awesome and supports scientific notation!
rmspe = (np.sqrt(np.mean(np.square((sales_data - final_predictions) / (sales_data + EPSILON))))) * 100
rmspe

#Forecasting
model = ARIMA(sales_data ,order =(1, 1, 0))
result = model.fit()
result.summary()
x = result.forecast(steps = 12)
x[0]
len(x[0])
result.plot_predict(1, len(sales_data_sqrt_diff)+12)

