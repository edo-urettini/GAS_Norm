# series generator

import numpy as np
import pandas as pd
import math

def series_generator(n_samples, growing=True):
    # Define constants
    if growing:
        A = 1
        b = 0.2
    else:
        A = 130
        b = -0.2

    f = 1.25
    phi = 0
    w = 2*math.pi*f
    T = 1/f
    dt = T/100
    n=n_samples/100
    t = np.arange(0, n*T, dt)
    y = A*np.exp(b*t)*np.sin(w*t + phi)+ 7*t

    #Add an increasing gaussian noise to the data
    noise = np.zeros(len(y))
    for i in range(0, len(y)):
        noise[i] = np.random.normal(0, 0.005*i, 1)
    y = y+noise

    #Separate the series in train and test (half and half)
    y_train = y[0:int(len(y)/2)]
    y_test = y[int(len(y)/2):len(y)]

    return y, y_train, y_test, t


#Define a function that outputs the univariate time series of the stock price of a company

import yfinance as yf

def stock_price_generator(stock_name):
    # Get the data for the stock AAPL
    data = yf.download(stock_name,'2006-01-01','2023-01-01')
    y = data['Adj Close'].values

    #Compute the log returns
    #log_returns = np.zeros(len(y)-1)
    #for i in range(0, len(y)-1):
     #   log_returns[i] = np.log(y[i+1]/y[i])
    log_returns = y
    y = log_returns[0 : (len(log_returns) - len(log_returns)%50)]

    #Divide the data in train and test
    y_train = y[0:int(len(y)*0.8 - len(y)*0.8 % 50)]
    y_test = y[int(len(y)*0.8 - len(y)*0.8 % 50):len(y)]
    
    t = np.arange(0, len(y), 1)

    return y, y_train, y_test, t  