# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:01:30 2021

@author: wanti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime as DT
from datetime import timedelta


########
df = pd.read_excel(r"E:\MMA\MMA841 Ops\topic5\05 EXCEL model and data -- Wells Fargo.xlsx", sheet_name="Demand and Solar Output",na_values=['NA',''])

df.columns
#
df.dtypes()
df['Date/Time'].dtypes()



df['Dates'] = pd.to_datetime(df['Date/Time']).dt.date
df['Time'] = pd.to_datetime(df['Date/Time']).dt.time







## plot before aggreatation


plt.plot(df['Date/Time'], df['Electricity Demand for the Branch (kW)'],alpha=0.7,label='Electricity Demand for the Branch (kW)')
plt.plot(df['Date/Time'], df['Solar System Output (kW)'],alpha=0.5,label='Solar System Output (kW)')
#plt.legend( loc='upper right')
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2) 
plt.title("Demand vs Output by Date Time (kW)")
plt.xlabel("Date Time")
plt.ylabel("kW")
plt.show()

###
#   aggregate

date = df.groupby(['Dates'],as_index=False).agg(
    {
        # Find the min, max, and sum of the duration column
        'Date/Time': "count",
        'Electricity Demand for the Branch (kW)': [sum],     
        'Solar System Output (kW)': [sum]

    }
)

date = date.droplevel(1, axis=1) 

date['Dates'][5].strftime('%A')


# can be use either in 3) or 4)

plt.plot(date['Dates'], date['Electricity Demand for the Branch (kW)'],label='Electricity Demand for the Branch (kW)')
plt.plot(date['Dates'], date['Solar System Output (kW)'],label='Solar System Output (kW)')
#plt.legend( loc='upper right')
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2) 
plt.title("Demand vs Output by Date (kW)")
plt.xlabel("Date")
plt.ylabel("kW")
plt.show()


###############################################
#   aggregate

time = df.groupby(['Time'],as_index=False).agg(
    {
        # Find the min, max, and sum of the duration column
        'Date/Time': "count",
        'Electricity Demand for the Branch (kW)': [sum],     
        'Solar System Output (kW)': [sum]

    }
)

time = time.droplevel(1, axis=1) 
xtickes = np.arange(0, len(time['Time'])+1, 8)



# can be use either in 3) or 4)
time['Time']=time['Time'].astype(str)

plt.plot(time['Time'], time['Electricity Demand for the Branch (kW)'],label='Electricity Demand for the Branch (kW)')
plt.plot(time['Time'], time['Solar System Output (kW)'],label='Solar System Output (kW)')
#plt.legend( loc='upper right')
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.35), shadow=True, ncol=2) 
plt.xticks(time['Time'][xtickes[0:11]],rotation=45)
plt.title("Demand vs Output by Time (kW)")
plt.xlabel("Time")
plt.ylabel("kW")
plt.show()







########################################################################
# time series

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.tsa
import statsmodels.tsa.seasonal
from statsmodels.tsa.seasonal import STL
import copy
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as StatespaceExponentialSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS, BATS
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


##
#

price = pd.read_excel(r"E:\MMA\MMA841 Ops\topic5\05 EXCEL model and data -- Wells Fargo.xlsx", sheet_name="Electricity Prices", index_col='Month')



price.index = pd.date_range('2004-01-01', periods=120, freq='M')
price.head()


# price['Month']= price['Month'].dt.date.astype(str)
# price['Month']= pd.to_numeric(price['Month'].dt.date)


decomposition = statsmodels.tsa.seasonal.seasonal_decompose(price, model='multiplicative')
fig = decomposition.plot()
plt.show()


#
decomposition = sm.tsa.seasonal.seasonal_decompose(price, model='additive')
fig = decomposition.plot()
plt.show()



# 1.3. Loess STL decomposition
stl = STL(price, period=12, robust=True)
res = stl.fit()
fig = res.plot()


##################
# Statespace Exponential smoothing with confidence intervals.

class Callback():
    def __init__(self):
        self.count = 0
    def __call__(self, point):
        print("Interation {} is finished.".format(self.count))
        self.count += 1

es_state_model = StatespaceExponentialSmoothing(price, trend='add', seasonal=12)
es_state_fit = es_state_model.fit(method='lbfgs', maxiter=30, disp=True, callback=Callback(), gtol=1e-2, ftol=1e-2, xtol=1e-2)
es_state_fit.summary()


# Forecast for 30 years ahead.
PERIODS_AHEAD = 360

simulations = es_state_fit.simulate(PERIODS_AHEAD, repetitions=100, error='mul')
simulations.plot(style='-', alpha=0.05, color='grey', legend=False)
plt.title("Simulation Plot")
plt.show()


##
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Prediction with confidence intervals")

# Plot the data (here we are subsetting it to get a better look at the forecasts)
price.plot(ax=ax)

# Construct the forecasts
fcast = es_state_fit.get_forecast(PERIODS_AHEAD).summary_frame()
fcast['mean'].plot(ax=ax, style='k--', label='Forecast')
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.legend()
plt.show()




# 2.2. Holt-Winters exponential smoothing. Unlike statespace models, these ones do not support confidence intervals (since at their current impolementation they do not have underlying statistical model).

from statsmodels.tsa.holtwinters import ExponentialSmoothing

fig, axs = plt.subplots(2, 2, figsize=(30, 30))

# Forecast for 30 years ahead.
PERIODS_AHEAD = 360

ets_models, ets_fits = {}, {}
for idx_trend, trend in enumerate(['add', 'mul']):
    for idx_seasonal, seasonal in enumerate(['add', 'mul']):
        key = 'trend:_{},seasonal:_{}'.format(trend, seasonal)
        title = 'Trend: {}, Seasonal: {}'.format(trend, seasonal)
        
        # Fitting the model and making forecast.
        ets_model = ExponentialSmoothing(price, trend=trend,
                            seasonal=seasonal, seasonal_periods=12)
        ets_fit = ets_model.fit()
        forecast = ets_fit.forecast(PERIODS_AHEAD)
        
        # Saving parameters for future use.
        ets_models[key], ets_fits[key] = ets_model, ets_fit
        
        # Visualization of intial data, fitted model and forecasts.
        axs[idx_trend, idx_seasonal].plot(price, label='Initial Rates')
        axs[idx_trend, idx_seasonal].plot(ets_fit.fittedvalues, label='Fitten Rates')
        axs[idx_trend, idx_seasonal].plot(forecast, label = 'Forecast')
        axs[idx_trend, idx_seasonal].set_title(title)
        axs[idx_trend, idx_seasonal].legend()

plt.show()



# Printing the values of fitted models.

for key, ets_model in ets_fits.items():
    print("Exponential Smooting", key, "\n")
    print(ets_model.summary())



#########################################################
# 3. TBATS
# For more information see here: https://pypi.org/project/tbats/

from tbats import TBATS, BATS

# Initialization and fit for TBATS.
tbats_estimator = TBATS(seasonal_periods=[12])
tbats_model = tbats_estimator.fit(price)

print(tbats_model.summary())


# Forecast for 30 years ahead.
y_forecast, confidence_info = tbats_model.forecast(steps=PERIODS_AHEAD, confidence_level=0.95)

index_of_fc = pd.date_range(price.index[-1], periods = PERIODS_AHEAD + 1, freq='MS')[1:]
fitted_series = pd.Series(y_forecast, index=index_of_fc)
lower_series = pd.Series(confidence_info['lower_bound'], index=index_of_fc)
upper_series = pd.Series(confidence_info['upper_bound'], index=index_of_fc)

plt.plot(price, label='Initial Data')
plt.plot(fitted_series, label='TBATS Forecast')
plt.fill_between(index_of_fc, lower_series, upper_series, alpha=0.15)
plt.legend()
plt.show()





# Time-Series Cross-Validation single-function call is rather complicated for this first session 
# and requires writing Wrapper Class, so let us implement CV functions this time on our own.
import math

# CV for Exponential Smoothing models.
def exponentialSmoothingCVscore(series, trend, seasonal, loss_function):
    errors = []
    tscv = TimeSeriesSplit(n_splits=3) 
    for train, test in tscv.split(series):
        train_length = train.shape[0]
        train_set = series.values[train]
        periodic_length = math.floor(train_length / 12) * 12
        train_set = train_set[-periodic_length:]
        estimator = ExponentialSmoothing(
            train_set, trend=trend, seasonal=seasonal, seasonal_periods=12)
        model = estimator.fit()
        predictions = model.forecast(len(test))
        actual = series.values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
    return errors, np.mean(np.array(errors))


# CV for TBATS.
def scoreCVforTBATS(series, loss_function):
    errors = []
    tscv = TimeSeriesSplit(n_splits=3)
    for train, test in tscv.split(series):
        train_length = train.shape[0]
        estimator = TBATS(n_jobs=1)
        train_set = series.values[train]
        periodic_length = math.floor(train_length / 12) * 12
        train_set = train_set[-periodic_length:]
        model = estimator.fit(train_set)
        predictions = model.forecast(len(test))
        actual = series.values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
    return errors, np.mean(np.array(errors))




##############################
# MAE
print("Score MAE ES add-add", score_mae_add_add)
print("Score MAE ES mul-mul", score_mae_mul_mul)
print("Score MAE TBATS", score_tbats_mae)

plt.plot(errors_mae_add_add, label='MAE ES add-add')
plt.plot(errors_mae_mul_mul, label='MAE ES mul-mul')
plt.plot(errors_tbats_mae,   label='MAE TBATS')
plt.legend()
plt.title("MAE")
plt.show()




##############################
# MSE
print("Score MSE ES add-add", score_mse_add_add)
print("Score MSE ES mul-mul", score_mse_mul_mul)
print("Score MSE TBATS", score_tbats_mse)

plt.plot(errors_mse_add_add, label='MSE ES add-add')
plt.plot(errors_mse_mul_mul, label='MSE ES mul-mul')
plt.plot(errors_tbats_mse,   label='MSE TBATS')
plt.title("MSE")
plt.legend()
plt.show()


## 5. Saving Prediction
final_result = ets_fits['trend:_mul,seasonal:_mul'].forecast(360)
final_result.to_csv(r'E:\MMA\MMA841 Ops\topic5\Predicted Electric Rates.csv')






