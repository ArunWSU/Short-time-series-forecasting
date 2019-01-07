import numpy as np
import pandas as pd
import math
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

# Fixing the random number for neural net
np.random.seed(1)
# Creates the input and output data for time series forecasting
def data_preparation(X,nlags):
    X_t,Y_t=[],[]
    # One for zero Index One for last point  #  if(i+nlags!=len(X))
    for i in range(0,len(X)-window_size,1):
            Row=X[i:i+nlags]
            X_t.append(Row)
            Y_t.append(X[i+nlags])
    return np.array(X_t).reshape(-1,window_size),np.array(Y_t).ravel()

def mape(X,Y):
    X1,Y1=np.array(X).reshape(-1,1),np.array(Y).reshape(-1,1)
    APE=abs((X1-Y1)/X1)
    mape_calc=np.mean(APE)*100
    return mape_calc


# history vector
Annual=pd.read_csv("3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) # Annual_data.idxmax() # Annual_data.idxmin()
history_start_date='2017-01-02' 
history_end_date='2017-10-17'
history=Annual[history_start_date:history_end_date] # Approximately 0.8=9.5 months
history=history.values # Long way history.index=np.arange(0,len(history),1) history=history.values.reshape(-1,1)
forecast=Annual['2017-10-18':'2017-12-31']
forecast=forecast.values
'''
## CHOICE OF WINDOW SIZE FOR FORECASTING
Metrics_output_window_select=np.zeros((20,6))

for x in range(1,15):
    # Input to MLP is of form No of samples, No of features
    window_size=x
    history_input,history_output=data_preparation(history,window_size)
    forecast_input,forecast_output=data_preparation(forecast,window_size)
    
    #  Specify MLP regressor model
    mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',solver='lbfgs',random_state=1)
    mlp.fit(history_input,history_output)
    
    # Predict the outputs
    mlp_history_output=mlp.predict(history_input).reshape(-1,1)
    mlp_forecast_output=mlp.predict(forecast_input).reshape(-1,1)
    
    # Calculate the error metrics
    Metrics_output_window_select[x][0]=mean_squared_error(history_output,mlp_history_output)
    Metrics_output_window_select[x][1]=mean_absolute_error(history_output,mlp_history_output)
    Metrics_output_window_select[x][2]=mape(history_output,mlp_history_output)
    Metrics_output_window_select[x][3]=mean_squared_error(forecast_output,mlp_forecast_output)
    Metrics_output_window_select[x][4]=mean_absolute_error(forecast_output,mlp_forecast_output)
    Metrics_output_window_select[x][5]=mape(forecast_output,mlp_forecast_output)
    
Metrics_output_window_select1=pd.DataFrame(data=Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
Metrics_output_window_select1.to_csv('window_check.csv')


## CHOICE OF NUMBER OF NEURONS FOR HIDDEN LAYER
Metrics_output_neurons_select=np.zeros((20,6))
window_size=4
history_input,history_output=data_preparation(history,window_size)
forecast_input,forecast_output=data_preparation(forecast,window_size)

for x in range(1,10):
    
    #  Specify MLP regressor model
    mlp=MLPRegressor(hidden_layer_sizes=(x,),activation='identity',solver='lbfgs',random_state=1)
    mlp.fit(history_input,history_output)
    
    # Predict the outputs
    mlp_history_output=mlp.predict(history_input).reshape(-1,1)
    mlp_forecast_output=mlp.predict(forecast_input).reshape(-1,1)
    
    # Calculate the error metrics
    Metrics_output_neurons_select[x][0]=mean_squared_error(history_output,mlp_history_output)
    Metrics_output_neurons_select[x][1]=mean_absolute_error(history_output,mlp_history_output)
    Metrics_output_neurons_select[x][2]=mape(history_output,mlp_history_output)
    Metrics_output_neurons_select[x][3]=mean_squared_error(forecast_output,mlp_forecast_output)
    Metrics_output_neurons_select[x][4]=mean_absolute_error(forecast_output,mlp_forecast_output)
    Metrics_output_neurons_select[x][5]=mape(forecast_output,mlp_forecast_output)
    
Metrics_output_neurons_select1=pd.DataFrame(data=Metrics_output_neurons_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
Metrics_output_neurons_select1.to_csv('neuron_check.csv')
      
'''
# Choice of window size for window based forecasting
window_size=4
history_input,history_output=data_preparation(history,window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(9,),activation='identity',solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(history_input,history_output)
mlp_history_output=mlp.predict(history_input)

# Weeks
error_metrics=np.zeros((52,3))
error_metrics[0][0]=(mean_squared_error(history_output,mlp_history_output))
error_metrics[0][1]=(mean_absolute_error(history_output,mlp_history_output))
error_metrics[0][2]=(mape(history_output,mlp_history_output))

# Checking for normal fit
annual_data=Annual['2017'].use
#plt.figure()
#annual_data.plot.hist()
#plt.title('Histogram of Annual data',fontsize=18)
##plt.figure()
##probplot(annual_data)
##plt.show()
#normalized_values=norm.pdf(annual_data.values)
#plt.hist(normalized_values)
#plt.show()

# Calculate thresholds
annual_data_mean=statistics.mean(annual_data)
annual_data_sd=statistics.stdev(annual_data)
upper_threshold=annual_data_mean+3*annual_data_sd # 18089 small
lower_threshold=annual_data_mean-3*annual_data_sd


B=[]
y=0
forecast_start_date_list=pd.Series(history_start_date)
forecast_end_date_list=pd.Series(history_end_date)

# Length and time to be forecasted
forecast_start_date='2017-10-18' 
forecast_end_date='2017-12-31'
forecast=Annual[forecast_start_date:forecast_end_date].use.copy()
dates=pd.date_range(start=forecast_start_date,end=forecast_end_date)

# Check on data frequency
daily_datapoints=96 
max_weekly_points=daily_datapoints*7
no_of_dates=dates.shape[0]
no_of_weeks=math.floor(no_of_dates/7) # Instead of int use ceil func to round up

Lastwindow=[]
retrain=0
ADAM=1
adam_forecast_output=np.zeros(forecast.shape[0]).reshape(-1,1)
threshold_violations=adam_forecast_output.copy()
Act1=np.array([]) 
MLP1=np.array([])
window_index=0
for x in range(0,no_of_weeks,1):
    # Modification maybe Mon to Mon
    start_index=str(dates[y].date())
    y=y+6
    if(x==no_of_weeks):
        y=-1
    end_index=str(dates[y].date())
    y=y+1
    
    forecast_start_date_list=forecast_start_date_list.append(pd.Series(start_index),ignore_index=True)
    forecast_end_date_list=forecast_end_date_list.append(pd.Series(end_index),ignore_index=True)

    # Weekly Data
    weekly_data=forecast[start_index:end_index] 
    weekly_data.index=np.arange(0,len(weekly_data),1)
    forecast_output_actual=weekly_data.values.copy()
    if(x==0):
        adam_forecast_output[0:window_size]=forecast_output_actual[0:window_size].reshape(-1,1).copy() 
        forecast_output_actual=forecast_output_actual[window_size:]

    mlp_forecast_weekly_actual=np.zeros(forecast_output_actual.shape[0])    
    no_of_datapoints_in_week=forecast_output_actual.shape[0]
    for i in range(0,no_of_datapoints_in_week,1):
         previous_window=adam_forecast_output[window_index:window_index+window_size].reshape(-1,window_size)
         mlp_forecast_weekly_actual[i]=mlp.predict(previous_window) 
         if((lower_threshold < forecast_output_actual[i]) and (forecast_output_actual[i] < upper_threshold)):
                 adam_forecast_output[window_index+window_size]=forecast_output_actual[i].copy()
         else:
                 adam_forecast_output[window_index+window_size]=mlp_forecast_weekly_actual[i].copy()
                 threshold_violations[window_index+window_size]=1
         window_index=window_index+1

    # Calculate RMSE
    error_metrics[x+1][0]=(mean_squared_error(forecast_output_actual,mlp_forecast_weekly_actual))

    # calculate MAE
    error_metrics[x+1][1]=(mean_absolute_error(forecast_output_actual,mlp_forecast_weekly_actual))
    
    # Calculate MAPE
    error_metrics[x+1][2]=(mape(forecast_output_actual,mlp_forecast_weekly_actual))
    B.append(mlp_forecast_weekly_actual)
    
    Act1=np.append(Act1,forecast_output_actual.reshape(-1,1))
    MLP1=np.append(MLP1,mlp_forecast_weekly_actual.reshape(-1,1))

    if(x==0): # USE CLASS DEFINITION
        Actual_forecast1=forecast_output_actual.reshape(-1,1) # Alternate Actual_forecast=Forecast[-Window_Size:]
        MLP_forecast=mlp_forecast_weekly_actual.reshape(-1,1)
    else:
        Actual_forecast1=np.concatenate([Actual_forecast1,forecast_output_actual.reshape(-1,1)])
        MLP_forecast=np.concatenate([MLP_forecast,mlp_forecast_weekly_actual.reshape(-1,1)])

Actual_forecast=Actual_forecast1[window_size:].reshape(-1,1)

#filename='Annual1.xlsx'
different_regression_metrics=pd.DataFrame(data=error_metrics,columns=['RMSE_Actual','MAE_Actual','MAPE'])
different_regression_metrics['forecast_start_date_list'],different_regression_metrics['forecast_end_date_list']=[forecast_start_date_list,forecast_end_date_list] # different_regression_metrics['forecast_start_date_list']=forecast_start_date_list # different_regression_metrics['forecast_end_date_list']=forecast_end_date_list
# different_regression_metrics.to_excel(filename,'Sheet1')

# filename='Annual1.xlsx'
complete_forecast_output=np.stack((Actual_forecast1,MLP_forecast),axis=1).reshape(-1,2)
complete_forecast_data=pd.DataFrame(data=complete_forecast_output,columns=['Actual','MLP'])


# plotting in matplotlib
'''
Scale_Xh=np.arange(1,len(History_output)+1,1)
fig=plt.figure()
plt.plot(Scale_Xh,Actual_forecast,color='gray',label='Training load',linewidth=2,linestyle='-')
plt.plot(Scale_Xh,MLP_History_output_inv,color='crimson',label='MLP Training Load',linewidth=2,linestyle='--')
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Training Set')
plt.legend()
plt.show()
'''
# Testing dataset
fig=plt.figure()
Scale_Xh=np.arange(1,len(Actual_forecast1)+1,1)
plt.plot(Scale_Xh,Actual_forecast1,color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh,MLP_forecast,color='crimson',label='MLP Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Testing Set')
plt.legend()
plt.show()

# Testing dataset
fig=plt.figure()
Scale_Xh=np.arange(1,len(forecast)+1,1)
plt.plot(Scale_Xh,forecast,color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh,adam_forecast_output,color='crimson',label='ADAM Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Testing Set')
plt.legend()
plt.show()
