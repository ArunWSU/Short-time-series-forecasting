import numpy as np
import pandas as pd
import statistics
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from pandas import ExcelWriter
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

def error_compute(X,Y):
    return np.mean(np.power(abs(np.array(X)-np.array(Y)),2)),np.mean(abs(np.array(X)-np.array(Y))),np.mean(abs(np.array(X)-np.array(Y))/np.array(X))*100

class ModelPerformance:
    # To view the values used in error computation
    actual_list,forecast_list=[],[]
    
    def __init__(self,forecast_size):
        ModelPerformance.error_metrics=np.zeros((forecast_size+3,3))
    
    def rolling_window_error(self,forecast_output_actual,mlp_forecast_actual,i,performance_window_size):
        if(performance_window_size==1):
            self.actual_values=forecast_output_actual[i] 
            self.forecast_values=mlp_forecast_actual[i]
            self.error_metrics[i][0],self.error_metrics[i][1],self.error_metrics[i][2]=error_compute(self.actual_values,self.forecast_values)
        else:
            end_index=i+1
            if(end_index>=performance_window_size):
             self.actual_values=forecast_output_actual[end_index-performance_window_size:end_index] 
             self.actual_list.append(self.actual_values)
             self.forecast_values=mlp_forecast_actual[end_index-performance_window_size:end_index] 
             self.forecast_list.append(self.forecast_values)
             self.error_metrics[i][0],self.error_metrics[i][1],self.error_metrics[i][2]=error_compute(self.actual_values,self.forecast_values)

class AnomalyDetect:
    select=1
    threshold_violations=[]
    
    def __init__(self,forecast_size):
        AnomalyDetect.threshold_violations=np.zeros((forecast_size,1)) 
        
    def check_anomalies(GaussianThresholds,select,current_value,i):
        if(select==1):
            GaussianThresholds.gaussian_check(current_value,i)
        else:
            raise Exception('Not a valid selection!')

class GaussianThresholds(AnomalyDetect):
    def __init__(self,annual_data1,forecast_size):
         AnomalyDetect.__init__(self,forecast_size)
         GaussianThresholds.annual_data=annual_data1
         annual_data_mean=statistics.mean(annual_data)
         annual_data_sd=statistics.stdev(annual_data)
         self.upper_threshold=annual_data_mean+3*annual_data_sd
         self.lower_threshold=annual_data_mean-3*annual_data_sd
     
    def gaussian_check(GaussianThresholds,current_value,i): 
      if((GaussianThresholds.lower_threshold < current_value) and (current_value < GaussianThresholds.upper_threshold)):
          GaussianThresholds.threshold_violations[i]=0
      else:
          GaussianThresholds.threshold_violations[i]=1
    
     
    
# Reading and creating the history vector
Annual=pd.read_csv("Annual_load_profile_PJM.csv",header=0,index_col=0,parse_dates=True)
Week1=Annual['2017-01-02':'2017-01-08']
history=Week1['mw']
history.index=np.arange(0,len(history),1)
history=history.values.reshape(-1,1)
annual_data=Annual['2017'].mw

'''
# Checking for normal fit
plt.figure()
annual_data.plot.hist()
plt.title('Histogram of Annual data',fontsize=18)
plt.figure()
probplot(annual_data,plot=plt)
plt.show()
normalized_values=norm.pdf(annual_data.values)
scale=np.arange(0,normalized_values.shape[0],1)
plt.plot(scale,normalized_values)
plt.show()
'''


'''
### FINDING MODEL PARAMETERS
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

## Training using the determined model parameters
# Choice of window size for window based forecasting
window_size=5
history_input,history_output=data_preparation(history,window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(history_input,history_output)
mlp_history_output=mlp.predict(history_input)

'''
# Visualize the training
Scale_Xh=np.arange(1,len(history_output)+1,1)
fig=plt.figure()
plt.plot(Scale_Xh,history_output,color='gray',label='Training load',linewidth=2,linestyle='-')
plt.plot(Scale_Xh,mlp_history_output,color='crimson',label='MLP Training Load',linewidth=2,linestyle='--')
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Training Set')
plt.legend()
plt.show()
'''
# history=pd.Series(np.ones((len(Week1),))) Input all ones
## Forecasting
dates=pd.date_range(start='2017-01-09',end='2017-12-31')
B=[]
y=0
StartDate=pd.Series('2017-01-02')
EndDate=pd.Series('2017-01-08')

# Length and time to be forecasted
forecast_start_date='2017-01-09' #str(dates[0].date())
forecast_end_date='2017-01-15'
forecast=Annual[forecast_start_date:forecast_end_date].mw.copy() # 2017-12-31
#forecast['2017-01-10 00:00:00':'2017-01-10 16:00:00']=0 24:41
forecast=forecast.values
actual_forecast=forecast.copy()
zero_start_index=24
zero_end_index=41
#forecast[zero_start_index:zero_end_index]=0
#forecast.index=np.arange(0,len(forecast),1)
window_index=0

forecast_output_actual=forecast[window_size:].copy()
mlp_forecast_actual=np.zeros(forecast_output_actual.shape[0]) 
adam_forecast_output=np.zeros(forecast.shape[0]).reshape(-1,1)
adam_forecast_output[0:window_size]=forecast[0:window_size].reshape(-1,1).copy() 
threshold_violations=mlp_forecast_actual.copy()

model_mlp_obj=ModelPerformance(forecast_output_actual.shape[0])
performance_window_size=1
anomaly_detection_method=1

annual_data_mean=statistics.mean(annual_data)
annual_data_sd=statistics.stdev(annual_data)
upper_threshold=annual_data_mean+3*annual_data_sd
lower_threshold=annual_data_mean-3*annual_data_sd

no_of_datapoints=forecast_output_actual.shape[0]
gaus_obj=GaussianThresholds(annual_data,forecast_output_actual.shape[0]) 

for i in range(0,no_of_datapoints,1):
     previous_window=adam_forecast_output[window_index:window_index+window_size].reshape(-1,window_size)
     mlp_forecast_actual[i]=mlp.predict(previous_window) 
     gaus_obj.check_anomalies(anomaly_detection_method,forecast_output_actual[i],i)
     if(gaus_obj.threshold_violations[i]==0):
             adam_forecast_output[window_index+window_size]=forecast_output_actual[i].copy()
             model_mlp_obj.rolling_window_error(forecast_output_actual,mlp_forecast_actual,i,performance_window_size)
#             model_mlp_obj.error_compute(forecast_output_actual,mlp_forecast_actual,i,performance_window_size)
     else:
         adam_forecast_output[window_index+window_size]=mlp_forecast_actual[i].copy()          
     window_index=window_index+1


# Check on data frequency
daily_datapoints=96 
max_weekly_points=daily_datapoints*7
dates=pd.date_range(start=forecast_start_date,end=forecast_end_date)
no_of_dates=dates.shape[0]
no_of_weeks=math.floor(no_of_dates/7) 

# Writing outputs to Excel file
#different_regression_metrics=pd.DataFrame(data=model_mlp_obj.error_metrics,columns=['RMSE_Actual One datapoint', 'MAE_Actual','MAPE'])
#filename='MLP_performance.xlsx'
#writer=ExcelWriter(filename)
#different_regression_metrics.to_excel(writer,'sheet1')
#writer.save()


# Testing dataset
#fig=plt.figure()
#Scale_Xh=np.arange(1,len(Actual_forecast1)+1,1)
#plt.plot(Scale_Xh,Actual_forecast1,color='gray',label='Actual load',marker="v")
#plt.plot(Scale_Xh,MLP_forecast,color='crimson',label='MLP Forecasted Load',marker="^")
#plt.plot(Scale_Xh,forecast_with_zero_interval,color='blue',label='Actual Load without',marker="o") # linewidth=2,linestyle='--'
#plt.xlabel('Time(Hours)', fontsize=18)
#plt.ylabel('Load(MW)', fontsize=18)
#plt.title('Testing Set')
#plt.legend()
#plt.show()

## Complete forecast vector
#fig=plt.figure()
#Scale_Xh=np.arange(1,len(forecast)+1,1)
#plt.plot(Scale_Xh,forecast,color='gray',label='Load with simulated zero values',marker="v")
#plt.plot(Scale_Xh,adam_forecast_output,color='crimson',label='ADAM Forecasted Load',marker="^") # linewidth=2,linestyle='--'
#plt.plot(Scale_Xh,actual_forecast,color='blue',label='Actual Load without zero values',marker="o")
#plt.xlabel('Time(Hours)', fontsize=18)
#plt.ylabel('Load(MW)', fontsize=18)
#plt.title('Testing Set with 9 missing points')
#plt.legend()
#plt.show()
#plt.savefig('Weekly Forecast 5')
#
## Values forecasted
#fig=plt.figure()
#Scale_Xh=np.arange(1,len(forecast_output_actual)+1,1)
#plt.plot(Scale_Xh,forecast_output_actual,color='gray',label='Actual load to be forecasted',marker="v")
#plt.plot(Scale_Xh,mlp_forecast_actual,color='crimson',label='MLP Forecasted Load',marker="^")
#plt.xlabel('Time(Hours)', fontsize=18)
#plt.ylabel('Load(MW)', fontsize=18)
#plt.title('Testing Set with 9 missing points excluding window size')
#plt.legend()
#plt.show()
#plt.savefig('Weekly Forecast 6')

#
## Error for the whole of forecast
#index=np.arange(19,36,1)
# Weeks
#complete_error_metrics=np.zeros((4,3))
#complete_error_metrics[0][0]=(mean_squared_error(history_output,mlp_history_output))
#complete_error_metrics[0][1]=(mean_absolute_error(history_output,mlp_history_output))
#complete_error_metrics[0][2]=(mape(history_output,mlp_history_output))
#if(case)
#original_forecast=forecast_output_actual
#forecast_output_actual=np.delete(forecast_output_actual,index)
#mlp_forecast_actual=np.delete(mlp_forecast_actual,index)
#MSE_forecast=mean_squared_error(forecast_output_actual,mlp_forecast_actual)
#MAE_forecast=mean_absolute_error(forecast_output_actual,mlp_forecast_actual)
#MAPE_forecast=mape(forecast_output_actual,mlp_forecast_actual)

# zero time interval error calculation
mlp_forecast_zero_interval=adam_forecast_output[zero_start_index:zero_end_index]
actual_val_zero_interval=actual_forecast[zero_start_index:zero_end_index]
MSE_zero_time_instant=mean_squared_error(actual_val_zero_interval,mlp_forecast_zero_interval)
MAE_zero_time_instant=mean_absolute_error(actual_val_zero_interval,mlp_forecast_zero_interval)
#MAPE_zero_time_instant=mape(actual_val_zero_interval,mlp_forecast_zero_interval)