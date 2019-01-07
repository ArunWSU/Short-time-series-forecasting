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
import seaborn as sns
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
    X1,Y1=np.array(X),np.array(Y)
    APE=abs((X1-Y1)/X1)
    mape_calc=np.mean(APE)*100
    return mape_calc


# History vector
Annual=pd.read_csv("Annual_load_profile_PJM.csv",header=0,index_col=0,parse_dates=True)
Week1=Annual['2017-01-02':'2017-01-08']
History=Week1['mw']
History.index=np.arange(0,len(History),1)
History=History.values.reshape(-1,1)

# Checking for normal fit
annual_data=Annual['2017'].mw
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

# Calculate thresholds
annual_data_mean=statistics.mean(annual_data)
annual_data_sd=statistics.stdev(annual_data)
upper_threshold=annual_data_mean+3*annual_data_sd # 18089 small
lower_threshold=annual_data_mean-3*annual_data_sd


# Choice of window size for window based forecasting
window_size=5
History_input,History_output=data_preparation(History,window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',
             solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(History_input,History_output)
MLP_History_output=mlp.predict(History_input)


# Weeks
error_metrics=np.zeros((52,3))
error_metrics[0][0]=(mean_squared_error(History_output,MLP_History_output))
error_metrics[0][1]=(mean_absolute_error(History_output,MLP_History_output))
error_metrics[0][2]=(mape(History_output,MLP_History_output))

# History=pd.Series(np.ones((len(Week1),))) Input all ones
dates=pd.date_range(start='2017-01-09',end='2017-12-31')
B=[]
Start_list=[0]
y=0
StartDate=pd.Series('2017-01-02')
EndDate=pd.Series('2017-01-08')

# Length and time to be forecasted
forecast_start_date='2017-01-09' #str(dates[0].date())
forecast_end_date='2017-01-31'
Forecast=Annual[forecast_start_date:forecast_end_date].mw.copy # 2017-12-31
Forecast['2017-01-10']=0
no_of_dates=Forecast.size
w=math.ceil(no_of_dates/(24*7)) # Instead of int use ceil func

Lastwindow=[]
retrain=0
ADAM=1
adam_forecast_output=np.zeros(Forecast.shape[0]).reshape(-1,1)
threshold_violations=adam_forecast_output.copy()
Act1=np.array([]) 
MLP1=np.array([])
window_index=0
# Time stamp object to convert to date and then string str function
for x in range(0,w,1):
    StartIndex=str(dates[y].date())
    EndIndex=str(dates[y+6].date())
    y=y+7
    
    StartDate=StartDate.append(pd.Series(StartIndex),ignore_index=True)
    EndDate=EndDate.append(pd.Series(EndIndex),ignore_index=True)
#    # Pandas list to Series conversions
#    if(x==0):
#        Start_list=[StartIndex]
#    else:
#        Start_list.append(StartIndex)
#
#    StartDate=pd.Series(Start_list)
#
#    # Direct Series Object creation
#    if(x==0):
#        EndDate=pd.Series(EndIndex)
#    else:
#        EndDate=EndDate.append(pd.Series(EndIndex),ignore_index=True)

    # Weekly Data
    weekly_data=Forecast[StartIndex:EndIndex] # Annual change to Forecast
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

    '''
     if(x==0):
        rolling_forecast_window_input=forecast_output_actual
        forecast_output_actual=forecast_output_actual[window_size:]
        no_of_datapoints_in_week=forecast_output_actual.shape[0]
    else:
        rolling_forecast_window_input=np.concatenate([rolling_forecast_window_input,forecast_output_actual.reshape(-1,1)])  
        no_of_datapoints_in_week=forecast_output_actual.shape[0]          
    if(retrain==1):
        history=Annual[StartDate[x]:EndDate[x]].mw
        history.index=np.arange(0,len(history),1)
        history_input,history_output=data_preparation(history,window_size)
        mlp.fit(history_input,history_output)
        
    mlp_forecast_weekly_actual=np.zeros(forecast_output_actual.shape[0])
    
    for i in range(0,no_of_datapoints_in_week,1):  
            previous_window=rolling_forecast_window_input[i:i+window_size].reshape(-1,window_size)
            mlp_forecast_weekly_actual[i]=mlp.predict(previous_window) 
            # mlp_forecast_weekly_actual_individual[i]=scaler_forecast.inverse_transform(mlp_forecast_weekly_norm[i]).reshape(1,-1)
            if(ADAM==1):
                if((lower_threshold < forecast_output_actual[i]) and (forecast_output_actual[i] < upper_threshold)):
                    adam_forecast_output[i]=forecast_output_actual[i].copy()
                else:
                    adam_forecast_output[i]=mlp_forecast_weekly_actual[i].copy()
                    rolling_forecast_window_input[i+window_size]= mlp_forecast_weekly_actual[i]
    '''                
    #rolling_forecast_window_input=forecast_output_actual[-window_size:].reshape(-1,1)   # change forecast_output_actual
    
    # Check both
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
different_regression_metrics['StartDate'],different_regression_metrics['EndDate']=[StartDate,EndDate] # different_regression_metrics['StartDate']=StartDate # different_regression_metrics['EndDate']=EndDate
# different_regression_metrics.to_excel(filename,'Sheet1')

# filename='Annual1.xlsx'
complete_forecast_output=np.stack((Actual_forecast1,MLP_forecast),axis=1).reshape(-1,2)
complete_forecast_data=pd.DataFrame(data=complete_forecast_output,columns=['Actual','MLP'])
#complete_forecast_data.to_excel(filename,'Sheet2')

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
Scale_Xh=np.arange(1,len(Forecast)+1,1)
plt.plot(Scale_Xh,Forecast,color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh,adam_forecast_output,color='crimson',label='ADAM Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Testing Set')
plt.legend()
plt.show()
