import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import statistics as stats
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    MAPE=np.mean(APE)*100
    return MAPE


# History vector
Annual=pd.read_csv("Annual_load_profile_PJM.csv",header=0,index_col=0,parse_dates=True)
Annual_data=Annual['mw']
Week1=Annual['2017-01-02':'2017-01-08']
History=Week1['mw']
History.index=np.arange(0,len(History),1)
History=History.values.reshape(-1,1)

scaler_history=MinMaxScaler(feature_range=(0,1))
History_norm=scaler_history.fit_transform(History)

# Choice of window size for window based forecasting
window_size=5
History_input,History_output=data_preparation(History_norm,window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',
             solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(History_input,History_output)
MLP_History_output=mlp.predict(History_input)

#Reshape for Inverse Transform
History_input_inv=scaler_history.inverse_transform(History_input).reshape(-1)
History_output_inv=scaler_history.inverse_transform(History_output.reshape(-1,1))
MLP_History_output_inv=scaler_history.inverse_transform(MLP_History_output.reshape(-1,1))

# Weeks
Output=np.zeros((52,5))
Output[0][0]=(mean_squared_error(History_output,MLP_History_output))
Output[0][1]=(mean_absolute_error(History_output,MLP_History_output))
Output[0][2]=(mean_squared_error(History_output_inv,MLP_History_output_inv))
Output[0][3]=(mean_absolute_error(History_output_inv,MLP_History_output_inv))
Output[0][4]=(mape(History_output_inv,MLP_History_output_inv))

# History=pd.Series(np.ones((len(Week1),))) Input all ones
dates=pd.date_range(start='2017-01-09',end='2017-12-31')
n=dates.size
B=[]
Start_list=[0]
y=0
w=int(n/7)
StartDate=pd.Series('2017-01-02')
EndDate=pd.Series('2017-01-08')
retrain=1
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
    weekly=Annual[StartIndex:EndIndex]
    weekly_data=weekly['mw']
    weekly_data.index=np.arange(0,len(weekly_data),1)
    
    # Reshaping and scaler fit transformation
    weekly_data=weekly_data.values.reshape(-1,1)
    scaler_forecast=MinMaxScaler(feature_range=(0,1))
    weekly_data_norm=scaler_forecast.fit_transform(weekly_data)
    Forecast_input,Forecast_output_norm=data_preparation(weekly_data_norm,window_size)
    Forecast_output_actual=scaler_forecast.inverse_transform(Forecast_output_norm.reshape(-1,1))
    if(retrain==1):
        scaler_hist=MinMaxScaler(feature_range=(0,1))
        history=Annual[StartDate[x]:EndDate[x]].mw
        # history.index=np.arange(0,len(history),1)
        history_norm=scaler_hist.fit_transform(history.values.reshape(-1,1))
        history_input,history_output=data_preparation(history_norm,window_size)
        mlp.fit(history_input,history_output)
        
    mlp_forecast_weekly_norm=np.zeros(Forecast_input.shape[0])
    for i in range(0,weekly_data_norm.shape[0]-window_size,1):
        previous_window=weekly_data_norm[i:i+window_size].reshape(-1,window_size)
        mlp_forecast_weekly_norm[i]=mlp.predict(previous_window)

    mlp_forecast_weekly_actual=scaler_forecast.inverse_transform(mlp_forecast_weekly_norm.reshape(-1,1))
    # calculate RMSE
    Test_score=(mean_squared_error(Forecast_output_norm,mlp_forecast_weekly_norm))
    Output[x+1][0]=Test_score

    # calculate MAE
    Test_score1=(mean_absolute_error(Forecast_output_norm,mlp_forecast_weekly_norm))
    Output[x+1][1]=Test_score1

    # Calculate RMSe
    Test_score=(mean_squared_error(Forecast_output_actual,mlp_forecast_weekly_actual))
    Output[x+1][2]=Test_score

    # calculate MAE
    Test_score1=(mean_absolute_error(Forecast_output_actual,mlp_forecast_weekly_actual))
    Output[x+1][3]=Test_score1

    # Calculate MAPE
    Test_score1=(mape(Forecast_output_actual,mlp_forecast_weekly_actual))
    Output[x+1][4]=Test_score1
    B.append(mlp_forecast_weekly_actual)

    if(x==0): # USE CLASS DEFINITION
        Actual_Forecast=Forecast_output_actual
        MLP_Forecast=mlp_forecast_weekly_actual
    else:
        Actual_Forecast=np.concatenate([Actual_Forecast,Forecast_output_actual])
        MLP_Forecast=np.concatenate([MLP_Forecast,mlp_forecast_weekly_actual])

Outdata=pd.DataFrame(data=Output,columns=['RMSE_Scale','MAE_Scale','RMSE_Actual','MAE_Actual','MAPE'])
Outdata['StartDate']=StartDate
Outdata['EndDate']=EndDate
# Outdata.to_excel('Annual.xlsx','Sheet1')
'''
# plotting in matplotlib
Scale_Xh=np.arange(1,len(History_output)+1,1)
fig=plt.figure()
plt.plot(Scale_Xh,History_output_inv,color='gray',label='Training load',linewidth=2,linestyle='-')
plt.plot(Scale_Xh,MLP_History_output_inv,color='crimson',label='MLP Training Load',linewidth=2,linestyle='--')
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Training Set')
plt.legend()
plt.show()

# Testing dataset
fig=plt.figure()
Scale_Xh=np.arange(1,len(Actual_Forecast)+1,1)
plt.plot(Scale_Xh,Actual_Forecast,color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh, MLP_Forecast,color='crimson',label='MLP Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Testing Set')
plt.legend()
plt.show()
'''