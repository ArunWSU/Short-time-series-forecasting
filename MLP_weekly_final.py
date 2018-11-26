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
def data_Preparation(X,nlags):
    X_t,Y_t=[],[]
    for i in range(0,len(X)-Window_size,1):# One for zero Index One for last point  #  if(i+nlags!=len(X))
            Row=X[i:i+nlags]
            X_t.append(Row)
            Y_t.append(X[i+nlags])
    return np.array(X_t),np.array(Y_t)

def MAPE(X,Y):
    X1,Y1=np.array(X),np.array(Y)
    APE=abs((X1-Y1)/X1)
    MAPE=np.mean(APE)*100
    return MAPE

def MAPE1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# History vector
Annual=pd.read_csv("Annual_load_profile_PJM.csv",header=0,index_col=0,parse_dates=True)
Annual_data=Annual['mw']
Week1=Annual['2017-01-02':'2017-01-08']
History=Week1['mw']
History.index=np.arange(0,len(History),1)
History=History.values.reshape(-1,1)

scaler1=MinMaxScaler(feature_range=(0,1))
History_norm=scaler1.fit_transform(History)

# Choice of window size for window based forecasting
Window_size=5

History_input,History_output=data_Preparation(History_norm,Window_size)
History_output1=History_output.ravel()
History_input1=History_input.reshape(-1,Window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',
             solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(History_input1,History_output1)
MLP_History_output=mlp.predict(History_input1).reshape(-1,1)

#Reshape for Inverse Transform
History_output=History_output.reshape(-1,1)
History_input_inv=scaler1.inverse_transform(History_input1)
History_input_inv=History_input_inv.reshape(-1)
History_output_inv=scaler1.inverse_transform(History_output)
MLP_History_output_inv=scaler1.inverse_transform(MLP_History_output)

RMSE_Train_scaled=(mean_squared_error(History_output,MLP_History_output))
RMSE_Train_actual=(mean_squared_error(History_output_inv,MLP_History_output_inv))
MAE_Train_scaled=(mean_absolute_error(History_output,MLP_History_output))
MAE_Train_actual=(mean_absolute_error(History_output_inv,MLP_History_output_inv))
MAPE_Train=(MAPE(History_output_inv,MLP_History_output_inv))

# History=pd.Series(np.ones((len(Week1),))) Input all ones
dates=pd.date_range(start='2017-01-09',end='2017-12-31')
n=dates.size
B=[]
Start_list=[0]
y=0
w=int(n/7)
Output=np.zeros((w+1,5))
# Time stamp object to convert to date and then string str function
#  Initial logic
#  start=187
#  for x in range(0,n,1):
#  end=start+23
#  Annual_index=Annual[start:end].index
#  start=end
for x in range(0,w,1):

    StartIndex=str(dates[y].date())
    # Pandas list to Series conversion
    if(x==0):
        Start_list=[StartIndex]
    else:
        Start_list.append(StartIndex)

    StartDate=pd.Series(Start_list)

    EndIndex=str(dates[y+6].date())
    # Direct Series Object creation
    if(x==0):
        EndDate=pd.Series(EndIndex)
    else:
        EndDate=EndDate.append(pd.Series(EndIndex),ignore_index=True)   
    
    y=y+7
    
    if x > 0:
        Start=Start_list[x-1]
        End=EndDate[x-1]
        Previousweek=Annual[Start:End]
        History_prev=Previousweek['mw']
        History_prev.index=np.arange(0,len(History_prev),1)
        History_prev=History_prev.values.reshape(-1,1)
        History_norm=scaler1.fit_transform(History_prev)
        History_input,History_output=data_Preparation(History_norm,Window_size)
        History_output1=History_output.ravel()
        History_input1=History_input.reshape(-1,Window_size)
        mlp.fit(History_input1,History_output1)
        
    # Forecast vector
    Daily=Annual[StartIndex:EndIndex] # '2017-01-10':'2017-12-31' Annual
    Forecast=Daily['mw']
    # Forecast['2017-01-12']=0
    Forecast.index=np.arange(0,len(Forecast),1)

    # Reshaping and sclaer fit transformation
    Forecast=Forecast.values.reshape(-1,1)

    #normalization
    scaler=MinMaxScaler(feature_range=(0,1))
    Forecast_norm=scaler.fit_transform(Forecast)
    Forecast_input,Forecast_output=data_Preparation(Forecast_norm,Window_size)

    # Input column vector and Output is one example
    Forecast_output=Forecast_output.ravel()

    # Input to MLP is of form No of samples, No of features
    Forecast_input=Forecast_input.reshape(-1,Window_size)

    # Predictions for training and testing data set
    MLP_Forecast_output=mlp.predict(Forecast_input).reshape(-1,1)
    Forecast_output=Forecast_output.reshape(-1,1)

    # Change scale
    Forecast_output_inv=scaler.inverse_transform(Forecast_output)
    MLP_Forecast_output_inv=scaler.inverse_transform(MLP_Forecast_output)

    # calculate RMSE
    Test_score=(mean_squared_error(Forecast_output,MLP_Forecast_output))
    Output[x][0]=Test_score

    # calculate MAE
    Test_score1=(mean_absolute_error(Forecast_output,MLP_Forecast_output))
    Output[x][1]=Test_score1

    # Calculate RMSe
    Test_score=(mean_squared_error(Forecast_output_inv,MLP_Forecast_output_inv))
    Output[x][2]=Test_score

    # calculate MAE
    Test_score1=(mean_absolute_error(Forecast_output_inv,MLP_Forecast_output_inv))
    Output[x][3]=Test_score1

    # Calculate MAPE
    Test_score1=(MAPE(Forecast_output_inv,MLP_Forecast_output_inv))
    Output[x][4]=Test_score1
    B.append(MLP_Forecast_output_inv)

    if(x==0): # USE CLASS DEFINITION
        Actual_Forecast=Forecast_output_inv
        MLP_Forecast=MLP_Forecast_output_inv
    else:
        Actual_Forecast=np.concatenate([Actual_Forecast,Forecast_output_inv])
        MLP_Forecast=np.concatenate([MLP_Forecast,MLP_Forecast_output_inv])

Outdata=pd.DataFrame(data=Output,columns=['RMSE_Scale','MAE_Scale','RMSE_Actual','MAE_Actual','MAPE'])
Outdata['StartDate']=StartDate
Outdata['EndDate']=EndDate
Outdata.to_excel('Annual_retrain.xlsx','Sheet1')
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