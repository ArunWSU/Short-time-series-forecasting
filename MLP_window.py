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
'''
History=pd.read_csv("History_noise.csv",squeeze=True)
Forecast=pd.read_csv("Forecast_validation_noise.csv",squeeze=True)
''' 

# History vector
Annual=pd.read_csv("Annual_load_profile_PJM.csv",header=0,index_col=0,parse_dates=True)
Annual_data=Annual['mw']
Mean=stats.mean(Annual_data)
Stdev=stats.stdev(Annual_data)
Upper_threshold=Mean+3*Stdev
Lower_threshold=Mean-3*Stdev

Week1=Annual['2017-01-02':'2017-01-08']
History=Week1['mw']
History.index=np.arange(0,len(History),1)
# History=pd.Series(np.ones((len(Week1),))) # Input all ones

# Forecast vector
Week2=Annual['2017-01-10':'2017-12-31'] # '2017-01-10':'2017-12-31' Annual
Forecast=Week2['mw']
Forecast['2017-01-12 07:00:00']= Mean+20*Stdev
Forecast['2017-01-12 14:00:00']= Mean+7*Stdev
Forecast['2017-01-12 22:00:00']= Mean+4*Stdev
#Forecast['2017-01-12']=0 # case when it is zero
#Forecast[24:30]=0
Forecast.index=np.arange(0,len(Forecast),1)

# Reshaping and sclaer fit transformation
History=History.values.reshape(-1,1)
Forecast=Forecast.values.reshape(-1,1)

#normalization
scaler1=MinMaxScaler(feature_range=(0,1))
History_norm=scaler1.fit_transform(History)
scaler=MinMaxScaler(feature_range=(0,1))
Forecast_norm=scaler.fit_transform(Forecast)

# Choice of window size for window based forecasting
Window_size=1
History_input,History_output=data_Preparation(History_norm,Window_size)
Forecast_input,Forecast_output=data_Preparation(Forecast_norm,Window_size)

# Input column vector and Output is one example
History_output=History_output.ravel()
Forecast_output=Forecast_output.ravel()

# Input to MLP is of form No of samples, No of features
History_input=History_input.reshape(-1,Window_size)
Forecast_input=Forecast_input.reshape(-1,Window_size)

#  Specify MLP regressor model
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='identity',
                 solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(History_input,History_output)

# Predictions for training and testing data set
MLP_History_output=mlp.predict(History_input).reshape(-1,1)
MLP_Forecast_output=mlp.predict(Forecast_input).reshape(-1,1)

#Reshape for Inverse Transform
History_output=History_output.reshape(-1,1)
Forecast_output=Forecast_output.reshape(-1,1)

# Change scale
History_input_inv=scaler1.inverse_transform(History_input)
History_input_inv=History_input_inv.reshape(-1)
History_output_inv=scaler1.inverse_transform(History_output)
MLP_History_output_inv=scaler1.inverse_transform(MLP_History_output)

Forecast_output_inv=scaler.inverse_transform(Forecast_output)
MLP_Forecast_output_inv=scaler.inverse_transform(MLP_Forecast_output)

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
Scale_Xh=np.arange(1,len(Forecast_output)+1,1)
plt.plot(Scale_Xh,Forecast_output_inv,color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh,MLP_Forecast_output_inv,color='crimson',label='MLP Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Testing Set')
plt.legend()
plt.show()

History=History.reshape(-1)

'''
fig=plt.figure()
Scale_Xh=np.arange(1,len(Forecast_output)+1,1)
plt.scatter(Scale_Xh,Forecast_output_inv,color='gray',marker="v",label='Actual load')
plt.scatter(Scale_Xh,MLP_Forecast_output_inv,color='crimson',marker="^",label='MLP Forecasted Load') # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.legend()
plt.show()
'''

'''
# https://www.propharmagroup.com/blog/understanding-statistical-intervals-part-2-prediction-intervals/
Alpha=0.05
Delta=Alpha/2*
'''
#Output=[0]*15
Output=np.zeros((10,10))
# calculate RMSE
Train_score=(mean_squared_error(History_output,MLP_History_output))
Test_score=(mean_squared_error(Forecast_output,MLP_Forecast_output))
print('Training RMSE is %f'%(Train_score))
print('Testing RMSE is %f'%(Test_score))
Output[0][0]=Train_score
Output[0][1]=Test_score

# calculate MAE
Train_score1=(mean_absolute_error(History_output,MLP_History_output))
Test_score1=(mean_absolute_error(Forecast_output,MLP_Forecast_output))
print('Training MAE is %f'%(Train_score1))
print('Testing MAE is %f'%(Test_score1))
Output[1][0]=Train_score1
Output[1][1]=Test_score1

# Calculate RMSe
Train_score=(mean_squared_error(History_output_inv,MLP_History_output_inv))
Test_score=(mean_squared_error(Forecast_output_inv,MLP_Forecast_output_inv))
print('Training RMSE is %f'%(Train_score))
print('Testing RMSE is %f'%(Test_score))
Output[2][0]=Train_score
Output[2][1]=Test_score

# calculate MAE
Train_score1=(mean_absolute_error(History_output_inv,MLP_History_output_inv))
Test_score1=(mean_absolute_error(Forecast_output_inv,MLP_Forecast_output_inv))
print('Training MAE is %f'%(Train_score1))
print('Testing MAE is %f'%(Test_score1))
Output[3][0]=Train_score1
Output[3][1]=Test_score1

# Calculate MAPE
Train_score1=(MAPE(History_output_inv,MLP_History_output_inv))
Test_score1=(MAPE(Forecast_output_inv,MLP_Forecast_output_inv))
print('Training MAPE is %f'%(Train_score1))
print('Testing MAPE is %f'%(Test_score1))
Output[4][0]=Train_score1
Output[4][1]=Test_score1

Output1=np.zeros((2,40))
Output1[0]=Forecast_output_inv[60:100].reshape(-1)
Output1[1]=MLP_Forecast_output_inv[60:100].reshape(-1)
Output1=Output1.transpose()

fig=plt.figure()
Scale_Xh=np.arange(1,Output1.shape[0]+1,1)
plt.plot(Scale_Xh,Output1[:,0],color='gray',label='Actual load',marker="v")
plt.plot(Scale_Xh,Output1[:,1],color='crimson',label='MLP Forecasted Load',marker="^") # linewidth=2,linestyle='--'
plt.xlabel('Time(Hours)', fontsize=18)
plt.ylabel('Load(MW)', fontsize=18)
plt.title('Zoomed Section')
plt.legend()
plt.show()

filename='Window1.xlsx'
complete_forecast_output=np.stack((Forecast_output,Forecast_output_inv,MLP_Forecast_output,MLP_Forecast_output_inv),axis=1)
complete_forecast_data=pd.DataFrame(data=complete_forecast_output,columns=['Actual Norm','Actual','MLP norm','MLP'])
complete_forecast_data.to_excel(filename,'Sheet2')
'''
# Residual Check for fit
Residuals=History_input_inv-MLP_History_output_inv
Residuals=Forecast_output_inv-MLP_Forecast_output_inv
Residuals=np.array(Residuals).reshape(-1)
plot_acf(Residuals)
plt.show()

Mean,variance=norm.fit(Residuals)
Output[10]=Mean
Output[11]=variance

# Check the normality usiong probaility plot Default normal
fig=plt.figure()
probplot(Residuals,plot=plt)

# Histogram of residuals
fig=plt.figure()
plt.hist(Residuals)
plt.show()

sns.kdeplot(Residuals)
Acorr=acorr_ljungbox(Residuals)
'''
'''
Meanse=[0]*10
for i in range(1,10):
    Meanse[i]=1
'''
