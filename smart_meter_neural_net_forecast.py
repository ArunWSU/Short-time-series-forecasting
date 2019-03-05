#%% IMPORT FILES
import numpy as np
import pandas as pd
import statistics
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from scipy.stats import norm
import seaborn as sns

# Fixing the random number for neural net
np.random.seed(1)
# FUNCTIONS AND CLASS DEFINITIONS
# Creates the input and output data for time series forecasting
# Model select 1.MLP 2.LSTM

class FeaturePreparation:
    actual_time_series,time_series_values=[],[] # Available attributes of class
    data_input,data_output,model_forecast_output=[],[],[]
    time_related_features=[]
    window_size=[]
    
    def __init__(self,data):
        self.actual_time_series=data
        
    def find_time_related_features(self):
        self.time_related_features=pd.DataFrame(self.actual_time_series.index.dayofweek)
        self.time_related_features['Working day']=np.logical_not(((self.time_related_features==5)|(self.time_related_features==6))).astype('int')
        self.time_related_features['Hour']=pd.DataFrame(self.actual_time_series.index.hour)
        self.time_related_features=self.time_related_features[self.window_size:]
        self.time_related_features.reset_index(drop=True,inplace=True)
        self.time_related_features.rename(columns={'local_15min':'Day of week'},inplace=True)
        
    def prepare_neural_input_output(self,model_select,individual_paper_select):
        self.time_series_values=self.actual_time_series.values
        self.data_input,self.data_output=[],[]
        for i in range(0,len(self.time_series_values)-self.window_size,1):
               last_lag_data=self.time_series_values[i:i+self.window_size]
               if(individual_paper_select==1):
                   last_lag_data=np.vstack((last_lag_data,self.individual_meter_forecast_features(last_lag_data).reshape(-1,1))) # Overloading might be possible variable args
               self.data_input.append(last_lag_data)
               self.data_output.append(self.time_series_values[i+self.window_size])# Y_t=history[5:]     
        self.data_input=np.array(self.data_input)
        self.data_output=np.array(self.data_output)
        if(model_select==1):
            self.data_input=self.data_input.reshape(-1,last_lag_data.shape[0])
            self.data_output=self.data_output.ravel()
            
    def individual_meter_forecast_features(self,input):
        individual_forecast_paper_features=np.zeros((16,1))
        x=np.array((-3,-6,-12,0))
        split_list=[input[a:] for a in x]
        for i,current_value in enumerate(split_list):
            individual_forecast_paper_features[i],individual_forecast_paper_features[i+4],individual_forecast_paper_features[i+8],individual_forecast_paper_features[i+12]=self.value_calculate(current_value)
        return individual_forecast_paper_features
    
    def value_calculate(self,previous_3):
        return np.average(previous_3),np.amax(previous_3),np.amin(previous_3),np.ptp(previous_3)
    
class MLPModelParameters(FeaturePreparation):
    # CHOICE OF WINDOW SIZE FOR FORECASTING      
    def __init__(self,data,window_size_max,neuron_number_max):
        self.window_size_max=window_size_max
        self.neuron_number_max=neuron_number_max
        FeaturePreparation.__init__(self,data)
         
    def window_size_select(self,MLPModelParameters): # self is hist_obj, MLPModelParameters refers to the fore_obj
          self.hist_inp_list,self.hist_out_list,self.fore_inp_list,self.fore_out_list=[0]*(self.window_size_max+3),[0]*(self.window_size_max+3),[0]*(self.window_size_max+3),[0]*(self.window_size_max+3)
          self.Metrics_output_window_select=[]
          
          # Data as function of window sizes
          for x in range(1,self.window_size_max):
            # Input to MLP is of form No of samples, No of features
            self.window_size,MLPModelParameters.window_size=x,x
            self.prepare_neural_input_output(model_select,individual_paper_select)
            self.hist_inp_list[x-1],self.hist_out_list[x-1]=self.data_input,self.data_output
            MLPModelParameters.prepare_neural_input_output(model_select,individual_paper_select)
            self.fore_inp_list[x-1],self.fore_out_list[x-1]=MLPModelParameters.data_input,MLPModelParameters.data_output
            
            mlp=MLPRegressor(hidden_layer_sizes=(9,),activation='logistic',solver='lbfgs',random_state=1)
            self.mlp_fit_predict(mlp,MLPModelParameters,x)
            self.Metrics_output_window_select.append(self.current_iter_error)
          
          self.Metrics_output_window_select1=pd.DataFrame(data=self.Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
          self.Metrics_output_window_select1.to_excel('Window_check.xlsx') 
            
    # CHOICE OF NUMBER OF NEURONS FOR HIDDEN LAYER
    def neuron_select(self,MLPModelParameters,window_size):
        self.data_input,self.data_output=self.hist_inp_list[window_size-1],self.hist_out_list[window_size-1]
        MLPModelParameters.data_input,MLPModelParameters.data_output=self.fore_inp_list[window_size-1],self.fore_out_list[window_size-1]
       
        self.Metrics_output_neuron_select=[]
        for x in range(1,self.neuron_number_max):
            mlp=MLPRegressor(hidden_layer_sizes=(x,),activation='logistic',solver='lbfgs',random_state=1)
            self.mlp_fit_predict(mlp,MLPModelParameters,x)
            self.Metrics_output_neuron_select.append(self.current_iter_error)
            
        self.Metrics_output_neurons_select1=pd.DataFrame(data=self.Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
        self.Metrics_output_neurons_select1.to_excel('Neuron_check.xlsx') 
          
    def mlp_fit_predict(self,mlp,MLPModelParameters,x):
        mlp.fit(hist_object.data_input,hist_object.data_output)
        
        # Predict the outputs
        self.model_forecast_output=mlp.predict(hist_object.data_input).reshape(-1,1)
        MLPModelParameters.model_forecast_output=mlp.predict(fore_object.data_input).reshape(-1,1)
        
        hist_perf_obj=ModelPerformance(self.data_output,self.model_forecast_output)
        fore_perf_obj=ModelPerformance(MLPModelParameters.data_output,MLPModelParameters.model_forecast_output)
        
        hist_perf_obj.error_compute()
        fore_perf_obj.error_compute()
        
        self.current_iter_error=np.array((hist_perf_obj.MSE,hist_perf_obj.MAE,hist_perf_obj.MAPE,fore_perf_obj.MSE,fore_perf_obj.MAE,fore_perf_obj.MAPE))

class LSTMModelParameters():
    
    def LSTM_model_param(LSTMModelParameters):
        # Train LSTM model
        model=Sequential()
        model.add(LSTM(14,input_shape=(LSTMModelParameters.data_input.shape[1],LSTMModelParameters.data_input.shape[2])))
        model.add(Dense(1)) # First argument specifies the output
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(LSTMModelParameters.data_input,LSTMModelParameters.data_output,epochs=500,batch_size=1,verbose=2)
        
        
class ModelPerformance():
    # To view the values used in error computation
    actual_list,forecast_list=[],[]
    data_output,model_forecast_output=[],[]
    
    def __init__(self,X,Y):
        self.data_output=X
        self.model_forecast_output=Y
                                                                                                                                                                                                         
    def rolling_window_error(self,forecast_output_actual,mlp_forecast_actual,i,performance_window_size):
        ModelPerformance.error_metrics=np.zeros((forecast_output_actual.shape[0]+3,3))
        if(performance_window_size==1):
            self.actual_values=forecast_output_actual[i] 
            self.forecast_values=mlp_forecast_actual[i]
            self.error_metrics[i][0],self.error_metrics[i][1],self.error_metrics[i][2]=self.error_compute(self.actual_values,self.forecast_values)
        else:
            end_index=i+1
            if(end_index>=performance_window_size):
             self.actual_values=forecast_output_actual[end_index-performance_window_size:end_index] 
             self.actual_list.append(self.actual_values)
             self.forecast_values=mlp_forecast_actual[end_index-performance_window_size:end_index] 
             self.forecast_list.append(self.forecast_values)
             self.error_metrics[i][0],self.error_metrics[i][1],self.error_metrics[i][2]=self.error_compute(self.actual_values,self.forecast_values)
    
    # returns MSE,MAE, MAPE       
    def error_compute(self):
        X,Y=self.data_output,self.model_forecast_output
        np.seterr(divide='ignore')
        self.MSE,self.MAE,self.MAPE=np.mean(np.power(abs(np.array(X)-np.array(Y)),2)),np.mean(abs(np.array(X)-np.array(Y))),np.mean(abs(np.array(X)-np.array(Y))/np.array(X))*100

    def point_accuracy_compute(self):
        X,Y=self.data_output,self.model_forecast_output
        Accurate_forecast_points=((((X > 1.5) & ((abs(X-Y)) < (X*0.10)))) | ((X < 1.5) & ((abs(X-Y)) < (0.10))))
        self.model_accuracy=np.sum(Accurate_forecast_points)/Accurate_forecast_points.shape[0]

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
    
## Does Line plot if length of different input matches
def plot_results(plot_list,individual_plot_labels,fig_labels,mark_select,save_plot,save_plot_name):
    no_datapoints=plot_list[0].size
    no_line_plots=len(plot_list)
    color_list=['crimson','gray','blue','green']
    if(mark_select==1):
        marker_list=["o","^","+","x",'*']
        line_style=['-','--',':','-.']
    else:
        marker_list=['None']*no_line_plots
        line_style=['-','--']*no_line_plots
    try:
        if(all(x.size==no_datapoints for x in plot_list)):
                plt.figure()
                X_scale=np.arange(1,no_datapoints+1,1)
                for i in range(no_line_plots):
                     plt.plot(X_scale,plot_list[i],color=color_list[i],label=individual_plot_labels[i],linewidth=4,marker=marker_list[i],linestyle=line_style[i])
                plt.title(fig_labels[0]) 
                plt.xlabel(fig_labels[1], fontsize=18)
                plt.ylabel(fig_labels[2], fontsize=18)
                plt.legend()
                plt.show()
                if(save_plot==1):
                    plt.savefig(save_plot_name)  
        else:
            raise Exception
    except Exception:
        print('Length mismatch among different vectors to plot')    
    
# writing outputs to excel or csv file
def file_store(input_data,excel_select,filename,index_name):
    column_list=['RMSE', 'MAE','MAPE']
    sheet_list=['sheet1','sheet2','sheet3','sheet4','sheet5','sheet6','sheet7','sheet8','sheet9','sheet10','sheet11']
    if(excel_select==1):
      filename_with_ext=''.join([filename,'.xlsx'])  
      writer=ExcelWriter(filename_with_ext)
      for n,df in enumerate(input_data):
          output_dataframe=pd.DataFrame(data=df,columns=column_list)
          output_dataframe.index.name=index_name[n]
          output_dataframe.to_excel(writer,sheet_list[n])
      writer.save()   
    else:
      output_dataframe=pd.DataFrame(data=input_data,columns=column_list)
      output_dataframe.index.name=index_name
      filename_with_ext=''.join([filename,'.csv'])  
      output_dataframe.to_csv(filename_with_ext)  

# Checking for normal fit
def norm_fit(annual_data,plot_select):
    annual_data_series=pd.Series(annual_data)
    if(plot_select==1):
        plt.figure()
        plt.title('Histogram of annual data',fontsize=14)
        plt.hist(annual_data_series)
    mean_data=statistics.mean(annual_data_series)
    stddev_data=statistics.stdev(annual_data_series)
    m,s=norm.fit(annual_data_series)
    print('Mean %f Stddev %f using normal fit'%(m,s))
    print('Mean %f Stddev %f  using statistics'%(mean_data,stddev_data))
    return m,s

from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
#%%
# history vector    
#with cd('C:/Users/WSU-PNNL/Desktop/Data-pec'):
#with cd('C:/Users/Arun Imayakumar/Desktop/Pecan street data'):
#     Annual=pd.read_csv("3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) # Annual_data.idxmax() # Annual_data.idxmin()
#     Annual_complete=pd.read_csv("3967_data_2015_2018_all.csv",header=0,index_col=1,parse_dates=True)   
#Annual=Annual_complete.furnace1

#Specify MLP regressor model
window_size=5
model_select=1
individual_paper_select=0
history_start_date='2017-01-02' 
history_end_date='2017-01-08'
#history_end_date='2017-10-08'
history=Annual[history_start_date:history_end_date]
#history=history.resample('H').asfreq()

hist_object=MLPModelParameters(history) # use parameter
hist_object.window_size=window_size
hist_object.find_time_related_features()
hist_object.prepare_neural_input_output(model_select,individual_paper_select)
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
history_values=history.values# Approximately 0.8=9.5 months # Long way history.index=np.arange(0,len(history),1) history=history.values.reshape(-1,1)
transform=0
if(transform==1):
    history_old=history.copy()
    history=np.log(history)
    
# forecast vector
forecast_start_date='2017-01-09' 
forecast_end_date='2017-01-15'
#forecast_start_date='2017-10-09'
#forecast_end_date='2017-10-15'
forecast=Annual[forecast_start_date:forecast_end_date]
#forecast=forecast.resample('H').asfreq()
fore_object=MLPModelParameters(forecast)
fore_object.window_size=window_size
fore_object.find_time_related_features()
fore_object.prepare_neural_input_output(model_select,individual_paper_select)
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))
forecast=forecast.values.copy()
if(transform==1):
    forecast_old=forecast.copy()
    forecast=np.log(forecast)
#%% MODEL SELECTION and forecast 
'''
## CHOICE OF WINDOW SIZE FOR FORECASTING    
Metrics_output_window_select=np.zeros((20,6))

for x in range(1,15):
    # Input to MLP is of form No of samples, No of features
    window_size=x
    hist_object.data_input,hist_object.data_output=data_preparation(history,window_size,model_select)
    fore_object.data_input,fore_object.data_output=data_preparation(forecast,window_size,model_select)
    
    #  Specify MLP regressor model
    mlp=MLPRegressor(hidden_layer_sizes=(9,),activation='logistic',solver='lbfgs',random_state=1)
    mlp.fit(hist_object.data_input,hist_object.data_output)
    
    # Predict the outputs
    hist_object.model_forecast_output=mlp.predict(hist_object.data_input).reshape(-1,1)
    fore_object.model_forecast_output=mlp.predict(fore_object.data_input).reshape(-1,1)
    
    # Calculate the error metrics
    Metrics_output_window_select[x][0],Metrics_output_window_select[x][1],Metrics_output_window_select[x][2]=error_compute(hist_object.data_output,hist_object.model_forecast_output)
    Metrics_output_window_select[x][3],Metrics_output_window_select[x][4],Metrics_output_window_select[x][5]=error_compute(fore_object.data_output,fore_object.model_forecast_output)
   
Metrics_output_window_select1=pd.DataFrame(data=Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
Metrics_output_window_select1.to_excel('Window_check_pecan_31_Jan.xlsx')

## CHOICE OF NUMBER OF NEURONS FOR HIDDEN LAYER
Metrics_output_neurons_select=np.zeros((20,6))
window_size=4
hist_object.data_input,hist_object.data_output=data_preparation(history,window_size,model_select)
fore_object.data_input,fore_object.data_output=data_preparation(forecast,window_size,model_select)

for x in range(1,10):
    
    #  Specify MLP regressor model
    mlp=MLPRegressor(hidden_layer_sizes=(x,),activation='logistic',solver='lbfgs',random_state=1)
    mlp.fit(hist_object.data_input,hist_object.data_output)
    
    # Predict the outputs
    hist_object.model_forecast_output=mlp.predict(hist_object.data_input).reshape(-1,1)
    fore_object.model_forecast_output=mlp.predict(fore_object.data_input).reshape(-1,1)
    
    # Calculate the error metrics
    Metrics_output_neurons_select[x][0],Metrics_output_neurons_select[x][1],Metrics_output_neurons_select[x][2]=error_compute(hist_object.data_output,hist_object.model_forecast_output)
    Metrics_output_neurons_select[x][3],Metrics_output_neurons_select[x][4],Metrics_output_neurons_select[x][5]=error_compute(fore_object.data_output,fore_object.model_forecast_output)
    
Metrics_output_neurons_select1=pd.DataFrame(data=Metrics_output_neurons_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
Metrics_output_neurons_select1.to_csv('neuron_check.csv')     
'''
'''
# Train
# Choice of window size for window based forecasting
window_size=4 
model_select=0
hist_object.data_input,hist_object.data_output=data_preparation(history,window_size,model_select)

# Train LSTM model
model=Sequential()
model.add(LSTM(14,input_shape=(hist_object.data_input.shape[1],hist_object.data_input.shape[2])))
model.add(Dense(1)) # First argument specifies the output
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(hist_object.data_input,hist_object.data_output,epochs=500,batch_size=1,verbose=2)

#Make predictions
lstm_history_output=model.predict(hist_object.data_input)
fore_object.data_input,fore_object.data_output=data_preparation(history,window_size,model_select)
lstm_forecast_output=model.predict(fore_object.data_input)

# plot visualization
#individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
individual_plot_labels=['Actual Jan 2','LSTM forecast']
fig_labels=['Training Set','Time(Datapoint(15min))','Load(KW)']
#plot_list=[annual_data_series[140:170]]
plot_list=[hist_object.data_output[0:97],lstm_history_output[0:97]]#97:193 history_output_old[0:97]
save_plot_name='Try 2'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
plot_list=[fore_object.data_output[0:97],lstm_forecast_output[0:97]]
individual_plot_labels[0]='Actual Jan 9'
fig_labels[0]='Testing dataset'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
history_MSE,history_MAE,history_MAPE=error_compute(hist_object.data_output,lstm_history_output)
forecast_MSE,forecast_MAE,forecast_MAPE=error_compute(fore_object.data_output,lstm_forecast_output)
'''

#hist_object.data_input,hist_object.data_output=data_preparation(history,window_size,model_select)
#hist_object.data_input,hist_object.data_output=data_prep_feature(history,window_size,model_select,Time_related_features_hist)
mlp=MLPRegressor(hidden_layer_sizes=(10,),activation='logistic',solver='lbfgs',random_state=1)

# LBFGS for small samples No batch size Learning rate for SGD
mlp.fit(hist_object.data_input,hist_object.data_output)
hist_object.model_forecast_output=mlp.predict(hist_object.data_input)
if(transform==1):
    history_output_old=hist_object.data_output.copy()
    hist_object.data_output=np.exp(hist_object.data_output)
    mlp_history_output_old=hist_object.model_forecast_output.copy()
    hist_object.model_forecast_output=np.exp(hist_object.model_forecast_output)
#history_MSE,history_MAE,history_MAPE=error_compute(hist_object.data_output,hist_object.model_forecast_output)
hist_absolute_error=abs(hist_object.data_output-hist_object.model_forecast_output)
hist_APE=hist_absolute_error/hist_object.data_output
history_data_calc=np.hstack((hist_object.data_output.reshape(-1,1),hist_object.model_forecast_output.reshape(-1,1),hist_absolute_error.reshape(-1,1),hist_APE.reshape(-1,1)))
history_df=pd.DataFrame(data=history_data_calc,columns=['History','MLP_History','Abs_error','APE'])
hist_perf_obj=ModelPerformance(hist_object.data_output,hist_object.model_forecast_output)
hist_perf_obj.error_compute()
hist_perf_obj.point_accuracy_compute()
    
    
#df.to_excel('MAPE_check_relu.xlsx')

# Test
#fore_object.data_input,fore_object.data_output=data_preparation(history,window_size,model_select)
#fore_object.data_input,fore_object.data_output=data_prep_feature(history,window_size,model_select,Time_related_features_fore)
fore_object.model_forecast_output=mlp.predict(fore_object.data_input)
if(transform==1):
    forecast_output_old=fore_object.data_output.copy()
    fore_object.data_output=np.exp(fore_object.data_output)
    mlp_forecast_output_old=fore_object.model_forecast_output.copy()
    fore_object.model_forecast_output=np.exp(fore_object.model_forecast_output)
#forecast_MSE,forecast_MAE,forecast_MAPE=error_compute(fore_object.data_output,fore_object.model_forecast_output)
fore_absolute_error=abs(fore_object.data_output-fore_object.model_forecast_output)
fore_APE=(fore_absolute_error/fore_object.data_output)*100
fore_data_calc=np.hstack((fore_object.data_output.reshape(-1,1),fore_object.model_forecast_output.reshape(-1,1),fore_absolute_error.reshape(-1,1),fore_APE.reshape(-1,1)))
fore_df=pd.DataFrame(data=history_data_calc,columns=['Forecast','MLP_Forecast','Abs_error','APE'])
fore_perf_obj=ModelPerformance(fore_object.data_output,fore_object.model_forecast_output)
fore_perf_obj.error_compute()
fore_perf_obj.point_accuracy_compute()


# plot visualization
#individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
individual_plot_labels=['Actual Jan 2','MLP forecast']
fig_labels=['Training Set','Time(1 Datapoint-1 hour))','Load(KW)']
#plot_list=[annual_data_series[140:170]]
plot_list=[hist_object.data_output[0:97],hist_object.model_forecast_output[0:97]]#97:193 history_output_old[0:97]
save_plot_name='Try 2'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
plot_list=[fore_object.data_output[0:97],fore_object.model_forecast_output[0:97]]
individual_plot_labels[0]='Actual Jan 9'
fig_labels[0]='Testing dataset'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)

'''      
# history vector
#Annual=pd.read_csv("3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) # Annual_data.idxmax() # Annual_data.idxmin()
year_list=['2015','2016','2017']
deviation_error_metrics=np.zeros((5,4))
for y in range(3):
    annual_data=Annual[year_list[y]].use.values
    annual_data_series=pd.Series(annual_data)
    annual_data_series_diff=annual_data_series.diff(periods=1).dropna()
    deviation_error_metrics[0][y],deviation_error_metrics[1][y]=norm_fit(annual_data_series_diff,0)
    deviation_error_metrics[2][y],deviation_error_metrics[3][y] =min(annual_data_series_diff),max(annual_data_series_diff)
    upper_threshold=0+3*0.82
    lower_threshold=0-3*0.82

mask=((annual_data_series_diff>lower_threshold)&(annual_data_series_diff<upper_threshold)).astype(int)
outlier_index=mask[mask==0]
outlier_index.to_excel('Flagged Outliers.xlsx')
mask_1=((abs(annual_data_series_diff)>upper_threshold)).astype(int)

# 1. Normal 2.Retraining 3. Missing measurements, Singlepoint, Collective outliers 4. Effect of noise
actual_forecast=forecast.copy()
window_index=0
use_case=1
if(use_case==1):
    forecast_output_actual=forecast[window_size:].copy()
    mlp_forecast_actual=np.zeros(forecast_output_actual.shape[0]) 
    adam_forecast_output=np.zeros(forecast.shape[0]).reshape(-1,1)
    adam_forecast_output[0:window_size]=forecast[0:window_size].reshape(-1,1).copy() 
    threshold_violations=mlp_forecast_actual.copy()
    model_mlp_obj=ModelPerformance(forecast_output_actual.shape[0])
    performance_window_size=2
    max_window_size=11
    overall_error_metrics=np.zeros((max_window_size,3))
    rolling_error_list,actual_list,forecast_list=[0]*(max_window_size-1),[0]*max_window_size,[0]*max_window_size
    for y in range(1,max_window_size,1):
        window_index=0
        performance_window_size=y
        anomaly_detection_method=1 
        no_of_forecast_points=forecast_output_actual.shape[0]
        gaus_obj=GaussianThresholds(annual_data,forecast_output_actual.shape[0])
        
        for i in range(0,no_of_forecast_points,1):
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
        actual_list[y-1] =model_mlp_obj.actual_list
        forecast_list[y-1]=model_mlp_obj.forecast_list
        model_mlp_obj.actual_list,model_mlp_obj.forecast_list=[],[]
        rolling_error_list[y-1]=model_mlp_obj.error_metrics
        model_mlp_obj.error_metrics=np.zeros((forecast_output_actual.shape[0]+3,3))
        overall_error_metrics[y][0],overall_error_metrics[y][1],overall_error_metrics[y][2]=error_compute(forecast_output_actual,mlp_forecast_actual)
         # plot visulaiztion with zeros
    individual_plot_labels=['ActuaL_load','MLP Forecasted Load','Actual load without zeros']
    fig_labels=['Testing Set','Time(Hours)','Load(KW)']
    plot_list=[forecast_output_actual,mlp_forecast_actual]
    save_plot_name='Try 2'
    plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
         
    # Storing in files
    index_names=['Rolling_window_forecast']*(max_window_size+2)
    file_store(rolling_error_list,1,'Effect_rolling_window_pecan',index_names)

elif(use_case==3):
    forecast_end_date='2017-10-25'
    zero_start_index=19
    no_simulated_zeros=48
    overall_error_metrics,error_zero_interval,error_post_zero_interval=np.zeros((no_simulated_zeros+1,3)),np.zeros((no_simulated_zeros+1,3)),np.zeros((no_simulated_zeros+1,3))
    mlp_zero_forecast_list,forecast_zero_list=[],[]
    mlp_forecast_list,forecast_list=[],[]
#    y=10
    for y in range(1,no_simulated_zeros+1,1):
        forecast_output_actual=forecast[window_size:].copy()
        forecast_output_actual_unmodified=forecast_output_actual.copy()
        #zero_end_index=36
        zero_end_index=zero_start_index+y
        forecast_output_actual[zero_start_index:zero_end_index]=0
        forecast_list.append(forecast_output_actual)
        no_of_forecast_points=forecast_output_actual.shape[0]
        mlp_forecast_actual=np.zeros(no_of_forecast_points)
        mlp_forecast_list.append(mlp_forecast_actual)
        window_index=0
        adam_forecast_output=np.zeros(forecast.shape[0]).reshape(-1,1)
        adam_forecast_output[0:window_size]=forecast[0:window_size].reshape(-1,1).copy() 
        threshold_violations=mlp_forecast_actual.copy()
        
        model_mlp_obj=ModelPerformance(forecast_output_actual.shape[0])
        performance_window_size=1
        anomaly_detection_method=1
        
        gaus_obj=GaussianThresholds(annual_data,forecast_output_actual.shape[0])
        
        for i in range(0,no_of_forecast_points,1):
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
     
        # Simpler approach using mask
        zero_index=np.zeros((no_of_forecast_points,1))
        zero_index[zero_start_index:zero_end_index]=1
        mask=(zero_index==1).reshape(-1)
        mlp_forecast_zero_interval_mask,actual_val_zero_interval_mask=mlp_forecast_actual[mask],forecast_output_actual_unmodified[mask]
        mlp_forecast_actual_no_zeros_mask,forecast_output_actual_no_zeros_mask=mlp_forecast_actual[~mask],forecast_output_actual_unmodified[~mask]
        mlp_forecast_post_zeros,forecast_output_post_zeros=mlp_forecast_actual[zero_end_index:],forecast_output_actual_unmodified[zero_end_index:]
        
        overall_error_metrics[y][0],overall_error_metrics[y][1],overall_error_metrics[y][2]=error_compute(forecast_output_actual_no_zeros_mask,mlp_forecast_actual_no_zeros_mask)
        error_zero_interval[y][0],error_zero_interval[y][1],error_zero_interval[y][2]=error_compute(actual_val_zero_interval_mask,mlp_forecast_zero_interval_mask)
        error_post_zero_interval[y][0],error_post_zero_interval[y][1],error_post_zero_interval[y][2]=error_compute(forecast_output_post_zeros,mlp_forecast_post_zeros)
        # To check individual forecasts
        mlp_zero_forecast_list.append(mlp_forecast_zero_interval_mask)
        forecast_zero_list.append(actual_val_zero_interval_mask)
    
#     #Storing in files
    performance_list=[error_zero_interval,overall_error_metrics,error_post_zero_interval]
    index_names=['Errors_zero_interval','Overall_performance_metrics','Post_error_metrics']
    file_store(performance_list,1,'Errors_missing_measurements_post_pecan_after_check',index_names)
'''