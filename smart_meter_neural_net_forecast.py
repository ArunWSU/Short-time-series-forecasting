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
    neural_model=[]
    
    def __init__(self,data):
        self.actual_time_series=data
    
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
             
    def find_time_related_features(self):
        self.time_related_features=pd.DataFrame(self.actual_time_series.index.dayofweek)
        self.time_related_features['Working day']=np.logical_not(((self.time_related_features==5)|(self.time_related_features==6))).astype('int')
        self.time_related_features['Hour']=pd.DataFrame(self.actual_time_series.index.hour)
        self.time_related_features=self.time_related_features[self.window_size:]
        self.time_related_features.reset_index(drop=True,inplace=True)
        self.time_related_features.rename(columns={'local_15min':'Day of week'},inplace=True)
        
    def individual_meter_forecast_features(self,input):
        individual_forecast_paper_features=np.zeros((16,1))
        x=np.array((-3,-6,-12,0))
        split_list=[input[a:] for a in x]
        for i,current_value in enumerate(split_list):
            individual_forecast_paper_features[i],individual_forecast_paper_features[i+4],individual_forecast_paper_features[i+8],individual_forecast_paper_features[i+12]=self.individual_meter_forecast_value_calculate(current_value)
        return individual_forecast_paper_features
    
    def individual_meter_forecast_value_calculate(self,previous_3): # Average, Maximum, Minimum, range of values
        return np.average(previous_3),np.amax(previous_3),np.amin(previous_3),np.ptp(previous_3)
    
    def neural_predict(self,FeaturePrepration,accuracy_select):
        self.accuracy_select=accuracy_select
        self.model_forecast_output=self.neural_model.predict(self.data_input)
        FeaturePrepration.neural_model=self.neural_model
        FeaturePrepration.model_forecast_output=FeaturePrepration.neural_model.predict(FeaturePrepration.data_input)
        
        # Model Performance Computation
        self.hist_perf_obj=ModelPerformance(self.data_output.reshape(-1,1),self.model_forecast_output.reshape(-1,1))
        self.hist_perf_obj.error_compute()
        FeaturePrepration.fore_perf_obj=ModelPerformance(FeaturePrepration.data_output.reshape(-1,1),FeaturePrepration.model_forecast_output.reshape(-1,1))
        FeaturePrepration.fore_perf_obj.error_compute()
        if(self.accuracy_select==1):
            self.hist_perf_obj.point_accuracy_compute()
            FeaturePrepration.fore_perf_obj.point_accuracy_compute()
                
    
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
          for x in range(1,self.window_size_max+1):
            # Input to MLP is of form No of samples, No of features
            self.window_size,MLPModelParameters.window_size=x,x
            self.prepare_neural_input_output(model_select,individual_paper_select)
            self.hist_inp_list[x-1],self.hist_out_list[x-1]=self.data_input,self.data_output
            MLPModelParameters.prepare_neural_input_output(model_select,individual_paper_select)
            self.fore_inp_list[x-1],self.fore_out_list[x-1]=MLPModelParameters.data_input,MLPModelParameters.data_output
            
#            mlp=MLPRegressor(hidden_layer_sizes=(9,),activation='logistic',solver='lbfgs',random_state=1)
#            self.mlp_fit_predict(mlp,MLPModelParameters,x)
            self.neural_fit()
            self.neural_predict(MLPModelParameters,0)
            self.current_iter_error=np.array((self.hist_perf_obj.MSE,self.hist_perf_obj.MAE,self.hist_perf_obj.MAPE,MLPModelParameters.fore_perf_obj.MSE,MLPModelParameters.fore_perf_obj.MAE,MLPModelParameters.fore_perf_obj.MAPE))
            self.Metrics_output_window_select.append(self.current_iter_error)
          
          self.Metrics_output_window_select1=pd.DataFrame(data=self.Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
          self.Metrics_output_window_select1.to_excel('Window_check.xlsx') 
            
    # CHOICE OF NUMBER OF NEURONS FOR HIDDEN LAYER
    def neuron_select(self,MLPModelParameters,window_size):
        self.data_input,self.data_output=self.hist_inp_list[window_size-1],self.hist_out_list[window_size-1]
        MLPModelParameters.data_input,MLPModelParameters.data_output=self.fore_inp_list[window_size-1],self.fore_out_list[window_size-1]
       
        self.Metrics_output_neuron_select=[]
        for x in range(1,self.neuron_number_max+1):
#            mlp=MLPRegressor(hidden_layer_sizes=(x,),activation='logistic',solver='lbfgs',random_state=1)
#            self.mlp_fit_predict(mlp,MLPModelParameters,x)
            self.neural_fit()
            self.neural_predict(MLPModelParameters,0)
            self.Metrics_output_neuron_select.append(self.current_iter_error)
            
        self.Metrics_output_neurons_select1=pd.DataFrame(data=self.Metrics_output_neuron_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
        self.Metrics_output_neurons_select1.to_excel('Neuron_check.xlsx') 
    
    def neural_fit(self):
        print("MLP")
        mlp=MLPRegressor(hidden_layer_sizes=(10,),activation='logistic',solver='lbfgs',random_state=1)
        self.neural_model=mlp
        self.neural_model.fit(self.data_input,self.data_output)
        
    def mlp_fit_predict(self,mlp,MLPModelParameters,x):
        mlp.fit(self.data_input,self.data_output)
        
        # Predict the outputs
        self.model_forecast_output=mlp.predict(self.data_input).reshape(-1,1)
        MLPModelParameters.model_forecast_output=mlp.predict(self.data_input).reshape(-1,1)
        
        hist_perf_obj=ModelPerformance(self.data_output,self.model_forecast_output)
        fore_perf_obj=ModelPerformance(MLPModelParameters.data_output,MLPModelParameters.model_forecast_output)
        
        hist_perf_obj.error_compute()
        fore_perf_obj.error_compute()
        
        self.current_iter_error=np.array((hist_perf_obj.MSE,hist_perf_obj.MAE,hist_perf_obj.MAPE,fore_perf_obj.MSE,fore_perf_obj.MAE,fore_perf_obj.MAPE))

class LSTMModelParameters(FeaturePreparation):
    
    def neural_fit(self):
        print("LSTM Model")
        self.LSTM_model_param()
    
    def LSTM_model_param(LSTMModelParameters):
        # Train LSTM model
        model=Sequential()
        model.add(LSTM(14,input_shape=(LSTMModelParameters.data_input.shape[1],LSTMModelParameters.data_input.shape[2])))
        model.add(Dense(1)) # First argument specifies the output
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(LSTMModelParameters.data_input,LSTMModelParameters.data_output,epochs=5,batch_size=1,verbose=2)
        LSTMModelParameters.neural_model=model
        
        
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
    def __init__(self,annual_data,forecast_size):
         AnomalyDetect.__init__(self,forecast_size)
         GaussianThresholds.annual_data=annual_data
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
        
def obj_create(annual_data,start_date,end_date,model_select,individual_paper_select,*var_args):
    data=annual_data[start_date:end_date]
#    a=[var_args[i] for i in range(len(var_args))]
    if(model_select==1):
        hist_object=MLPModelParameters(data,var_args[1],var_args[2]) # use parameter
    else:
        hist_object=LSTMModelParameters(data)
    hist_object.window_size=var_args[0]
    hist_object.find_time_related_features()
    hist_object.prepare_neural_input_output(model_select,individual_paper_select)  
    return hist_object
#%%
'''
# history vector    
#with cd('C:/Users/WSU-PNNL/Desktop/Data-pec'):
#with cd('C:/Users/Arun Imayakumar/Desktop/Pecan street data'):
#     Annual=pd.read_csv("3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) # Annual_data.idxmax() # Annual_data.idxmin()
#     Annual_complete=pd.read_csv("3967_data_2015_2018_all.csv",header=0,index_col=1,parse_dates=True)   
#Annual=Annual_complete.furnace1
#Annual=Annual.resample('H').asfreq()

#Specify MLP regressor model
x=int(input("Enter 1. MLP 2. LSTM"))
model_select=1
individual_paper_select=0
window_size_final=5
transform=0

variable_args=window_size_final
if model_select==1:
        window_size_max=int(input('Enter the maximum window size:'))
        neuron_number_max=int(input('Enter the maximum number of neurons:'))
        hist_object=obj_create(Annual,'2017-01-02','2017-01-08',model_select,individual_paper_select,window_size_final,window_size_max,neuron_number_max)
        fore_object=obj_create(Annual,'2017-01-09','2017-01-15',model_select,individual_paper_select,window_size_final,window_size_max,neuron_number_max)
else:
    hist_object=obj_create(Annual,'2017-01-02','2017-01-08',0,0,5)
    fore_object=obj_create(Annual,'2017-01-09','2017-01-15',0,0,5)
    

#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
        
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))
        
hist_object.neural_fit()
hist_object.window_size_select(fore_object)
hist_object.neuron_select(fore_object,5)
hist_object.neural_predict(fore_object,1)

# plot visualization
#individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
individual_plot_labels=['Actual Jan 2','MLP forecast']
#individual_plot_labels=['Actual Jan 2','LSTM forecast']
fig_labels=['Training Set','Time(Datapoint(15min))','Load(KW)']
#plot_list=[annual_data_series[140:170]]
plot_list=[hist_object.data_output[0:97],hist_object.model_forecast_output[0:97]]#97:193 history_output_old[0:97]
save_plot_name='Try 2'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
plot_list=[fore_object.data_output[0:97],fore_object.model_forecast_output[0:97]]
individual_plot_labels[0]='Actual Jan 9'
fig_labels[0]='Testing dataset'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
'''