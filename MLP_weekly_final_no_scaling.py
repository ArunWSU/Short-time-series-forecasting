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
        line_style=['--']*no_line_plots
    try:
        if(all(x.size==no_datapoints for x in plot_list)):
                plt.figure()
                X_scale=np.arange(1,no_datapoints+1,1)
                for i in range(no_line_plots):
                     plt.plot(X_scale,plot_list[i],color=color_list[i],label=individual_plot_labels[i],linewidth=3,marker=marker_list[i],linestyle=line_style[i])
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
# plot visulaiztion with zeros
individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
fig_labels=['Training Set','Time(Hours)','Load(MW)']
plot_list=[history_output,mlp_history_output]
save_plot_name='Try 2'
plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
'''
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
window_index=0

# 1. Normal 2.Retraining 3. Missing measurements, Singlepoint, Collective outliers 4. Effect of noise
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
#    individual_plot_labels=['ActuaL_load','MLP Forecasted Load','Actual load without zeros']
#    fig_labels=['Testing Set','Time(Hours)','Load(MW)']
#    plot_list=[forecast_output_actual,mlp_forecast_actual]
#    save_plot_name='Try 2'
#    plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
         
    # Storing in files
    index_names=['Rolling_window_forecast']*(max_window_size+2)
    file_store(rolling_error_list,1,'Effect_rolling_window',index_names)

elif(use_case==3):
    zero_start_index=19
    no_simulated_zeros=48
    overall_error_metrics,error_zero_interval,error_post_zero_interval=np.zeros((no_simulated_zeros+1,3)),np.zeros((no_simulated_zeros+1,3)),np.zeros((no_simulated_zeros+1,3))
    mlp_zero_forecast_list,forecast_zero_list=[],[]
    mlp_forecast_list,forecast_list=[],[]
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
        error_post_zero_interval[y][0],error_post_zero_interval[y][1],error_post_zero_interval[y][2]=error_compute(mlp_forecast_post_zeros,forecast_output_post_zeros)
        # To check individual forecasts
        mlp_zero_forecast_list.append(mlp_forecast_zero_interval_mask)
        forecast_zero_list.append(actual_val_zero_interval_mask)
       
        
    # Storing in files
#    performance_list=[error_zero_interval,overall_error_metrics,error_post_zero_interval]
#    index_names=['Errors_zero_interval','Overall_performance_metrics','Post_error_metrics']
#    file_store(performance_list,1,'Errors_missing_measurements_post',index_names)
        
#    
#    # plot visulaiztion with zeros
#    individual_plot_labels=['ActuaL_load','MLP Forecasted Load','Actual load without zeros']
#    fig_labels=['Testing Set','Time(Hours)','Load(MW)']
#    plot_list=[forecast_output_actual,mlp_forecast_actual,forecast_output_actual_unmodified]
#    save_plot_name='Try 2'
#    plot_results(plot_list,individual_plot_labels,fig_labels,1,0,save_plot_name)
#    
#    # plotting just forecast vector
#    plot_list=[forecast_output_actual,mlp_forecast_actual]
#    fig_labels[0]='Forecasting set'
#    plot_results(plot_list,individual_plot_labels,fig_labels,0,0,save_plot_name)