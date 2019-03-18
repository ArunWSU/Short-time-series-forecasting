import numpy as np

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