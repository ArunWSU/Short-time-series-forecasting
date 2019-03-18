import statistics
import numpy as np

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