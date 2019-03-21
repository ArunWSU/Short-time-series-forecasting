#%% IMPORT FILES
import FeatureNeuralfile
import Plotstorefile

from contextlib import contextmanager
import os
import pandas as pd
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
#Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) # Annual_data.idxmax() # Annual_data.idxmin()
#Annual_complete=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/3967_data_2015_2018_all.csv",header=0,index_col=1,parse_dates=True)
#Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/Low freq REDD data/low_freq/house_1/channel_1.dat",delimiter='\s+',header=None,names=['mains power'],index_col=0) #0-11706 8895
#Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/Electricity_P.csv",header=0,index_col=0) # Annual_data.idxmax() # Annual_data.idxmin()
#Annual=pd.read_csv("C:/Electricity_P.csv",header=0,index_col=0)
#Annual_data_MHE=Annual['MHE']
##Annual=Annual_complete.furnace1
##Annual=Annual.resample('H').asfreq()
#Annual.index=pd.to_datetime(Annual.index,unit='s')
#Annual_data_15min=Annual.resample('0.25H').asfreq()
#Annual_data_15min=Annual_data_15min.fillna(0)
#Specify MLP regressor model
#x=int(input("Enter 1. MLP 2. LSTM"))

# AMPD dataset
Annual=Annual_data_15min
Annual=Annual/1000
# AMPD dataset
hist_start_date='2012-04-07'
hist_end_date='2012-04-13'
fore_start_date='2012-04-14'
fore_end_date='2012-04-20'

#hist_start_date='2017-01-02'
#hist_end_date='2017-01-08'
#fore_start_date='2017-01-09'
#fore_end_date='2017-01-15'
model_select=1
individual_paper_select=0
window_size_final=5
mlp_parm_determination=1
scale_input=0

if model_select==1:
        if(mlp_parm_determination==1):
            window_size_max=40
            neuron_number_max=40
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max) 
            hist_object.window_size_select(fore_object)
            hist_object.neuron_select(fore_object,5)
        else:
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input) 
            hist_object.neural_fit(7)
else:
    hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input)
    hist_object.neural_fit()
    fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input)
    

#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
        
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))

hist_object.accuracy_select=1
hist_object.neural_predict(fore_object)

# plot visualization
#individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
individual_plot_labels=['Actual Jan 2','MLP forecast']
#individual_plot_labels=['Actual Jan 2','LSTM forecast']
fig_labels=['Training Set','Time(Datapoint(15min))','Load(KW)']
#plot_list=[annual_data_series[140:170]]
plot_list=[hist_object.data_output[0:97],hist_object.model_forecast_output[0:97]]#97:193 history_output_old[0:97]
histplotobj=Plotstorefile.Plotstore(individual_plot_labels,fig_labels,plot_list) #'Try 2'
histplotobj.plot_results()
plot_list=[fore_object.data_output[0:97],fore_object.model_forecast_output[0:97]]
individual_plot_labels[0]='Actual Jan 9'
fig_labels[0]='Testing dataset'
foreplotobj=Plotstorefile.Plotstore(individual_plot_labels,fig_labels,plot_list) #'Try 2'
foreplotobj.plot_results()
'''
'''