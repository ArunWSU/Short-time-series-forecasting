#%% IMPORT FILES
from contextlib import contextmanager
import os
import FeatureNeuralfile
import Plotstorefile

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
#Annual=Annual.resample('H').asfreq()

#Specify MLP regressor model
#x=int(input("Enter 1. MLP 2. LSTM"))
model_select=1
individual_paper_select=0
window_size_final=5
transform=0

variable_args=window_size_final
if model_select==1:
        window_size_max=int(input('Enter the maximum window size:'))
        neuron_number_max=int(input('Enter the maximum number of neurons:'))
        hist_object=FeatureNeuralfile.obj_create(Annual,'2017-01-02','2017-01-08',model_select,individual_paper_select,window_size_final,window_size_max,neuron_number_max)
        hist_object.neural_fit(10)
        fore_object=FeatureNeuralfile.obj_create(Annual,'2017-01-09','2017-01-15',model_select,individual_paper_select,window_size_final,window_size_max,neuron_number_max)
else:
    hist_object=FeatureNeuralfile.obj_create(Annual,'2017-01-02','2017-01-08',0,0,5)
    hist_object.neural_fit()
    fore_object=FeatureNeuralfile.obj_create(Annual,'2017-01-09','2017-01-15',0,0,5)
    

#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
        
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))
        

hist_object.window_size_select(fore_object)
hist_object.neuron_select(fore_object,5)
#hist_object.accuracy_select=1
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