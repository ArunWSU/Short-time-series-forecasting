import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from scipy.stats import norm

class Plotstore:     
    def __init__(self,individual_plot_labels,fig_labels,plot_list,*args):
        self.individual_plot_labels=individual_plot_labels
        self.fig_labels=fig_labels
        self.plot_list=plot_list
        self.save_plot_name=''
        args_len=len(args)
        self.mark_select=1
        if(args_len==1):
            self.save_plot_name=args[0]
        
 
    ## Does Line plot if length of different input matches
    def plot_results(self):
        no_datapoints=self.plot_list[0].size
        no_line_plots=len(self.plot_list)
        color_list=['crimson','gray','blue','green']
        if(self.mark_select==1):
            marker_list=["o","^","+","x",'*']
            line_style=['-','--',':','-.']
        else:
            marker_list=['None']*no_line_plots
            line_style=['-','--']*no_line_plots
        try:
            if(all(x.size==no_datapoints for x in self.plot_list)):
                    plt.figure()
                    X_scale=np.arange(1,no_datapoints+1,1)
                    for i in range(no_line_plots):
                         plt.plot(X_scale,self.plot_list[i],color=color_list[i],label=self.individual_plot_labels[i],linewidth=4,marker=marker_list[i],linestyle=line_style[i])
                    plt.title(self.fig_labels[0]) 
                    plt.xlabel(self.fig_labels[1], fontsize=18)
                    plt.ylabel(self.fig_labels[2], fontsize=18)
                    plt.legend()
                    plt.show()
                    if not self.save_plot_name.strip():
                        plt.savefig(self.save_plot_name)  
            else:
                raise Exception
        except Exception:
            print('Length mismatch among different vectors to plot')    
        
    # writing outputs to excel or csv file
    def file_store(self,input_data,excel_select,filename,index_name):
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
