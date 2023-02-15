# rongfei@ucsb.edu; carol.hcxy@gmail.com
import pathlib

from desktop_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd


class WayFindingAnalyzer:
    

    def __init__(self, cleaned_up_csv):
        self._input_df = pd.read_csv(cleaned_up_csv)
        self._output_df = pd.DataFrame()
    
    def analyze(self):
        self._output_df = self._input_df.groupby(['Subject','TrialID'])[['Order','Time','X_d','Z_d','X','Z']].apply(self.trial_agg).reset_index()
        
        pass
    

    def trial_agg(self,df):
        
        order = df.Order.max()
        time = df.Time.max()
        grid_distance = self.moving_distance(np.array(df[['X_d','Z_d']]))
        human_distance = self.moving_distance(np.array(df[['X','Z']]))
        
        return pd.Series([order,time,grid_distance,human_distance]).rename({0:'Order',1:'Time',2:'grid_distance',3:"human_distance"})
    
    
    def get_dataframe(self):
        return self._output_df

    def save(self, filename):
        self.get_dataframe().to_csv(filename,index=False)
        pass
    
    @staticmethod
    def moving_distance(array):
        
        difference = np.diff(array, axis=0)  # difference between column (vector difference)
        norm = np.linalg.norm(difference, axis=1)  # get magnitudes of row vectors
        
        return np.sum(norm).round(2)
    
