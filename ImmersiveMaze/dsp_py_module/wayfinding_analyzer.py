# rongfei@ucsb.edu; carol.hcxy@gmail.com
import pathlib

from dsp_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd


class WayFindingAnalyzer:
    
    euclidean_distance_data_type = [('ParticipantID', 'i4'), ('TrialNumber', 'i4'), 
                        ('LevelDistanceTraveled', 'f4')]

    def __init__(self, cleaned_up_csv,directory, suffix=".csv"):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._input_df = pd.read_csv(cleaned_up_csv)
        self._euclidean_distance_array = np.empty(0, dtype=self.euclidean_distance_data_type)
        self._output_df = pd.DataFrame()
    
    def analyze(self):
        for file in self._files_paths:
            array = DataStorage.get_array(file)

            for level_id in np.unique(array[:, 1]):
                self._euclidean_distance_array = np.append(self._euclidean_distance_array, np.array([(
                    self.get_participant_id(array),
                    level_id,
                    self.euclidean_distance(array,level_id),
                )], dtype=self.euclidean_distance_data_type))
        
        input_df = self._input_df.groupby(['SubjectNum','TrialNum'])[['X_d','Z_d','X','Z']].apply(self.total_moving_distance).reset_index()
        self._output_df = pd.DataFrame(self._euclidean_distance_array).merge(input_df, how = 'left', left_on=['ParticipantID','TrialNumber'],right_on = ['SubjectNum','TrialNum']).drop(['SubjectNum','TrialNum'],axis=1)
        
        pass
    

    def total_moving_distance(self,df):
        
        grid_distance = self.moving_distance(np.array(df[['X_d','Z_d']]))
        human_distance = self.moving_distance(np.array(df[['X','Z']]))
        
        return pd.Series([grid_distance,human_distance]).rename({0:'grid_distance',1:"human_distance"})
    
    
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
        
    
    @staticmethod
    def euclidean_distance(array, level_id):
        
        array = array[array[:,1] == level_id]
        array = array[:, [3, 5]]
        
        return WayFindingAnalyzer.moving_distance(array)

    
    @staticmethod
    def get_participant_id(array):
        return int(array[(0,0)])
    
