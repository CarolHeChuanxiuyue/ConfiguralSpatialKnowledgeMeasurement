# Carol He: carol.hcxy@gmail.com
import pathlib

from desktop_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd
import csv


class WayFindingTrajectory:
    output_type = [('Subject', 'i4'), ('gender', 'i4'),('Order','i4'),('TrialID','i4'), ('Time', 'f16'), ('X', 'f4'),
                  ('Z', 'f4'),('Orientation','f4')]
    processed_output_type = [('Subject', 'i4'), ('Order','i4'),('TrialID','i4'), ('Time', 'f16'), ('X', 'f4'),
                  ('Z', 'f4'),('X_d', 'i4'),('Z_d','i4')]

    def __init__(self, trial_info,directory, suffix=".txt"):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._output_array = np.empty(0, dtype=self.output_type)
        self._processed_array = np.empty(0, dtype=self.processed_output_type)
        self._trial_info = np.loadtxt(trial_info, delimiter=",", skiprows=1)
        self._xgrid = np.linspace(3, 225, 12)
        self._ygrid = np.linspace(0, 225, 12)
        self._xctr = self._xgrid[:-1] + np.diff(self._xgrid)/2
        self._yctr = self._ygrid[:-1] + np.diff(self._ygrid)/2
        self._xctr = self._xctr.astype(int)
        self._yctr = self._yctr.astype(int)
    
    def read_shortcutting_data(self):
        for file in self._files_paths:

            array = self.process_each_file(file)
            self._output_array = np.append(self._output_array,np.array(list(map(tuple,array)),dtype=self.output_type))

        pass

    @staticmethod
    def process_each_file(file):

        with open(file) as f:
            array = []
            dataPieces = 0
            currentOrder = 0
            for row in f:
                if dataPieces < 6:
                    if row.startswith("ParticipantNo"):
                        partNo = row.split()[1]
                        dataPieces += 1
                    if row.startswith("ParticipantGen"):
                        partGen = row.split()[1]
                        dataPieces += 1
                    if row.startswith("Stressor"):
                        Stressor = row.split()[1]
                        #outputFile.writerow(["Date:", date])
                        dataPieces += 1
                    if row.startswith("DSPType"):
                        DSPType = row.split()[1]
                        #outputFile.writerow(["AltExperiment:", altEx])
                        dataPieces += 1
                    if row.startswith("AutoExperiment"):
                        autoEx = row.split()[1]
                        #outputFile.writerow(["AutoExperiment:", autoEx])
                        dataPieces += 1
                    if row.startswith("Encoding Tours"):
                        EnTour = row.split()[1]
                        dataPieces += 1
                elif (row.startswith("!!") & ("Learning" not in row) & ("Training" not in row)):
                    currentTrial = int(row.split("_")[2])
                    currentOrder += 1
                elif ("box" in row):
                    time = row.split(",")[0].split(" ")[0][:-1]
                    x = row.split(",")[0].split(" ")[2]
                    z = row.split(",")[1]
                    orientation = row.split(",")[2]
                    array.append([partNo,partGen,currentOrder,currentTrial,time,x,z,orientation])
        
        return array

        
    def clean_up_trajectory(self):
        tmp = self.get_raw_dataframe()
        tmp['X_d'] = pd.cut(tmp.X, self._xgrid, labels=self._xctr).astype(np.float16)
        tmp['Z_d'] = pd.cut(tmp.Z, self._ygrid, labels=self._yctr).astype(np.float16)
        tmp = tmp.round({'Time': 0}).drop_duplicates(subset=['Subject','TrialID','Time','X_d','Z_d'])
        grouped = tmp[['Subject','Order','TrialID','Time','X','Z','X_d','Z_d']].groupby(['Subject','TrialID'])
        for key, gp in grouped:
            array = np.array(gp)
            processed_array = self.process_trial(array)
            self._processed_array = np.append(self._processed_array,np.array(list(map(tuple,processed_array)),dtype=self.processed_output_type))

        pass

    
    def process_trial(self,array):
        sub_id = array[0][0]
        trial_order = array[0][1]
        trial_id = array[0][2]
        array = array[:,3:]
        array = self.check_start_end(array,trial_id)
        array = np.insert(array, 0, sub_id, axis=1)
        array = np.insert(array, 1, trial_order, axis=1)
        array = np.insert(array,2,trial_id,axis=1)
        
        return array

    def check_start_end(self,array,level):
        trial = self.trial_info(level)
        
        startright = 0
        startindex = 0
            
        for i in range(int(len(array)/4)):
            if np.linalg.norm([trial[0]-array[i][3],trial[1]-array[i][4]])<0.01:
                startright = 1
                startindex = i
        
        if startright == 0:
            array = np.insert(array,0,np.array([array[0][0],trial[0],trial[1],trial[0],trial[1]]),axis=0)
        else: array = array[startindex:,:]
        
        endright = 0
        endindex = 0
        for i in range(len(array)-int(len(array)/4),len(array)):
            if np.linalg.norm([trial[2]-array[i][3],trial[3]-array[i][4]])<0.01:
                endright = 1
                endindex = i
                break
        
        if endright == 0:
            array = np.insert(array,len(array),np.array([array[len(array)-1][0],trial[2],trial[3],trial[2],trial[3]]),axis=0)
        else:
            array = array[:endindex+1,:]
                
        return array
    
    def trial_info(self,level_id):
        start_x = self._trial_info[self._trial_info[:,0]==level_id,1][0]
        start_y = self._trial_info[self._trial_info[:,0]==level_id,2][0]
        end_x = self._trial_info[self._trial_info[:,0]==level_id,3][0]
        end_y = self._trial_info[self._trial_info[:,0]==level_id,4][0]
        return [start_x, start_y, end_x, end_y]
        
    def get_raw_dataframe(self):
        return pd.DataFrame(self._output_array)
    
    def save_raw(self, filename):
        self.get_raw_dataframe().to_csv(filename,index=False)
        pass
    
    def get_processed_dataframe(self):
        return pd.DataFrame(self._processed_array)

    def save_processed(self,filename):
        self.get_processed_dataframe().to_csv(filename,index=False)
        pass
