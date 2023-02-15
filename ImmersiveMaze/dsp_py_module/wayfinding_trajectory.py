# Carol He: carol.hcxy@gmail.com
import pathlib

from dsp_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd


class WayFindingTrajectory:
    output_data_type = [('SubjectNum', 'i4'), ('TrialNum', 'i4'), ('Time', 'f4'), ('X', 'f4'),
                  ('Z', 'f4')]
    processed_output_type = [('SubjectNum', 'i4'), ('TrialNum', 'i4'), ('Time', 'f4'), ('X', 'f4'),
                  ('Z', 'f4'),('X_d','f4'),('Z_d','f4')]

    def __init__(self, trial_info, directory, suffix=".csv"):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._output_array = np.empty(0, dtype=self.output_data_type)
        self._processed_array = np.empty(0, dtype=self.processed_output_type)
        self._trial_info = np.loadtxt(trial_info, delimiter=",", skiprows=1)
        self._xgrid = [-3.7,-2.35,-1.25,-0.49,0.4,1.2,2.0,3.05]
        self._ygrid = [-3.7,-2.49,-1.49,-0.49,0.49,1.35,2.10,3.7]
        self._xctr = [round((self._xgrid[i+1] + self._xgrid[i])/2,1) for i in range(len(self._xgrid)-1)]
        self._yctr = [round((self._ygrid[i+1] + self._ygrid[i])/2,1) for i in range(len(self._ygrid)-1)]
    
    def clean_data_w_reboot(self):
        for file in self._files_paths:
            tmp = pd.read_csv(file)
            clean = 0
            #find where the data got duplicated due to reboot the testing program
            restart_flag = tmp.index[tmp['Participant ID'] == 'Participant ID'].to_list()
            
            #remove extra headers due to reboot before recording any data
            ## at the beginning of the task
            if sum([i < 5 for i in restart_flag]) != 0:
                header = [i for i in restart_flag if i < 5]
                real_start = max(header)+1
                tmp.drop([i for i in range(real_start)],axis=0,inplace=True)
                restart_flag = [i for i in restart_flag if i >= 5]
                clean += 1
            
            ## in the middle of the task
            reboot_list = []
            
            for idx in range(1, len(restart_flag)):
              if restart_flag[idx] - restart_flag[idx-1] == 1:
                reboot_list.append(restart_flag[idx])
                
            if len(reboot_list)!= 0:
                tmp.drop(reboot_list,axis=0,inplace=True)
                restart_flag = [i for i in restart_flag if i not in reboot_list]
                clean +=1
            
            # remove disrupted trials
            if len(restart_flag) != 0:
                flag = tmp[tmp['Participant ID'] != 'Participant ID'].reset_index().groupby('Level')['index'].first().reset_index()
                for i in restart_flag:
                ### check if the previous trial completed or disrupted
                    if tmp.loc[i-1,"Level"] in tmp.loc[tmp.index>i]['Level'].unique():
                      end = i+1
                      start = flag.sort_values('index').loc[flag['index']<end]['index'].tail(1).values[0]
                      tmp.drop([i for i in range(start,end)],axis=0,inplace=True)
                    else: 
                      tmp.drop(i,axis=0,inplace=True)
                
                clean +=1

            if clean!= 0:
                tmp.reset_index().drop(['index'],axis=1).to_csv(file,index=False)
        pass
    
    def combineTrajectory(self):
        for file in self._files_paths:
            array = DataStorage.get_array(file)
            self._output_array = np.append(self._output_array, np.core.records.fromarrays(array[:,[0,1,2,3,5]].transpose(),
                                      dtype=self.output_data_type))
        pass

    
    def processTrajectory(self):
        for file in self._files_paths:
            array = DataStorage.get_array(file)

            for level_id in np.unique(array[:, 1]):
                processed_array = self.process_trial(array,level_id)
                self._processed_array = np.append(self._processed_array,np.array(list(map(tuple,processed_array)),
                                      dtype=self.processed_output_type))
        pass
    
    def process_trial(self,array,level):
        sub_id = array[0][0]
        array = array[array[:,1] == level][5:,[2,3,5]]
        array = self.clean_up_trajectory(array)
        array = self.check_start_end(array,level)
        array = np.insert(array, 0, sub_id, axis=1)
        array = np.insert(array, 1, level, axis=1)
        
        return array
        
    def clean_up_trajectory(self,array):
        tmp = pd.DataFrame(array, columns = ['time','X','Z'])
        tmp['PosX_d'] = pd.cut(tmp.X, self._xgrid, labels=self._xctr).astype(np.float16)
        tmp['PosY_d'] = pd.cut(tmp.Z, self._ygrid, labels=self._yctr).astype(np.float16)
        tmp = tmp.round({'time': 0}).drop_duplicates(subset=['time','PosX_d','PosY_d'])

        # array = array[5:,:]
        # array = array[np.insert(np.diff(array,axis=0).astype(bool).any(1),0,True)]
        return np.array(tmp)
    
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
    
    def get_processed_dataframe(self):
        return pd.DataFrame(self._processed_array)

    def save_raw(self, filename):
        self.get_raw_dataframe().to_csv(filename,index=False)
        pass
    
    def save_processed(self,filename):
        self.get_processed_dataframe().to_csv(filename,index=False)
        pass
