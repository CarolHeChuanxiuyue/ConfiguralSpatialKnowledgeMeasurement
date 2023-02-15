#Carol He (UCSB-Oct-2022): carol.hcxy@gmail.com

import pandas as pd
from dsp_py_module.data_storage import DataStorage

class Pointing:
    def __init__(self, directory, suffix=".csv"):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
    
    def read_pointing_data(self):
        pointData = pd.DataFrame()
        for file in self._files_paths:
            tmp = pd.read_csv(file)
            tmp['ID'] = int(file.stem)
            pointData = pd.concat([pointData,tmp])
            
        pointData.columns = ['TrialOrder','TrialID','AngleResponse','AngleCorr','AngError','Start','End','RT','ID']
        pointData['AngErrorAbs'] =  pointData['AngError'].apply(lambda x: abs(x) if abs(x)<=180 else 360-abs(x))
        
        return pointData
    
    def save(self,filename):
        self.read_pointing_data().to_csv(filename,index=False)
        pass