#Carol He (UCSB-Dec-2022): carol.hcxy@gmail.com

import pandas as pd
import numpy as np
import csv
from desktop_py_module.data_storage import DataStorage

class Pointing:
    output_type = [('Subject', 'i4'), ('Age', 'i4'), ('Gender', np.unicode_, 16), ('AllowedTime', 'i4'),
                  ('Order', 'i4'),('Center',np.unicode_, 16),('Top',np.unicode_, 16),('Target',np.unicode_, 16),
                  ('TargetAngle','f4'),('EstimateAngle','f4'),('AngularError','f16')]

    def __init__(self, trial_info, directory, suffix=".txt"):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._data = np.empty(0, dtype=self.output_type)
        self._trial_info = pd.read_csv(trial_info)
    
    def read_pointing_data(self):
        for file in self._files_paths:

            array = self.process_each_file(file)
            self._data = np.append(self._data,np.array(list(map(tuple,array)),dtype=self.output_type))

        self._data= pd.DataFrame(self._data)

        pass
    
    def join_trial_info(self):

        self._data[['Top','Target']] = self._data[['Top','Target']].apply(lambda x: x.str.lower().str.replace(" ", ""))
        self._trial_info[['Top','Target']] = self._trial_info[['Top','Target']].apply(lambda x: x.str.lower().str.replace(" ", ""))
        self._data = self._data.merge(self._trial_info[['DSP_Trial','Top','Target','targetAngle']],on=['Top','Target'])

        pass


    @staticmethod
    def process_each_file(file):

        with open(file) as f:

            csv_f = csv.reader(f, delimiter = '\t', lineterminator='\n')
            array = []
            
            for row in csv_f: #for every line in the data .csv

                if row[0].startswith("Age"):
                    currAge = row[1]

                if row[0].startswith("AllowedTime"):
                    currTime = row[1]
                
                if row[0].startswith("Gender"):
                    Gender = row[1]

                if row[0].isdigit() and len(row)==8:
                    array.append([str(row[0]),str(currAge),str(Gender),str(currTime),str(row[1]),str(row[2]),str(row[3]),str(row[4]),str(row[5]),str(row[6]),str(row[7])])

        return array

    def save(self, filename):
        
        self._data.to_csv(filename,index=False)
        
        pass