# Carol He: carol.hcxy@gmail.com
import pathlib
from matplotlib import patches
from matplotlib.ticker import FormatStrFormatter

# from dsp_py_module.data_storage import DataStorage
from dsp_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_new_trajectory_record_from_point(discrete_trajectory, tile_pos, pos):
    first_row = discrete_trajectory.iloc[0]
    # print(f"Interpolating point:{first_row['SubjectNum']}-{first_row['TrialNum']} at {pos}")
    return pd.DataFrame([[first_row['SubjectNum'],
                          first_row['TrialNum'],
                          0,
                          tile_pos[0],
                          tile_pos[1],
                          np.array(pos)]], columns=discrete_trajectory.columns)


def find_pos_index(pos, df):
    df = df.reset_index()
    mask = df.apply(lambda row: np.allclose(row['pos'], pos), axis=1)
    result = df[mask].index
    if len(result) == 0:
        return -1
    else:
        return result[0]


class WayFindingTrajectory:
    output_data_type = [('SubjectNum', 'i4'), ('TrialNum', 'i4'), ('Time', 'f4'), ('X', 'f4'),
                        ('Z', 'f4')]
    processed_output_type = [('SubjectNum', 'i4'), ('TrialNum', 'i4'), ('Time', 'f4'), ('X', 'f4'),
                             ('Z', 'f4'), ('X_d', 'f4'), ('Z_d', 'f4')]

    def __init__(self, trial_info, directory, suffix=".csv", shortcut_map=None, learning_map=None):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._output_array = np.empty(0, dtype=self.output_data_type)
        self._processed_array = np.empty(0, dtype=self.processed_output_type)
        # self._trial_info = np.loadtxt(trial_info, delimiter=",", skiprows=1)
        self._trial_info = pd.read_csv(trial_info, delimiter=",")
        self._xgrid = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
        self._ygrid = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]
        self._xctr = [round((self._xgrid[i + 1] + self._xgrid[i]) / 2, 1) for i in range(len(self._xgrid) - 1)]
        self._yctr = [round((self._ygrid[i + 1] + self._ygrid[i]) / 2, 1) for i in range(len(self._ygrid) - 1)]
        self._shortcut_map = shortcut_map
        self._learning_map = learning_map
        self._discrete_trajectory = None

    def clean_data_w_reboot(self):
        for file in self._files_paths:
            print('clean: '+file.stem + "...")
            tmp = pd.read_csv(file)
            clean = 0
            # find where the data got duplicated due to reboot the testing program
            restart_flag = tmp.index[tmp['Participant ID'] == 'Participant ID'].to_list()

            # remove extra headers due to reboot before recording any data
            ## at the beginning of the task
            if sum([i < 5 for i in restart_flag]) != 0:
                header = [i for i in restart_flag if i < 5]
                real_start = max(header) + 1
                tmp.drop([i for i in range(real_start)], axis=0, inplace=True)
                restart_flag = [i for i in restart_flag if i >= 5]
                clean += 1

            ## in the middle of the task
            reboot_list = []

            for idx in range(1, len(restart_flag)):
                if restart_flag[idx] - restart_flag[idx - 1] == 1:
                    reboot_list.append(restart_flag[idx])

            if len(reboot_list) != 0:
                tmp.drop(reboot_list, axis=0, inplace=True)
                restart_flag = [i for i in restart_flag if i not in reboot_list]
                clean += 1

            # remove disrupted trials
            if len(restart_flag) != 0:
                flag = tmp[tmp['Participant ID'] != 'Participant ID'].reset_index().groupby('Level')[
                    'index'].first().reset_index()
                for i in restart_flag:
                    # check if the previous trial completed or disrupted
                    if tmp.loc[i - 1, "Level"] in tmp.loc[tmp.index > i]['Level'].unique():
                        end = i + 1
                        start = flag.sort_values('index').loc[flag['index'] < end]['index'].tail(1).values[0]
                        tmp.drop([i for i in range(start, end)], axis=0, inplace=True)
                    else:
                        tmp.drop(i, axis=0, inplace=True)

                clean += 1

            if clean != 0:
                tmp.reset_index().drop(['index'], axis=1).to_csv(file, index=False)
        pass

    def combineTrajectory(self):
        for file in self._files_paths:
            array = DataStorage.get_array(file)
            self._output_array = np.append(self._output_array,
                                           np.core.records.fromarrays(array[:, [0, 1, 2, 3, 5]].transpose(),
                                                                      dtype=self.output_data_type))
        pass

    def processTrajectory(self):
        for file in self._files_paths:
            print('process: '+file.stem + "...")
            array = DataStorage.get_array(file)

            for level_id in np.unique(array[:, 1]):
                processed_array = self.process_trial(array, level_id)
                self._processed_array = np.append(self._processed_array, np.array(list(map(tuple, processed_array)),
                                                                                  dtype=self.processed_output_type))
        pass

    def process_trial(self, array, level):
        sub_id = array[0][0]
        array = array[array[:, 1] == level][5:, [2, 3, 5]]
        array = self.clean_up_trajectory(array)
        array = self.check_start_end(array, level)
        array = np.insert(array, 0, sub_id, axis=1)
        array = np.insert(array, 1, level, axis=1)

        return array

    def clean_up_trajectory(self, array):
        tmp = pd.DataFrame(array, columns=['time', 'X', 'Z'])
        tmp['PosX_d'] = pd.cut(tmp.X, self._xgrid, labels=self._xctr).astype(np.float16)
        tmp['PosY_d'] = pd.cut(tmp.Z, self._ygrid, labels=self._yctr).astype(np.float16)
        tmp = tmp.round({'time': 0}).drop_duplicates(subset=['time', 'PosX_d', 'PosY_d'])

        # array = array[5:,:]
        # array = array[np.insert(np.diff(array,axis=0).astype(bool).any(1),0,True)]
        return np.array(tmp)

    def check_start_end(self, array, level):
        trial = self.trial_info(level)

        startright = 0
        startindex = 0

        for i in range(int(len(array) / 4)):
            if np.linalg.norm([trial[0] - array[i][3], trial[1] - array[i][4]]) < 0.01:
                startright = 1
                startindex = i

        if startright == 0:
            array = np.insert(array, 0, np.array([array[0][0], trial[0], trial[1], trial[0], trial[1]]), axis=0)
        else:
            array = array[startindex:, :]

        endright = 0
        endindex = 0
        for i in range(len(array) - int(len(array) / 4), len(array)):
            if np.linalg.norm([trial[2] - array[i][3], trial[3] - array[i][4]]) < 0.01:
                endright = 1
                endindex = i
                break

        if endright == 0:
            array = np.insert(array, len(array),
                              np.array([array[len(array) - 1][0], trial[2], trial[3], trial[2], trial[3]]), axis=0)
        else:
            array = array[:endindex + 1, :]

        return array

    def trial_info(self, level_id):
        # get the row where TrialNum matches level_id
        trial = self._trial_info[self._trial_info['TrialID'] == level_id].reset_index()
        start_x = trial['start_x'].values[0]
        start_y = trial['start_y'].values[0]
        end_x = trial['end_x'].values[0]
        end_y = trial['end_y'].values[0]
        return [start_x, start_y, end_x, end_y]

    def get_raw_dataframe(self):
        return pd.DataFrame(self._output_array)

    def get_processed_dataframe(self):
        return pd.DataFrame(self._processed_array)

    def save_raw(self, filename):
        self.get_raw_dataframe().to_csv(filename, index=False)
        pass

    def save_processed(self, filename):
        self.get_processed_dataframe().to_csv(filename, index=False)
        pass

    def get_closest_tile(self, point):
        for i in range(len(self._xgrid) - 1):
            for j in range(len(self._ygrid) - 1):
                if self._xgrid[i] <= point[0] <= self._xgrid[i + 1] \
                        and self._ygrid[j] <= point[1] <= self._ygrid[j + 1]:
                    # get the center point of the tile
                    x = (self._xgrid[i] + self._xgrid[i + 1]) / 2
                    y = (self._ygrid[j] + self._ygrid[j + 1]) / 2
                    return [x, y, np.array([i, j])]
        return [-999, -999, -999, -999]

    def get_trial_info(self, trajectory_data):
        trial_id = trajectory_data['TrialNum'].iloc[0]
        subject_id = trajectory_data['SubjectNum'].iloc[0]
        # get the "Top" and "Target" of the row where TrialID matches in the trial info dataframe
        trial_info = self._trial_info[self._trial_info["TrialID"] == trial_id]

        # top is the second-last column, target is the last column
        top = trial_info["Top"].iloc[0]
        target = trial_info["Target"].iloc[0]

        return trial_id, subject_id, top, target

    def get_raw_trajectory(self, subject_num, trial_num):
        raw_df = self.get_raw_dataframe()
        # get the rows where SubjectNum and TrialNum match
        raw_df = raw_df[(raw_df['SubjectNum'] == subject_num) & (raw_df['TrialNum'] == trial_num)]
        return raw_df

    def get_discrete_trajectory(self, subject_num, trial_num):
        if self._discrete_trajectory is None:
            self.save_discrete()
        return self._discrete_trajectory[(self._discrete_trajectory['SubjectNum'] == subject_num) & (
                self._discrete_trajectory['TrialNum'] == trial_num)]

    def get_interpolated_path(self, discrete_trajectory, shortcut):
        new_records = pd.DataFrame(columns=['SubjectNum', 'TrialNum', 'Time', 'dX', 'dZ', 'pos'])
        for s in shortcut:
            tile = self._shortcut_map.get_tile_center(s)
            new_row = get_new_trajectory_record_from_point(discrete_trajectory, tile, s)
            new_records = pd.concat([new_records, new_row], ignore_index=True)
        return new_records

    def convert_to_discrete(self, trajectory_data):
        # the origin here starts from the bottom left corner

        discrete_trajectory = trajectory_data.apply(lambda row: self.get_closest_tile([row['X'], row['Z']]), axis=1,
                                                    result_type='expand')
        # set the column names to dX and dZ
        discrete_trajectory.columns = ['dX', 'dZ', 'pos']

        discrete_trajectory = pd.concat([trajectory_data, discrete_trajectory], axis=1)
        # select the columns that is not X and Z
        discrete_trajectory = discrete_trajectory[['SubjectNum', 'TrialNum', 'Time', 'dX', 'dZ', 'pos']]

        # remove rows consecutive duplicates
        discrete_trajectory = discrete_trajectory.drop_duplicates(subset=['dX', 'dZ'], keep='first')

        # remove rows where a lambda function returns a boolean that check if pos position is wall
        discrete_trajectory = discrete_trajectory[~discrete_trajectory['pos'].apply(
            lambda pos: self._shortcut_map.check_coord_is_wall(pos))].copy()

        fraction_length = len(discrete_trajectory) // 3

        _, _, start_name, end_name = self.get_trial_info(trajectory_data)
        start_pos = self._shortcut_map.get_object_pos(start_name)
        end_pos = self._shortcut_map.get_object_pos(end_name)
        start_tile_pos = self._shortcut_map.get_tile_center(start_pos)

        # check if start_pos in the first fraction_length of discrete_trajectory['pos']
        start_pos_index = find_pos_index(start_pos, discrete_trajectory.iloc[:fraction_length])
        if start_pos_index == -1:
            # get a copy of the  first row of the dataframe
            new_row = get_new_trajectory_record_from_point(discrete_trajectory, start_tile_pos, start_pos)
            # insert the first row to the first row of the dataframe
            discrete_trajectory = pd.concat([new_row, discrete_trajectory], ignore_index=True)

        else:
            discrete_trajectory = discrete_trajectory.iloc[start_pos_index:]
        # interpolate points
        i = 0
        while i < len(discrete_trajectory) - 1:
            # get the row at i
            current_row = discrete_trajectory.iloc[i]
            current_pos = current_row['pos']
            next_row = discrete_trajectory.iloc[i + 1]
            next_pos = next_row['pos']
            offset = next_pos - current_pos
            if offset.dot(offset) > 1:
                shortcut = self._shortcut_map.get_shortest_path_from_two_coords(current_pos, next_pos)

                new_records = self.get_interpolated_path(discrete_trajectory, shortcut[1:-1])

                discrete_trajectory = pd.concat(
                    [discrete_trajectory.iloc[:i + 1], new_records, discrete_trajectory.iloc[i + 1:]],
                    ignore_index=True)
                i += len(shortcut) - 2
            else:
                i += 1

        # check if end_pos in the last freaction_length of discrete_trajectory['pos']
        end_pos_index = find_pos_index(end_pos, discrete_trajectory[-fraction_length:])
        if end_pos_index == -1:
            current_last_pos = discrete_trajectory['pos'].iloc[-1]
            shortcut = self._shortcut_map.get_shortest_path_from_two_coords(current_last_pos, end_pos)
            if len(shortcut) > 1:
                new_records = self.get_interpolated_path(discrete_trajectory, shortcut[1:])
                discrete_trajectory = pd.concat([discrete_trajectory, new_records], ignore_index=True)

        else:
            end_pos_index = find_pos_index(end_pos, discrete_trajectory)
            discrete_trajectory = discrete_trajectory.iloc[:end_pos_index + 1]

        return discrete_trajectory

    def save_discrete(self, save_name="ProcessedData/discrete_trajectory.csv"):
        raw_df = self.get_raw_dataframe()
        # get the unique subject numbers
        subject_nums = raw_df['SubjectNum'].unique()
        final_trajectory = pd.DataFrame(columns=['SubjectNum', 'TrialNum', 'Time', 'dX', 'dZ', 'pos'])
        for subject_num in subject_nums:
            print("Save discrete trajectory for the participant: " + str(subject_num) + "...")
            # get the unique trial numbers
            trial_nums = raw_df['TrialNum'].unique()
            for trial_num in trial_nums:
                raw_trajectory = self.get_raw_trajectory(subject_num, trial_num)
                # run lambda function on each row of the dataframe by calling get_closest_tile for X and Z columns
                try:
                    discrete_trajectory = self.convert_to_discrete(raw_trajectory)
                except Exception as e: print("Warning!! The participant : " + str(subject_num) + " did not complete trial " + str(trial_num) + ". Please remove it")
                # save the discrete trajectory to csv
                final_trajectory = pd.concat([final_trajectory, discrete_trajectory], ignore_index=True)

        final_trajectory.to_csv(save_name, index=False)
