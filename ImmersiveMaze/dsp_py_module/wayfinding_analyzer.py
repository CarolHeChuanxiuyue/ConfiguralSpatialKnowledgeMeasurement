# rongfei@ucsb.edu; carol.hcxy@gmail.com
import pathlib

from matplotlib import pyplot as plt, patches

from dsp_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd
import similaritymeasures

from dsp_py_module.strategies import Strategy



class WayFindingAnalyzer:
    euclidean_distance_data_type = [('ParticipantID', 'i4'), ('TrialNumber', 'i4'),
                                    ('LevelDistanceTraveled', 'f4')]

    def __init__(self, trial_info, time_csv, cleaned_up_csv, discrete_csv, directory, suffix=".csv", shortcut_map=None,
                 learning_map=None):
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)
        self._trial_info = pd.read_csv(trial_info, delimiter=",")
        self._input_df = pd.read_csv(cleaned_up_csv)
        self._euclidean_distance_array = np.empty(0, dtype=self.euclidean_distance_data_type)
        self._output_df = pd.DataFrame()
        self._time_info = pd.read_csv(time_csv)

        self._discrete_df = pd.read_csv(discrete_csv, converters={
            'pos': lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float64)})

        self._shortcut_map = shortcut_map
        self._learning_map = learning_map
        # self.strategy = Strategy("survey_coords.txt", "survey_landmarks.txt")

        self._xgrid = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
        self._ygrid = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]

    def analyze(self):
        for file in self._files_paths:
            array = DataStorage.get_array(file)

            for level_id in np.unique(array[:, 1]):
                self._euclidean_distance_array = np.append(self._euclidean_distance_array, np.array([(
                    self.get_participant_id(array),
                    level_id,
                    self.euclidean_distance(array, level_id),
                )], dtype=self.euclidean_distance_data_type))

        input_df = self._input_df.groupby(['SubjectNum', 'TrialNum'])[['X_d', 'Z_d', 'X', 'Z']].apply(
            self.total_moving_distance).reset_index()
        self._output_df = pd.DataFrame(self._euclidean_distance_array).merge(input_df, how='left',
                                                                             left_on=['ParticipantID', 'TrialNumber'],
                                                                             right_on=['SubjectNum', 'TrialNum']).drop(
            ['SubjectNum', 'TrialNum'], axis=1)

        pass

    def get_trial_info(self, trajectory_data):
        trial_id = trajectory_data['TrialNum'].iloc[0]
        subject_id = trajectory_data['SubjectNum'].iloc[0]
        # get the "Top" and "Target" of the row where TrialID matches in the trial info dataframe
        trial_info = self._trial_info[self._trial_info["TrialID"] == trial_id]

        # top is the second-last column, target is the last column
        top = trial_info["Top"].iloc[0]
        target = trial_info["Target"].iloc[0]

        return trial_id, subject_id, top, target

    def calculate_frechet(self):
        # create a new dataframe to store the frechet distance with the following columns
        # SubjectName, TrialNumber, FrechetLearn, FrechetShortcut, FrechetReversed, LearnDistance, ShortcutDistance, Failure
        frechet_df = pd.DataFrame(columns=['SubjectName',
                                           'TrialNumber',
                                           'FrechetLearn',
                                           'FrechetShortcut',
                                           'FrechetReversed',
                                           'LearnDistance',
                                           'ShortcutDistance',
                                           'Failure'])

        # get the unique trial numbers
        for subject_num in self._discrete_df['SubjectNum'].unique():
            print("Calculating Frechet distance for subject: " + str(subject_num) + "...")
            trial_nums = self._discrete_df['TrialNum'].unique()
            for trial_num in trial_nums:
                discrete_trajectory = self._discrete_df[
                    (self._discrete_df['SubjectNum'] == subject_num) & (self._discrete_df['TrialNum'] == trial_num)]
                _, _, start, end = self.get_trial_info(discrete_trajectory)

                # get dX and dZ and combine them into a tuple then a list
                dX = discrete_trajectory['dX'].tolist()
                dZ = discrete_trajectory['dZ'].tolist()
                p = list(zip(dX, dZ))

                shortcut_distance, q = self._shortcut_map.get_shortest_path_in_tiles(start, end)
                learning_distance, r = self._learning_map.get_shortest_path_in_tiles(start, end)
                reversed_r = r[::-1]

                # get the Status of the row where
                # SubjectNum and TrialNum and start and end matches in the time info dataframe
                status = self._time_info[
                    (self._time_info["ParticipantID"] == subject_num) & (
                            self._time_info["TrialNumber"] == trial_num) & (
                            self._time_info["StartingObject"] == start) & (self._time_info["TargetObject"] == end)][
                    "Status"]

                if status.empty:
                    status = True
                else:
                    status = False if status.iloc[0] == "Success" else True

                shortcut_index = similaritymeasures.frechet_dist(p, q)
                learning_index = similaritymeasures.frechet_dist(p, r)
                reversed_learning_index = similaritymeasures.frechet_dist(p, reversed_r)

                # all the values less than 0.001 are considered as 0
                if shortcut_index < 0.001:
                    shortcut_index = 0
                if learning_index < 0.001:
                    learning_index = 0
                if reversed_learning_index < 0.001:
                    reversed_learning_index = 0

                new_record = {'SubjectName': subject_num,
                              'TrialNumber': trial_num,
                              'FrechetLearn': learning_index,
                              'FrechetShortcut': shortcut_index,
                              'FrechetReversed': reversed_learning_index,
                              'LearnDistance': learning_distance,
                              'ShortcutDistance': shortcut_distance,
                              'Failure': status}

                # concat the new record to the frechet dataframe as pd series
                frechet_df = pd.concat([frechet_df, pd.DataFrame(new_record, index=[0])], axis=0, ignore_index=True)

        frechet_df.to_csv("ProcessedData/frechet.csv", index=False)
        return frechet_df

    def total_moving_distance(self, df):

        grid_distance = self.moving_distance(np.array(df[['X_d', 'Z_d']]))
        human_distance = self.moving_distance(np.array(df[['X', 'Z']]))

        return pd.Series([grid_distance, human_distance]).rename({0: 'grid_distance', 1: "human_distance"})

    def get_dataframe(self):
        return self._output_df

    def save(self, filename):
        self.get_dataframe().to_csv(filename, index=False)
        pass

    @staticmethod
    def moving_distance(array):

        difference = np.diff(array, axis=0)  # difference between column (vector difference)
        norm = np.linalg.norm(difference, axis=1)  # get magnitudes of row vectors

        return np.sum(norm).round(2)

    @staticmethod
    def euclidean_distance(array, level_id):

        array = array[array[:, 1] == level_id]
        array = array[:, [3, 5]]

        return WayFindingAnalyzer.moving_distance(array)

    @staticmethod
    def get_participant_id(array):
        return int(array[(0, 0)])

    def plot_map(self, trajectory_tuple=None):
        x_points = self._xgrid
        y_points = self._ygrid

        fig, ax = plt.subplots()

        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                rect = patches.Rectangle((x_points[i], y_points[j]), x_points[i + 1] - x_points[i],
                                         y_points[j + 1] - y_points[j], linewidth=1, edgecolor='gray', facecolor='none')
                ax.add_patch(rect)

            # check trajectory data is a tuple and not none
            if trajectory_tuple is not None and isinstance(trajectory_tuple, tuple):
                continuous_trajectory, discrete_trajectory = trajectory_tuple

                if continuous_trajectory is not None:
                    # get the title of the trajectory

                    trial_id, subject_id, top, target = self.get_trial_info(continuous_trajectory)

                    title = 'Subject ' + str(subject_id) + ' Trial ' + str(trial_id) + \
                            '\n From: ' + str(top) + ' To ' + str(target)
                    ax.set_title(title)

                    # plot lines using the X and Z columns of the dataframe
                    ax.plot(continuous_trajectory['X'], continuous_trajectory['Z'], color='green', linewidth=1)

                    if discrete_trajectory is not None:
                        # plot a connected scatter plot using the dX and dZ columns of the dataframe and step
                        ax.step(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='pink', linewidth=1)
                        ax.scatter(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='red', s=20)
                elif discrete_trajectory is not None:
                    ax.step(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='red', linewidth=1)

        ax.set_xlim([min(x_points), max(x_points)])
        ax.set_ylim([min(y_points), max(y_points)])
        ax.set_aspect('equal')
        # remove default ticks
        # ax.set_xticks(x_points, labels=x_points, minor=True)
        # ax.set_yticks(y_points, labels=y_points, minor=True)

        # set size of plot
        fig.set_size_inches(10, 10)

        plt.show()

    def plot_raw_trajectory_on_map(self, subject_num, trial_num):
        raw_df = self.get_raw_trajectory(subject_num, trial_num)
        self.plot_map(raw_df)

    def plot_discrete_trajectory_for_one_subject(self, subject_num, trial_num):
        raw_trajectory = self.get_raw_trajectory(subject_num, trial_num)
        # run lambda function on each row of the dataframe by calling get_closest_tile for X and Z columns
        discrete_trajectory = self.get_discrete_trajectory(subject_num, trial_num)

        self.plot_map((raw_trajectory, discrete_trajectory))
        pass

    def plot_all_discrete_trajectories_on_map(self):
        # get the unique subject numbers
        subject_nums = self.get_raw_dataframe()['SubjectNum'].unique()
        for subject_num in subject_nums:
            # get the unique trial numbers
            trial_nums = self.get_raw_dataframe()['TrialNum'].unique()
            for trial_num in trial_nums:
                print("Plotting subject " + str(subject_num) + " trial " + str(trial_num))
                self.plot_discrete_trajectory_for_one_subject(subject_num, trial_num)

    def get_raw_trajectory(self, subject_num, trial_num):
        print(self._input_df)
        return self._input_df[
            (self._input_df['SubjectNum'] == subject_num) & (self._input_df['TrialNum'] == trial_num)]

    def get_discrete_trajectory(self, subject_num, trial_num):
        return self._discrete_df[
            (self._discrete_df['SubjectNum'] == subject_num) & (self._discrete_df['TrialNum'] == trial_num)]

    def get_raw_dataframe(self):
        return self._input_df
