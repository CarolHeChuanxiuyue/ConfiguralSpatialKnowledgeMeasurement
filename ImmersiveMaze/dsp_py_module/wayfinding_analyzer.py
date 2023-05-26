# rongfei@ucsb.edu; carol.hcxy@gmail.com
import pathlib
import os

from matplotlib import pyplot as plt, patches

from dsp_py_module.shortcut_map import ShortcutMap
from dsp_py_module.data_storage import DataStorage
import numpy as np
import pandas as pd
import similaritymeasures

from dsp_py_module.strategies import Strategy


def compute_distance(path):
    """
    Compute the distance of a path
    :param path: a list of points
    :return: the distance of the path
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(path[i + 1] - path[i])
    return distance

class WayFindingAnalyzer:
    euclidean_distance_data_type = [('ParticipantID', 'i4'), ('TrialNumber', 'i4'),
                                    ('LevelDistanceTraveled', 'f4')]

    def __init__(self, trial_info, time_csv, cleaned_up_csv, discrete_csv,
                 strategy=None,
                 shortcut_map: ShortcutMap = None,
                 learning_map: ShortcutMap = None):
        self._trial_info = pd.read_csv(trial_info, delimiter=",")
        self._input_df = pd.read_csv(cleaned_up_csv)
        # self._euclidean_distance_array = np.empty(0, dtype=self.euclidean_distance_data_type)
        self._output_df = pd.DataFrame()
        self._time_info = pd.read_csv(time_csv)

        self._discrete_df = pd.read_csv(discrete_csv, converters={
            'pos': lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float64)})

        self._shortcut_map = shortcut_map
        self._learning_map = learning_map
        self._strategy = strategy

        self._xgrid = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
        self._ygrid = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]

    def get_trial_info(self, trajectory_data):
        trial_id = trajectory_data['TrialNum'].iloc[0]
        subject_id = trajectory_data['SubjectNum'].iloc[0]
        # get the "Top" and "Target" of the row where TrialID matches in the trial info dataframe
        trial_info = self._trial_info[self._trial_info["TrialID"] == trial_id]

        # top is the second-last column, target is the last column
        top = trial_info["Top"].iloc[0]
        target = trial_info["Target"].iloc[0]

        return trial_id, subject_id, top, target

    def get_status(self, subject_id, trial_id, start, end):
        status = self._time_info[
            (self._time_info["ParticipantID"] == subject_id) & (
                    self._time_info["TrialNumber"] == trial_id) & (
                    self._time_info["StartingObject"] == start) & (self._time_info["TargetObject"] == end)][
            "Status"]

        if status.empty:
            status = True
        else:
            status = False if status.iloc[0] == "Success" else True
        return status

    def calculate_frechet(self):
        # create a new dataframe to store the frechet distance with the following columns SubjectName, TrialNumber,
        # FrechetLearn, FrechetShortcut, FrechetReversed, LearnDistance, ShortcutDistance, Failure
        frechet_df = pd.DataFrame(columns=['SubjectName',
                                           'TrialOrder',
                                           'TrialNumber',
                                           'DiscreteDistance',
                                           'ContinuousDistance',
                                           'FrechetLearn',
                                           'FrechetLearnReversed',
                                           'FrechetShortcut',
                                           'FrechetReversed',
                                           'FrechetTopo',
                                           'LearnDistance',
                                           'ReversedLearnDistance',
                                           'ShortcutDistance',
                                           'TopoDistance',
                                           'Failure'])

        # get the unique trial numbers
        for subject_id in self._discrete_df['SubjectNum'].unique():
            print("Calculating Frechet distance for subject: " + str(subject_id) + "...")
            trial_nums = self._discrete_df['TrialNum'].unique()
            order = 1
            for trial_id in trial_nums:
                discrete_trajectory = self._discrete_df[
                    (self._discrete_df['SubjectNum'] == subject_id) & (self._discrete_df['TrialNum'] == trial_id)]
                continuous_trajectory = self._input_df[
                    (self._input_df['SubjectNum'] == subject_id) & (self._input_df['TrialNum'] == trial_id)]
                _, _, start, end = self.get_trial_info(discrete_trajectory)

                # get dX and dZ and combine them into a tuple then a list
                dX = discrete_trajectory['dX'].tolist()
                dZ = discrete_trajectory['dZ'].tolist()
                trajectory = list(zip(dX, dZ))

                discrete_distance = 0
                for i in range(len(trajectory) - 1):
                    discrete_distance += abs(trajectory[i][0] - trajectory[i + 1][0]) + abs(
                        trajectory[i][1] - trajectory[i + 1][1])

                diffs = continuous_trajectory[['X', 'Z']].diff()
                norms = np.linalg.norm(diffs, axis=1)
                continuous_distance = float(np.sum(norms[1:]))

                _, shortcut = self._shortcut_map.get_shortest_path_in_tiles(start, end)
                _, learning = self._learning_map.get_learning_path(start, end)
                learning = self._learning_map.convert_to_tile_path(learning)
                _, reverse_learning = self._learning_map.get_reverse_learning_path(start, end)
                reverse_learning = self._learning_map.convert_to_tile_path(reverse_learning)

                topo = self._shortcut_map.convert_to_tile_path(self._strategy.get_path(start, end))

                # sum up the distance between each point for shortcut, learning and topo using numpy.linalg.norm
                shortcut_distance = compute_distance(shortcut)
                learning_distance = compute_distance(learning)
                reverse_learning_distance = compute_distance(reverse_learning)
                topo_distance = compute_distance(topo)

                # get the Status of the row where
                # SubjectNum and TrialNum and start and end matches in the time info dataframe
                status = self.get_status(subject_id, trial_id, start, end)

                shortcut_index = similaritymeasures.frechet_dist(trajectory, shortcut)
                learning_index = similaritymeasures.frechet_dist(trajectory, learning)
                reversed_learning_index = similaritymeasures.frechet_dist(trajectory, reverse_learning)
                topo_index = similaritymeasures.frechet_dist(trajectory, topo)

                # all the values less than 0.001 are considered as 0
                if shortcut_index < 0.001:
                    shortcut_index = 0
                if learning_index < 0.001:
                    learning_index = 0
                if reversed_learning_index < 0.001:
                    reversed_learning_index = 0

                dec = 2
                new_record = {'SubjectName': subject_id,
                              'TrialOrder': order,
                              'TrialNumber': trial_id,
                              'DiscreteDistance': round(discrete_distance, dec),
                              'ContinuousDistance': round(continuous_distance, dec),
                              'FrechetLearn': round(learning_index, dec),
                              'FrechetLearnReversed': round(reversed_learning_index, dec),
                              'FrechetShortcut': round(shortcut_index, dec),
                              'FrechetTopo': round(topo_index, dec),
                              'LearnDistance': round(learning_distance, dec),
                              'ReversedLearnDistance': round(reverse_learning_distance, dec),
                              'ShortcutDistance': round(shortcut_distance, dec),
                              'TopoDistance': round(topo_distance, dec),
                              'Failure': status}

                order += 1

                # concat the new record to the frechet dataframe as pd series
                frechet_df = pd.concat([frechet_df, pd.DataFrame(new_record, index=[0])], axis=0, ignore_index=True)

        # save the frechet dataframe to csv
        frechet_df.to_csv("ProcessedData/frechet.csv", index=False, float_format="%.2f")
        return frechet_df

    def plot_map(self, trajectory_tuple=None, save_only=False):
        x_points = self._xgrid
        y_points = self._ygrid

        plt.close(plt.gcf())
        plt.gcf().set_size_inches(10, 10)

        # load a background image called map.png
        img = plt.imread('map3.png')
        plt.imshow(img, extent=[-3.7, 3.05, -3.7, 3.7])

        # for i in range(len(x_points) - 1):
        #     for j in range(len(y_points) - 1):
        #         rect = patches.Rectangle((x_points[i], y_points[j]), x_points[i + 1] - x_points[i],
        #                                  y_points[j + 1] - y_points[j], linewidth=0.5, edgecolor='gray',
        #                                  facecolor='none')
        #         plt.add_patch(rect)
        plt.vlines(x_points, min(y_points), max(y_points), colors='gray', linewidth=0.5)
        plt.hlines(y_points, min(x_points), max(x_points), colors='gray', linewidth=0.5)

        # check trajectory data is a tuple and not none
        if trajectory_tuple is not None and isinstance(trajectory_tuple, tuple):
            continuous_trajectory, discrete_trajectory = trajectory_tuple

            if continuous_trajectory is not None:
                # get the title of the trajectory

                trial_id, subject_id, top, target = self.get_trial_info(continuous_trajectory)

                title = 'Subject ' + str(subject_id) + ' Trial ' + str(trial_id) + \
                        '\n From: ' + str(top) + ' To ' + str(target) + \
                        '\n Failure: ' + str(self.get_status(subject_id, trial_id, top, target))
                plt.title(title)

                # plot lines using the X and Z columns of the dataframe
                plt.plot(continuous_trajectory['X'], continuous_trajectory['Z'], color='green', linewidth=1.5)

                if discrete_trajectory is not None:
                    # plot a connected scatter plot using the dX and dZ columns of the dataframe and step
                    plt.step(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='red', linewidth=1.5)
                    # ax.scatter(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='red', s=20)

                if save_only:
                    if not os.path.exists(f'path_plot/{subject_id}'):
                        os.makedirs(f'path_plot/{subject_id}')
                    plt.savefig(f'path_plot/{subject_id}/{trial_id}.png', dpi=120)

            elif discrete_trajectory is not None:
                plt.step(discrete_trajectory['dX'], discrete_trajectory['dZ'], color='pink', linewidth=1)

        plt.xlim([min(x_points), max(x_points)])
        plt.ylim([min(y_points), max(y_points)])
        # remove default ticks
        # ax.set_xticks(x_points, labels=x_points, minor=True)
        # ax.set_yticks(y_points, labels=y_points, minor=True)

        # set size of plot

        if not save_only:
            plt.show()

    def plot_topo(self, subject_num, trial_num):

        discrete_trajectory = self._discrete_df[
            (self._discrete_df['SubjectNum'] == subject_num) & (self._discrete_df['TrialNum'] == trial_num)]
        continuous_trajectory = self._input_df[
            (self._input_df['SubjectNum'] == subject_num) & (self._input_df['TrialNum'] == trial_num)]
        _, _, start, end = self.get_trial_info(discrete_trajectory)

        print(self._strategy.get_path(start, end))
        topo_path = self._shortcut_map.convert_to_tile_path(self._strategy.get_path(start, end))
        # convert topo path to dataframe
        topo_path = pd.DataFrame(topo_path, columns=['dX', 'dZ'])
        self.plot_map((self.get_raw_trajectory(subject_num, trial_num), topo_path))

    def plot_raw_trajectory_on_map(self, subject_num, trial_num):
        raw_df = self.get_raw_trajectory(subject_num, trial_num)
        self.plot_map(raw_df)

    def plot_trajectory_for_one_subject(self, subject_num, trial_num, save_only=False):
        raw_trajectory = self.get_raw_trajectory(subject_num, trial_num)
        # run lambda function on each row of the dataframe by calling get_closest_tile for X and Z columns
        discrete_trajectory = self.get_discrete_trajectory(subject_num, trial_num)

        self.plot_map((raw_trajectory, discrete_trajectory), save_only)
        pass

    def plot_all_discrete_trajectories_on_map(self, save_only=False):
        # get the unique subject numbers
        subject_nums = self.get_raw_dataframe()['SubjectNum'].unique()
        for subject_num in subject_nums:
            # get the unique trial numbers
            trial_nums = self.get_raw_dataframe()['TrialNum'].unique()
            for trial_num in trial_nums:
                print("Plotting subject " + str(subject_num) + " trial " + str(trial_num))
                self.plot_trajectory_for_one_subject(subject_num, trial_num, save_only)

    def get_raw_trajectory(self, subject_num, trial_num):
        # print(self._input_df)
        return self._input_df[
            (self._input_df['SubjectNum'] == subject_num) & (self._input_df['TrialNum'] == trial_num)]

    def get_discrete_trajectory(self, subject_num, trial_num):
        return self._discrete_df[
            (self._discrete_df['SubjectNum'] == subject_num) & (self._discrete_df['TrialNum'] == trial_num)]

    def get_raw_dataframe(self):
        return self._input_df
