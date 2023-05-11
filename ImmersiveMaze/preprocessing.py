import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dsp_py_module import WayFindingTrajectory
from dsp_py_module import ShortcutMap
from dsp_py_module import WayFindingAnalyzer

trial_info = "trial_info.csv"
data_path = "Shortcutting"


def plot_trajectory(trajectory):
    trajectory.plot_map()


def main():
    shortcut_map = ShortcutMap("walls.csv", "objects.csv", "ProcessedData/shortcuts.csv")
    learning_map = ShortcutMap("learning_walls.csv", "objects.csv", "ProcessedData/learning_shortcuts.csv")
    # trajectory = WayFindingTrajectory(trial_info, data_path, shortcut_map=shortcut_map)
    # trajectory.clean_data_w_reboot()
    # trajectory.combineTrajectory()
    # trajectory.convert_all_to_discrete()

    analyzer = WayFindingAnalyzer(
        trial_info,
        "ProcessedData/time.csv",
        "ProcessedData/processed_trajectory.csv",
        "ProcessedData/discrete_trajectory.csv",
        data_path,
        shortcut_map=shortcut_map,
        learning_map=learning_map)

    analyzer.calculate_frechet()


if __name__ == '__main__':
    main()
