import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dsp_py_module import WayFindingTrajectory
from dsp_py_module import ShortcutMap
from dsp_py_module import WayFindingAnalyzer
from dsp_py_module.strategies import Strategy

trial_info = "trial_info.csv"
data_path = "Shortcutting"


def plot_trajectory(trajectory):
    trajectory.plot_map()


def main():
    shortcut_map = ShortcutMap("walls.csv", "objects.csv", "ProcessedData/shortcuts.csv")
    # shortcut_map.save_survey_map()
    strategy = Strategy("survey_map.txt", "landmarks_on_survey_map.txt", shortcut_map)
    print(strategy.get_path("Stove", "Well"))
    print(strategy.get_path("Well", "Stove"))
    # strategy.plot_path("Mailbox", "Trashcan")
    # path = strategy.get_path("Piano", "Stove")

    # strategy.get_all_topological_plots("topo_plot", save_only=True)

    learning_map = ShortcutMap("learning_walls.csv", "objects.csv", "ProcessedData/learning_shortcuts.csv" , learning=True)
    #
    # trajectory = WayFindingTrajectory(trial_info, data_path, shortcut_map=shortcut_map)
    # trajectory.clean_data_w_reboot()
    # trajectory.combineTrajectory()
    # trajectory.processTrajectory()
    # trajectory.save_processed("ProcessedData/processed_trajectory.csv")
    # trajectory.convert_all_to_discrete()

    analyzer = WayFindingAnalyzer(
        trial_info,
        "ProcessedData/time.csv",
        "ProcessedData/processed_trajectory.csv",
        "ProcessedData/discrete_trajectory.csv",
        shortcut_map=shortcut_map,
        learning_map=learning_map,
        strategy=strategy,
    )

    # analyzer.plot_topo(401,1)
    # analyzer.calculate_frechet()
    analyzer.plot_all_discrete_trajectories_on_map(save_only=True)
    # analyzer.plot_trajectory_for_one_subject(401, 1, save_only=True)

if __name__ == '__main__':
    main()
