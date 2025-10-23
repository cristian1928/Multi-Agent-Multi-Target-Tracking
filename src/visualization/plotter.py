import os
from typing import Any, Dict, List, Tuple, cast
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # type: ignore

# Constants for data access
DATA_DIR = 'simulation_data'
AGENT_DATA_DIR = os.path.join(DATA_DIR, 'agent_data')
TARGET_DATA_DIR = os.path.join(DATA_DIR, 'target_data')
STATE_DATA_SUFFIX = '_state_data.csv'
NN_DATA_SUFFIX = '_nn_data.csv'

def configure_plot() -> None:
    plt.style.use(['science', 'ieee'])
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams.update({
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.5,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
    })

def get_simulation_data() -> Tuple[List[str], List[pd.DataFrame], List[str], List[pd.DataFrame]]:
    agent_state_files = sorted([f for f in os.listdir(AGENT_DATA_DIR) if f.endswith(STATE_DATA_SUFFIX)]) if os.path.isdir(AGENT_DATA_DIR) else []
    target_state_files = sorted([f for f in os.listdir(TARGET_DATA_DIR) if f.endswith(STATE_DATA_SUFFIX)]) if os.path.isdir(TARGET_DATA_DIR) else []

    agent_names = [f.replace(STATE_DATA_SUFFIX, '') for f in agent_state_files]
    target_names = [f.replace(STATE_DATA_SUFFIX, '') for f in target_state_files]

    agents_state_data = [pd.read_csv(os.path.join(AGENT_DATA_DIR, f)) for f in agent_state_files]
    targets_state_data = [pd.read_csv(os.path.join(TARGET_DATA_DIR, f)) for f in target_state_files]

    return agent_names, agents_state_data, target_names, targets_state_data

def get_color_map(names: List[str]) -> Dict[str, Tuple[float, ...]]:
    cmap = plt.get_cmap('tab20')
    listed_cmap = cast(ListedColormap, cmap)
    standard_colors = cast(List[Tuple[float, ...]], listed_cmap.colors)
    color_map: Dict[str, Tuple[float, ...]] = {}
    for i, name in enumerate(names):
        color_map[name] = standard_colors[i % len(standard_colors)]
    return color_map

def plot_from_csv() -> None:
    configure_plot()
    agent_names, agents_state_data, target_names, targets_state_data = get_simulation_data()

    if not agents_state_data and not targets_state_data:
        print("No simulation data found.")
        return

    time_values = None
    if agents_state_data:
        time_values = agents_state_data[0]['Time']
    elif targets_state_data:
        time_values = targets_state_data[0]['Time']

    agent_color_map = get_color_map(agent_names)
    target_color_map = get_color_map(target_names)

    # ─── Tracking Error Norm ───
    if agents_state_data:
        figure_error, axis_error = plt.subplots(figsize=(8, 6))
        for i, agent_dataframe in enumerate(agents_state_data):
            error_series = agent_dataframe['Synchronization Error Norm']
            rms_value = float(np.sqrt(np.mean(error_series**2)))
            name = agent_names[i]
            axis_error.plot(time_values, error_series, label=f'{name}: RMS {rms_value:.2f} $m$', color=agent_color_map[name], linestyle='solid')
        axis_error.set_xlabel('Time (s)')
        axis_error.set_ylabel('Synchronization Error Norm $(m)$')
        axis_error.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
        plt.tight_layout()

    # ─── Spatial Trajectories over Time ───
    t_seconds = None

    figure_traj = plt.figure(figsize=(8, 6))
    axis_traj = figure_traj.add_subplot(111, projection='3d')

    for i, agent_dataframe in enumerate(agents_state_data):
        plot_df = agent_dataframe if t_seconds is None else agent_dataframe[agent_dataframe['Time'] <= t_seconds]
        x_values = cast(Any, plot_df['Position X'].values)
        y_values = cast(Any, plot_df['Position Y'].values)
        z_values = cast(Any, plot_df['Position Z'].values)
        name = agent_names[i]
        axis_traj.plot(x_values, y_values, z_values, label=name, linestyle='solid', color=agent_color_map[name])

    for i, target_dataframe in enumerate(targets_state_data):
        plot_df = target_dataframe if t_seconds is None else target_dataframe[target_dataframe['Time'] <= t_seconds]
        x_values = cast(Any, plot_df['Position X'].values)
        y_values = cast(Any, plot_df['Position Y'].values)
        z_values = cast(Any, plot_df['Position Z'].values)
        name = target_names[i]
        axis_traj.plot(x_values, y_values, z_values, label=name, linestyle='dashed', linewidth=2.0, color=target_color_map[name])

    axis_traj.set_xlabel('X Position (m)')
    axis_traj.set_ylabel('Y Position (m)')
    axis_traj.set_zlabel('Z Position (m)')    # type: ignore
    axis_traj.set_box_aspect((1, 1, 1))       # type: ignore
    axis_traj.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    plt.tight_layout()

    plt.show()

def results() -> None:
    plot_from_csv()

if __name__ == "__main__":
    results()
