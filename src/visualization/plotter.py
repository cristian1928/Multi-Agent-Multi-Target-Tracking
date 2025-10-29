import os
from typing import Any, Dict, List, Tuple, cast
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # type: ignore
import re

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
    plt.rcParams["axes.labelsize"] = 12
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

def get_color_map(names: List[str], cmap_name: str = 'tab20') -> Dict[str, Tuple[float, ...]]:
    """
    Map each name to a color. If a name contains a numeric suffix (e.g., "A10" or "T2"),
    use that number to pick the color index (number-1) so color assignment matches code
    that indexes colors by agent/target index. Fallback to enumerated order if no number found.
    """
    cmap = plt.get_cmap(cmap_name)
    listed_cmap = cast(ListedColormap, cmap)
    standard_colors = cast(List[Tuple[float, ...]], listed_cmap.colors)
    color_map: Dict[str, Tuple[float, ...]] = {}
    for i, name in enumerate(names):
        m = re.search(r'(\d+)\s*$', name)  # trailing number
        if m:
            idx = max(0, int(m.group(1)) - 1)
        else:
            idx = i
        color_map[name] = standard_colors[idx % len(standard_colors)]
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

    # Build color maps using numeric-suffix-aware mapping so they match 2D subplot indexing
    agent_color_map = get_color_map(agent_names, cmap_name='tab20')
    target_color_map = get_color_map(target_names, cmap_name='tab10')

    # ─── Tracking Error Norm ───
    if agents_state_data:
        figure_error, axis_error = plt.subplots(figsize=(8, 6))

        rows = []
        for i, agent_dataframe in enumerate(agents_state_data):
            error_series = agent_dataframe['Synchronization Error Norm']
            rms_value = float(np.sqrt(np.mean(error_series**2)))
            name = agent_names[i]                       # e.g., "A10"
            # extract number from name; fallback to i+1 if not found
            m = re.search(r'\d+', name)
            agent_num = int(m.group()) if m else (i + 1)
            rows.append((agent_num, name, error_series, rms_value))

        # sort by numeric agent index so legend and plot ordering is stable
        rows.sort(key=lambda t: t[0])

        # plot in numeric order and collect legend handles that use the exact same colors
        legend_handles = []
        for agent_num, orig_name, error_series, rms_value in rows:
            color = agent_color_map.get(orig_name, 'grey')
            axis_error.plot(
                time_values,
                error_series,
                label=f'Agent {agent_num}: RMS {rms_value:.2f} $m$',
                color=color,
                linestyle='solid'
            )
            # create explicit legend handle with same color
            handle = Line2D([0], [0], color=color, lw=1.5)
            legend_handles.append(handle)

        axis_error.set_xlabel('Time (s)')
        axis_error.set_ylabel('Synchronization Error Norm $(m)$')

        # limit x-axis to 20 seconds (or the data max if shorter)
        if time_values is not None:
            try:
                tmax = float(time_values.max())
            except Exception:
                tmax = 1.0
            axis_error.set_xlim(0.0, min(1.0, tmax))

        # use the explicit handles so legend colors match the plotted colors
        axis_error.legend(handles=legend_handles,
                          labels=[f'Agent {t[0]}: RMS {t[3]:.2f} $m$' for t in rows],
                          loc='best', fontsize=12, frameon=True, edgecolor='black')
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
        axis_traj.plot(x_values, y_values, z_values, label=name, linestyle='solid', linewidth=1.0, color=agent_color_map[name])

    for i, target_dataframe in enumerate(targets_state_data):
        plot_df = target_dataframe if t_seconds is None else target_dataframe[target_dataframe['Time'] <= t_seconds]
        x_values = cast(Any, plot_df['Position X'].values)
        y_values = cast(Any, plot_df['Position Y'].values)
        z_values = cast(Any, plot_df['Position Z'].values)
        name = target_names[i]
        axis_traj.plot(x_values, y_values, z_values, label=name, linestyle='dashed', linewidth=1.0, color=target_color_map[name])

    axis_traj.set_xlabel('X Position (m)')
    axis_traj.set_ylabel('Y Position (m)')
    axis_traj.set_zlabel('Z Position (m)')    # type: ignore
    axis_traj.set_box_aspect((1, 1, 1))       # type: ignore
    # axis_traj.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    plt.tight_layout()

    plt.show()

def results() -> None:
    plot_from_csv()

if __name__ == "__main__":
    results()
