from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = False

from src.core.entity import Agent, Target
from src.io.data_manager import close_all_files, save_state_to_csv
from src.visualization.plotter import results
from src.simulation.dynamics import get_initial_conditions

# ---------------------------------------------------------------

def build_entity_specs(base_config: dict[str, Any], section_key: str) -> List[Tuple[NDArray[np.float64], dict[str, Any]]]:
    specs: List[Tuple[NDArray[np.float64], dict[str, Any]]] = []
    for index, item in enumerate(base_config[section_key]):
        dynamics_type = item["dynamics_type"]
        initial_position = np.array(
            item.get("initial_position", get_initial_conditions(dynamics_type)),
            dtype=float
        )
        merged_config = {**base_config, **item, "dynamics_type": dynamics_type}
        specs.append((initial_position, merged_config))
    return specs

# ---------------------------------------------------------------

def construct_undirected_neighborhood_set(entities: list[Any], edge_set: list[list[int]]) -> None:
    for i, j in edge_set:
        a, b = entities[i - 1], entities[j - 1]
        if b not in a.neighbors:
            a.neighbors.append(b)
        if a not in b.neighbors:
            b.neighbors.append(a)

# ---------------------------------------------------------------

# def make_offsets_agents(agent_specs: list[tuple[np.ndarray, dict]],
#                         agent_offsets: np.ndarray,
#                         target_specs: list[tuple[np.ndarray, dict]],
#                         time_steps: int,
#                         base_config: dict[str, Any],
#                         neighbors: list[list[int]]) -> np.ndarray:
# #
#     nd = int(agent_specs[0][1]['num_states'])
#     N = len(agent_specs)
#     agent_offsets = np.zeros((nd, N))
#
#     if target_specs and N >= 3:
#         target_pos_1 = target_specs[0][0]
#         target_pos_2 = target_specs[1][0]
#         agent_positions = np.array([spec[0] for spec in agent_specs])
#         l = np.linalg.norm(agent_positions[1] - agent_positions[0])
# #
#         tri_formation = [
#             np.array([0, l/np.sqrt(3), 0]),
#             np.array([l/2, -l*np.sqrt(3)/6, 0]),
#             np.array([-l/2, -l*np.sqrt(3)/6, 0])
#         ]
#
#         desired_positions_1 = [target_pos_1 + vertex for vertex in tri_formation]
#         desired_positions_2 = [target_pos_2 + vertex for vertex in tri_formation]
# #
#         for i in range(3):
#             for j in neighbors[i]:
#                 if i < j:
#                     # triangular formation around target 1
#                     delta_ij_1 = desired_positions_1[j] - desired_positions_1[i]
#                     agent_offsets[:, i] += delta_ij_1
#                     agent_offsets[:, j] -= delta_ij_1
#
#         for i in range(3, 6):
#             for j in neighbors[i]:
#                 if i < j:
#                     # triangular formation around target 2
#                     delta_ij_2 = desired_positions_2[j] - desired_positions_2[i]
#                     agent_offsets[:, i] += delta_ij_2
#                     agent_offsets[:, j] -= delta_ij_2
#
#     return agent_offsets

# ---------------------------------------------------------------

def make_offsets_targets(target_specs: list[tuple[np.ndarray, dict]],
                         target_offsets: np.ndarray,
                         time_steps: int,
                         base_config: dict[str, Any],
                         neighbors: list[list[int]]) -> np.ndarray:

    nd = int(target_specs[0][1]['num_states'])
    N = len(target_specs)
    target_offsets = np.zeros((nd, N))

    if target_specs and N >= 4:
        target_pos_1 = target_specs[0][0]
        target_pos_2 = target_specs[1][0]
        target_pos_3 = target_specs[2][0]
        target_pos_4 = target_specs[3][0]

        target_positions = np.array([spec[0] for spec in target_specs])
        d = np.linalg.norm(target_positions[1] - target_positions[0])
        h = np.sqrt(6) * d / 3

        tet_formation = [
            np.array([0, 0, 0]),
            np.array([d, 0, 0]),
            np.array([d / 2, np.sqrt(3) * d / 2, 0]),
            np.array([d / 2, np.sqrt(3) * d / 6, h])
        ]

        desired_positions_1 = [target_pos_1 + vertex for vertex in tet_formation]
        desired_positions_2 = [target_pos_2 + vertex for vertex in tet_formation]
        desired_positions_3 = [target_pos_3 + vertex for vertex in tet_formation]
        desired_positions_4 = [target_pos_4 + vertex for vertex in tet_formation]

        for i in range(4):
            for j in neighbors[i]:
                if i < j and j < 4:
                    delta_ij_1 = desired_positions_1[j] - desired_positions_1[i]
                    target_offsets[:, i] += delta_ij_1/4
                    target_offsets[:, j] -= delta_ij_1/4

    return target_offsets


# ---------------------------------------------------------------

def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]

    final_time: float = base_config["final_time"]
    time_step_delta: float = base_config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    np.random.seed(base_config["seed"])

    # Build specs
    target_specs = build_entity_specs(base_config, section_key="targets")
    # agent_specs = build_entity_specs(base_config, section_key="agents")

    # number of states and agents
    nd = int(target_specs[0][1]['num_states'])
    N = len(target_specs)

    # construct neighbors list (0-indexed)
    neighbors: list[list[int]] = [[] for _ in range(N)]
    for i, j in base_config["target_edge_set"]:
        neighbors[i - 1].append(j - 1)
        neighbors[j - 1].append(i - 1)

    # compute offsets
    '''
    agent_offsets = np.zeros((nd, N))
    agent_offsets = make_offsets_agents(agent_specs, agent_offsets, target_specs, time_steps, base_config, neighbors)
    '''

    # create targets
    target_offsets = np.zeros((nd, N))
    target_offsets = make_offsets_targets(target_specs, target_offsets, time_steps, base_config, neighbors)

    # --- FIX 1: Create an empty agents list ---
    # This is needed so save_state_to_csv has an 'agents' variable to use.
    agents: list[Agent] = [] 

    targets: list[Target] = [
        Target(
            initial_position=pos,
            time_steps=time_steps,
            config=conf
        )
        for pos, conf in target_specs
    ]

    # Assign offsets
    for i, target in enumerate(targets):
        target.offsets = target_offsets[:, i]

    # Assign full list of targets to each target
    for target in targets:
        target.targets = targets

    # Construct neighborhoods for targets
    construct_undirected_neighborhood_set(targets, base_config.get("target_edge_set", []))

    # Simulation loop
    for step in range(1, time_steps):
        '''
        for agent in agents:
            agent.compute_control_output(step)
        '''
        for target in targets:
            target.compute_control_output(step)

        '''
        for agent in agents:
            agent.update_dynamics(step)
        '''
        for target in targets:
            target.update_dynamics(step)

        simulation_time: float = step * time_step_delta

        # --- FIX 2: Uncommented this line ---
        # This now saves the state (empty agents list, and your targets)
        save_state_to_csv(step, simulation_time, agents, targets)

        print(f"Progress: {step / time_steps * 100:6.2f}%", end="\r", flush=True)

    print("\nSimulation completed.")
    close_all_files()

# ---------------------------------------------------------------

    # Create a 2D plot
    fig1, ax1 = plt.subplots(figsize=(8, 8))  # square aspect

    # Get target IDs for labeling
    target_ids = [conf.get("id", f"T{j+1}") for (_, conf) in target_specs]

    # --- Find when formation is reached ---
    formation_reached_step = time_steps - 1  # Default to end if not found
    formation_threshold = 0.1  # Adjust this threshold as needed

    for step in range(time_steps):
        if step % 10 == 0:  # Check every 10 steps for efficiency
            # Check if tetrahedral formation is reached
            formation_achieved = True
            for target in targets:
                # Using all states for movement check, assuming 3D dynamics
                current_pos = target.positions[:3, step]
                if step < time_steps - 1:
                    next_pos = target.positions[:3, step + 1]
                    movement = np.linalg.norm(next_pos - current_pos)
                    if movement > formation_threshold:
                        formation_achieved = False
                        break

            if formation_achieved:
                formation_reached_step = step
                break

    # --- Plot targets up to formation_reached_step ---
    step_skip = 5  # Reduced for smoother trajectories

    for j, target in enumerate(targets):
        # Extract X and Y data (first 2 states) up to formation step
        traj = target.positions[:2, :formation_reached_step + 1]
        x_full, y_full = traj[0, :], traj[1, :]
        x, y = x_full[::step_skip], y_full[::step_skip]

        t_color = "red"
        # Plot 2D trajectory up to formation
        ax1.plot(x_full, y_full, linestyle="-", color=t_color, alpha=0.8, linewidth=2.0)
        # Scatter 2D points up to formation
        # ax1.scatter(x, y, s=30, color=t_color, alpha=0.8, marker="o")
        # Plot 2D start point
        start_label = "Start" if j == 0 else ""
        ax1.scatter(x_full[0], y_full[0], s=80, marker="s", color=t_color,
                    edgecolor="darkred", linewidth=1.5, zorder=6, label=start_label)
        # Plot 2D formation point (where formation is reached)
        formation_label = "Final Position" if j == 0 else ""
        ax1.scatter(x_full[-1], y_full[-1], s=100, marker="o", color=t_color,
                    edgecolor="darkred", linewidth=1.5, zorder=7, label=formation_label)

        # Add small black dashed line between start and final position for each target
        ax1.plot([x_full[0], x_full[-1]], [y_full[0], y_full[-1]],
                'k--', alpha=0.6, linewidth=1.0)

    # Get final positions for drawing edges (still useful)
    T_xy = np.array([tg.positions[:2, formation_reached_step] for tg in targets]) if targets else np.empty((0, 2))

    # --- Formatting (updated to include initial + trajectory extents) ---
    # Collect ALL (x,y) points up to formation_reached_step so nothing is off-screen
    xy_points = []
    for tg in targets:
        traj_xy = tg.positions[:2, :formation_reached_step + 1]  # shape (2, T)
        xy_points.append(traj_xy.T)  # shape (T, 2)

    if xy_points:
        P = np.vstack(xy_points)  # shape (K, 2)
        x_min, y_min = P.min(axis=0)
        x_max, y_max = P.max(axis=0)

        # Pad limits a bit
        padding_factor = 0.10
        x_range = max(x_max - x_min, 1e-9)
        y_range = max(y_max - y_min, 1e-9)
        x_min -= x_range * padding_factor
        x_max += x_range * padding_factor
        y_min -= y_range * padding_factor
        y_max += y_range * padding_factor

        # Keep equal aspect by expanding the smaller span
        span = max(x_max - x_min, y_max - y_min)
        mid_x = 0.5 * (x_max + x_min)
        mid_y = 0.5 * (y_max + y_min)

        ax1.set_xlim(mid_x - span / 2, mid_x + span / 2)
        ax1.set_ylim(mid_y - span / 2, mid_y + span / 2)

    # Ensure equal aspect ratio for the 2D plot
    ax1.set_aspect('equal', adjustable='box')

    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()  # legend for "Start" and "Final Position"

    # Add connecting lines between targets in final formation to show shape
    if T_xy.shape[0] >= 2:
        # Connect all targets to show the final shape edges
        for i in range(T_xy.shape[0]):
            for j in range(i + 1, T_xy.shape[0]):
                ax1.plot([T_xy[i, 0], T_xy[j, 0]], [T_xy[i, 1], T_xy[j, 1]],
                        'k--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.show(block=False)

# ---------------------------------------------------------------

def load_configurations() -> list[dict[str, Any]]:
    config_dir = Path("configurations")
    config_files = list(config_dir.glob("*.json"))
    baseline_file = config_dir / "config_common.json"
    baseline_config = {}

    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_config = json.load(f)

    configs = []
    for config_file in sorted(config_files):
        if config_file.name == "config_common.json":
            continue
        with open(config_file, 'r') as f:
            config = json.load(f)
            merged_config = {**baseline_config, **config}
            configs.append(merged_config)

    if not configs and baseline_config:
        configs = [baseline_config]

    return configs

# ---------------------------------------------------------------

def run_batch_simulation_with_results() -> None:
    configs = load_configurations()
    run_simulation_from_configs(configs)
    results()

# ---------------------------------------------------------------

if __name__ == "__main__":
    run_batch_simulation_with_results()
