from __future__ import annotations
import itertools
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def make_offsets_agents_triangle(agent_specs: list[tuple[np.ndarray, dict]],
                                 target_specs: list[tuple[np.ndarray, dict]],
                                 neighbors: list[list[int]],
                                 group: list[int],
                                 target_index: int) -> np.ndarray:
    
    nd = int(agent_specs[0][1]['num_states'])
    N  = len(agent_specs)
    offsets = np.zeros((nd, N))

    target_pos = target_specs[target_index][0]
    agent_positions = np.array([spec[0] for spec in agent_specs])
    l = np.linalg.norm(agent_positions[1] - agent_positions[0])

    # define ideal triangle formation (equilateral)
    tri_formation = [
        np.array([0, l/np.sqrt(3), 0]),
        np.array([l/2, -l*np.sqrt(3)/6, 0]),
        np.array([-l/2, -l*np.sqrt(3)/6, 0])
    ]

    # define local indices within the global group
    g = group[:3]
    desired_positions = [target_pos + vertex for vertex in tri_formation]

    # offset assignments based on neighbor relationships
    for i in range(len(g)):
        for j in neighbors[g[i]]:
            if j in g:
                i_local = g.index(g[i])
                j_local = g.index(j)
                delta_ij = desired_positions[j_local] - desired_positions[i_local]
                delta_ij = desired_positions[j_local] - desired_positions[i_local]
                offsets[:, g[i]] += delta_ij / len(g)
                offsets[:, j]    -= delta_ij / len(g)
    return offsets

# ---------------------------------------------------------------

def make_offsets_agents_square(agent_specs: list[tuple[np.ndarray, dict]],
                               target_specs: list[tuple[np.ndarray, dict]],
                               neighbors: list[list[int]],
                               group: list[int],
                               target_index: int) -> np.ndarray:
    
    nd = int(agent_specs[0][1]['num_states'])
    N  = len(agent_specs)
    offsets = np.zeros((nd, N))

    target_pos = target_specs[target_index][0]
    agent_positions = np.array([spec[0] for spec in agent_specs])
    l = np.linalg.norm(agent_positions[1] - agent_positions[0])

    # ideal vertex positions (square of side l, centered at target)
    square_formation = [
        np.array([-l/2, -l/2, 0]),
        np.array([ l/2, -l/2, 0]),
        np.array([ l/2,  l/2, 0]),
        np.array([-l/2,  l/2, 0])
    ]
    
    # define local indices within the global group
    g = group[:4]
    desired_positions = [target_pos + vertex for vertex in square_formation]

    # offset assignments based on neighbor relationships
    for i in range(len(g)):
        for j in neighbors[g[i]]:
            if j in g:
                i_local = g.index(g[i])
                j_local = g.index(j)
                delta_ij = desired_positions[j_local] - desired_positions[i_local]
                delta_ij = desired_positions[j_local] - desired_positions[i_local]
                offsets[:, g[i]] += delta_ij / len(g)
                offsets[:, j]    -= delta_ij / len(g)
    return offsets

#---------------------------------------------------------------

def make_offsets_targets(target_specs: list[tuple[np.ndarray, dict]],
                         target_offsets: np.ndarray,
                         time_steps: int,
                         base_config: dict[str, Any],
                         neighbors: list[list[int]]) -> np.ndarray:
    """
    Build target offsets for tetrahedral formation, identify centroid target,
    and ensure centroid movement propagates to all others.
    """
    nd = int(target_specs[0][1]['num_states'])
    N = len(target_specs)
    target_offsets = np.zeros((nd, N))

    if target_specs and N >= 4:
        # --- Use first target as geometric anchor (local frame origin) ---
        target_pos_1 = target_specs[0][0]
        target_positions = np.array([spec[0] for spec in target_specs])

        # --- Geometry of tetrahedron (local coordinates) ---
        d = np.linalg.norm(target_positions[1] - target_positions[0])
        h = np.sqrt(6) * d / 3

        tet_formation = [
            np.array([0, 0, 0]),
            np.array([d, 0, 0]),
            np.array([d / 2, np.sqrt(3) * d / 2, 0]),
            np.array([d / 2, np.sqrt(3) * d / 6, h])
        ]

        # --- Compute geometric centroid in local & global frames ---
        # centroid_local = sum(tet_formation) / 4.0
        # centroid_global = target_pos_1 + centroid_local
        # print(f"Geometric centroid of tetrahedron: {centroid_global}")

        # # --- Assign centroid to the target closest to geometric center ---
        # distances = [np.linalg.norm(spec[0] - centroid_global) for spec in target_specs]
        # centroid_idx = int(np.argmin(distances))
        # target_specs[centroid_idx][1]['is_centroid'] = True
        # target_specs[centroid_idx][1]['centroid_global'] = centroid_global.tolist()
        # print(f"Auto-assigned centroid: Target {target_specs[centroid_idx][1]['id']}")

        desired_positions = [target_pos_1 + vertex for vertex in tet_formation]

        # --- Compute relative offsets among targets (local formation) ---
        for i in range(4):
            for j in neighbors[i]:
                if i < j and j < 4:
                    delta_ij = desired_positions[j] - desired_positions[i]
                    target_offsets[:, i] += delta_ij / 4
                    target_offsets[:, j] -= delta_ij / 4

    return target_offsets

# ---------------------------------------------------------------

def chunk_triplets(idxs: list[int]) -> list[list[int]]:
    return [idxs[i:i+3] for i in range(0, len(idxs), 3) if len(idxs[i:i+3]) == 3]

# ---------------------------------------------------------------

def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]

    final_time: float = base_config["final_time"]
    time_step_delta: float = base_config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    np.random.seed(base_config["seed"])

    target_specs = build_entity_specs(base_config, section_key="targets")
    agent_specs  = build_entity_specs(base_config, section_key="agents")

    nd_targets = int(target_specs[0][1]['num_states'])

    nd_agents  = int(agent_specs[0][1]['num_states'])
    N_targets  = len(target_specs)
    N_agents   = len(agent_specs)

    # neighbors on all the agents (then we filter by group inside the function)
    agent_neighbors: list[list[int]] = [[] for _ in range(N_agents)]
    for i, j in base_config["agent_edge_set"]:
        agent_neighbors[i - 1].append(j - 1)
        agent_neighbors[j - 1].append(i - 1)

    # define groups by slicing
    k_square = 4  # last 4 make square
    if N_agents < k_square:
        raise ValueError("Se necesitan al menos 4 agentes para el cuadrado.")

    square_group = list(range(N_agents - k_square, N_agents))
    tri_pool     = list(range(0, N_agents - k_square))  # all the previous ones
    tri_groups   = chunk_triplets(tri_pool)             

    # square around T1, triangles aruond T2 if exists
    sq_t_idx  = 0
    tri_t_idx = 1 if len(target_specs) >= 2 else 0

    # total offsets
    nd_agents = int(agent_specs[0][1]['num_states'])
    offsets_total = np.zeros((nd_agents, N_agents))

    # sum offsets of each tri group
    for g in tri_groups:
        off_tri = make_offsets_agents_triangle(agent_specs, target_specs,
                                               agent_neighbors, g, tri_t_idx)
        offsets_total[:, g] += off_tri[:, g]  # only columns of group 

    # sum offset of square group 
    off_sq = make_offsets_agents_square(agent_specs, target_specs,
                                        agent_neighbors, square_group, sq_t_idx)
    offsets_total[:, square_group] += off_sq[:, square_group]

    # create targets

    target_offsets = np.zeros((nd_targets, len(target_specs)))
    target_neighbors: list[list[int]] = [[] for _ in range(N_targets)]
    for i, j in base_config["target_edge_set"]:
        i0, j0 = i - 1, j - 1
        if 0 <= i0 < N_targets and 0 <= j0 < N_targets:
            target_neighbors[i0].append(j0)
            target_neighbors[j0].append(i0)
            
    target_offsets = make_offsets_targets(target_specs, target_offsets, time_steps, base_config, target_neighbors)

    targets: list[Target] = [
        Target(initial_position=pos, time_steps=time_steps, config=conf)
        for pos, conf in target_specs
    ]
    for i, t in enumerate(targets):
        t.offsets = target_offsets[:, i]

    agents: list[Agent] = [
        Agent(initial_position=pos, time_steps=time_steps, config=conf,
              targets=targets, pin_row=np.zeros(0, dtype=np.float64))
        for pos, conf in agent_specs
    ]

    for i, ag in enumerate(agents):
        ag.offsets = offsets_total[:, i]

    # pinning and targets
    pinning_matrix: NDArray[np.float64] = np.array(base_config["pinning_matrix"], dtype=float)
    for i, ag in enumerate(agents):
        ag.targets = targets
        ag.pin_row = pinning_matrix[i, :].copy()

    # neighborhoods
    construct_undirected_neighborhood_set(targets, base_config.get("target_edge_set", []))
    construct_undirected_neighborhood_set(agents,  base_config.get("agent_edge_set", []))

    for step in range(1, time_steps):
        for ag in agents:  ag.compute_control_output(step)
        for tg in targets: tg.compute_control_output(step)
        for ag in agents:  ag.update_dynamics(step)
        for tg in targets: tg.update_dynamics(step)

        simulation_time: float = step * time_step_delta
        save_state_to_csv(step, simulation_time, agents, targets)  

        print(f"Progress: {step / time_steps * 100:6.2f}%", end="\r", flush=True)



    # print("\nSimulation completed.")
    # print("coordinates of final position of A1:", agents[0].positions[:3, step])
    # print("coordinates of final position of A2:", agents[1].positions[:3, step])
    # print("coordinates of final position of A3:", agents[2].positions[:3, step])
    # print("coordinates of final postion of T1:", targets[0].positions[:3, step] if targets else "N/A")
    # print("------------------------------------------------------------------------------------")
    # print("coordinates of final position of A12:", agents[3].positions[:3, step])
    # print("coordinates of final position of A13:", agents[4].positions[:3, step])
    # print("coordinates of final position of A11:", agents[5].positions[:3, step])
    # print("coordinates of final position of A10:", agents[6].positions[:3, step])
    # print("coordinates of final postion of T4:", targets[3].positions[:3, step] if targets else "N/A")

    # print("------------------------------------------------------------------------------------")

    # # idxs = los 4 del grupo del cuadrado, en el orden g de arriba
    # pts = [agents[i].positions[:3, -1] for i in g]
    # def dist(a,b): 
    #     return float(np.linalg.norm(pts[a]-pts[b]))
    # print("Lados:", dist(0,1), dist(1,2), dist(2,3), dist(3,0))
    # print("Diagonales:", dist(0,2), dist(1,3))

    close_all_files()

# ---------------------------------------------------------------

 

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
