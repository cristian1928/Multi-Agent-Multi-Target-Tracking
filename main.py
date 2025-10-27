
from __future__ import annotations
import itertools
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

    nd = int(target_specs[0][1]['num_states'])
    N = len(target_specs)
    target_offsets = np.zeros((nd, N))

    if target_specs and N >= 4:
        target_pos_1 = target_specs[0][0]
        # target_pos_2 = target_specs[1][0]
        # target_pos_3 = target_specs[2][0]
        # target_pos_4 = target_specs[3][0]

        target_positions = np.array([spec[0] for spec in target_specs])
        d = np.linalg.norm(target_positions[1] - target_positions[0])
        h = np.sqrt(6)*d/3

        tet_formation = [
            np.array([0, 0, 0]),
            np.array([d, 0, 0]),
            np.array([d/2, np.sqrt(3)*d/2, 0]),
            np.array([d/2, np.sqrt(3)*d/6, h])
        ]

        desired_positions_1 = [target_pos_1 + vertex for vertex in tet_formation]
        # desired_positions_2 = [target_pos_2 + vertex for vertex in tet_formation]
        # desired_positions_3 = [target_pos_3 + vertex for vertex in tet_formation]
        # desired_positions_4 = [target_pos_4 + vertex for vertex in tet_formation]

        for i in range(4):
            for j in neighbors[i]:
                if i < j and j < 4:
                    delta_ij_1 = desired_positions_1[j] - desired_positions_1[i]
                    target_offsets[:, i] += delta_ij_1 / 4
                    target_offsets[:, j] -= delta_ij_1 / 4
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

    # --- definir grupos por slicing:
    k_square = 4  # últimos 4 hacen el cuadrado
    if N_agents < k_square:
        raise ValueError("Se necesitan al menos 4 agentes para el cuadrado.")

    square_group = list(range(N_agents - k_square, N_agents))
    tri_pool     = list(range(0, N_agents - k_square))  # todos los anteriores
    tri_groups   = chunk_triplets(tri_pool)             

    # el cuadrado va alrededor de T1 (índice 0), triángulos alrededor de T2 si existe, si no T1
    sq_t_idx  = 0
    tri_t_idx = 1 if len(target_specs) >= 2 else 0

    # --- offsets totales
    nd_agents = int(agent_specs[0][1]['num_states'])
    offsets_total = np.zeros((nd_agents, N_agents))

    # sumar offsets de cada grupo triangular
    for g in tri_groups:
        off_tri = make_offsets_agents_triangle(agent_specs, target_specs,
                                               agent_neighbors, g, tri_t_idx)
        offsets_total[:, g] += off_tri[:, g]  # solo columnas del grupo

    # sumar offsets del grupo cuadrado
    off_sq = make_offsets_agents_square(agent_specs, target_specs,
                                        agent_neighbors, square_group, sq_t_idx)
    offsets_total[:, square_group] += off_sq[:, square_group]

    # --- crear targets

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

    # pinning y targets
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


    # ===================== PLOTTING =====================

    def _plot_group_3d(ax, objs, color, start_lbl, end_lbl, zorder=3):
        if not objs: 
            return
        for idx, o in enumerate(objs):
            xyz = o.positions[:3, :end_step + 1]
            if xyz.shape[1] == 0:
                continue
            x, y, z = xyz[0], xyz[1], xyz[2]
            ax.plot(x, y, z, '-', color=color, alpha=0.9, linewidth=2.0, zorder=zorder)
            ax.scatter(x[0],  y[0],  z[0],  s=40, marker='s', color=color, edgecolor='k',
                    linewidth=0.6, zorder=zorder+1)
            ax.scatter(x[-1], y[-1], z[-1], s=55, marker='o', color=color, edgecolor='k',
                    linewidth=0.6, zorder=zorder+2)
            ax.plot([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]],
                    'k--', alpha=0.45, linewidth=0.9, zorder=zorder)

    def _plot_group_center(ax, group_indices, color, label=None):
        if not group_indices:
            return
        pts = [agents[i].positions[:3, end_step] for i in group_indices]
        P = np.stack(pts, axis=0)  # (m, 3)
        c = P.mean(axis=0)
        ax.scatter(c[0], c[1], c[2], s=160, marker='o', color=color, edgecolor='k',
                linewidth=1.0, zorder=10)

    # --- TETRA styling ---
    _TET_EDGE_W  = 2.8
    _TET_VERT_S  = 120
    _TET_VERT_EC = 'k'
    _TET_VERT_LW = 1.2

    def _plot_targets_and_tetra(ax):
        # target trajectories
        _plot_group_3d(ax, targets, color='red',
                    start_lbl='Target start', end_lbl='Target final', zorder=7)

        # draw tetrahedron at final step using the first 4 targets
        if len(targets) >= 4:
            T_xyz = np.array([tg.positions[:3, end_step] for tg in targets[:4]])
            # vertices (bigger, outlined)
            ax.scatter(T_xyz[:,0], T_xyz[:,1], T_xyz[:,2],
                    s=_TET_VERT_S, marker='o', color='red',
                    edgecolor=_TET_VERT_EC, linewidth=_TET_VERT_LW, zorder=9)
            # edges
            tet_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
            for (i, j) in tet_edges:
                ax.plot([T_xyz[i,0], T_xyz[j,0]],
                        [T_xyz[i,1], T_xyz[j,1]],
                        [T_xyz[i,2], T_xyz[j,2]],
                        '-', color='red', alpha=0.95, linewidth=_TET_EDGE_W, zorder=8)

    def _finalize_3d(ax):

        xyz_chunks = []
        for obj in (targets or []):
            xyz_chunks.append(obj.positions[:3, :end_step + 1].T)
        for obj in (agents or []):
            xyz_chunks.append(obj.positions[:3, :end_step + 1].T)

        T_final = None
        if len(targets) >= 4:
            T_final = np.array([tg.positions[:3, end_step] for tg in targets[:4]])

        if xyz_chunks:
            P = np.vstack(xyz_chunks)
            mins = P.min(axis=0); maxs = P.max(axis=0)
            spans = np.maximum(maxs - mins, 1e-9)

            pad = 0.35  
            mins = mins - spans * pad
            maxs = maxs + spans * pad

            span = float(np.max(maxs - mins))
            if T_final is not None:
                cT = T_final.mean(axis=0)
                boost = 1.6  
                span = span * boost
                center = cT
            else:
                center = (maxs + mins) / 2.0

            xlim = (center[0] - span/2, center[0] + span/2)
            ylim = (center[1] - span/2, center[1] + span/2)
            zlim = (center[2] - span/2, center[2] + span/2)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
            try: ax.set_box_aspect((1, 1, 1))
            except Exception: pass

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.grid(True)
        # NOTE: legend and title intentionally omitted

    formation_reached_step = time_steps - 1
    formation_threshold = 0.1
    check_stride = 10
    if targets:
        for step in range(0, time_steps - 1, check_stride):
            deltas = [np.linalg.norm(tg.positions[:3, step + 1] - tg.positions[:3, step]) for tg in targets]
            if all(d <= formation_threshold for d in deltas):
                formation_reached_step = step
                break

    cut_at_formation = False
    end_step = formation_reached_step if cut_at_formation else (time_steps - 1)

    target_ids = [conf.get("id", f"T{i+1}") for i, (_, conf) in enumerate(target_specs)]
    agent_ids  = [conf.get("id", f"A{i+1}") for i, (_, conf) in enumerate(agent_specs)]

    square_color = 'tab:blue'
    tri_colors   = ['tab:purple', 'tab:green', 'tab:orange', 'tab:brown']

    # ========== Figure A: Tetra + Square + Triangles ==========
    figA = plt.figure(figsize=(8, 7))
    axA  = figA.add_subplot(111, projection='3d')
    _plot_targets_and_tetra(axA)

    # Square group + center
    if 'square_group' in locals() and square_group:
        sq_objs = [agents[i] for i in square_group]
        _plot_group_3d(axA, sq_objs, color=square_color,
                    start_lbl='Square start', end_lbl='Square final', zorder=4)
        _plot_group_center(axA, square_group, color=square_color, label=None)

    # All triangles + centers
    if 'tri_groups' in locals() and tri_groups:
        for idx, g in enumerate(tri_groups):
            col = tri_colors[idx % len(tri_colors)]
            tri_objs = [agents[i] for i in g]
            _plot_group_3d(axA, tri_objs, color=col,
                        start_lbl=f'Tri#{idx+1} start', end_lbl=f'Tri#{idx+1} final', zorder=4)
            _plot_group_center(axA, g, color=col, label=None)

    _finalize_3d(axA)

    # ========== Figure B: Tetra + Triangle #1 ==========
    if 'tri_groups' in locals() and len(tri_groups) >= 1:
        figB = plt.figure(figsize=(8, 7))
        axB  = figB.add_subplot(111, projection='3d')
        _plot_targets_and_tetra(axB)
        g0 = tri_groups[0]
        _plot_group_3d(axB, [agents[i] for i in g0],
                    color=tri_colors[0], start_lbl='Tri#1 start', end_lbl='Tri#1 final', zorder=4)
        _plot_group_center(axB, g0, color=tri_colors[0], label=None)
        _finalize_3d(axB)

    # ========== Figure C: Tetra + Triangle #2 ==========
    if 'tri_groups' in locals() and len(tri_groups) >= 2:
        figC = plt.figure(figsize=(8, 7))
        axC  = figC.add_subplot(111, projection='3d')
        _plot_targets_and_tetra(axC)
        g1 = tri_groups[1]
        _plot_group_3d(axC, [agents[i] for i in g1],
                    color=tri_colors[1], start_lbl='Tri#2 start', end_lbl='Tri#2 final', zorder=4)
        _plot_group_center(axC, g1, color=tri_colors[1], label=None)
        _finalize_3d(axC)

    plt.tight_layout()
    plt.show(block=False)
    # =================== END PLOTTING ====================



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
