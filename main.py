
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

    t0 = 0
    tm = time_steps // 2   # still unused
    tf = time_steps - 1

    def _plot_snapshots_2d_subplots():
        import numpy as _np
        from matplotlib.lines import Line2D
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)

        # ---------- COLOR SCHEME ----------
        colors = {
            "square": 'tab:blue',
            "tris":   ['tab:purple', 'tab:green', 'tab:orange', 'tab:brown'],
            "tt_link": 'tab:red',
            "at_link": 'tab:cyan',
            "inter_form_link": 'tab:olive'
        }
<<<<<<< HEAD

        # One distinct color per agent (tab20 cycles)
        tab20 = plt.get_cmap('tab20').colors
        agent_colors = {i: tab20[i % len(tab20)] for i in range(len(agents))}
        link_alpha = 0.95

        # Helper for 2D lines
        def _line2d(ax, p, q, c, lw=1.8, a=link_alpha, ls='-'):
=======

        # NEW: one distinct color per agent (tab20 cycles after 20)
        tab20 = plt.get_cmap('tab20').colors
        agent_colors = {i: tab20[i % len(tab20)] for i in range(len(agents))}

        link_alpha = 0.95
        
        # helper for 2D lines
        def _line2d(p, q, c, lw=1.8, a=link_alpha, ls='-'):
>>>>>>> faa253b9248770939428bc40fcd34801a7ec8faf
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    linestyle=ls, color=c, alpha=a, linewidth=lw, zorder=8)

        # Build labels for formations
        agent_label = {}
        if 'square_group' in locals() and square_group:
            for idx in square_group:
                agent_label[idx] = 'square'
        if 'tri_groups' in locals() and tri_groups:
            for k, g in enumerate(tri_groups):
                for idx in g:
                    agent_label[idx] = f'tri{k}'

        # Steps to render
        steps = [t0, tf]

<<<<<<< HEAD
        # ---------- Compute global limits across both steps ----------
        all_pts = []
        for step in steps:
            if targets:
                for tg in targets:
                    all_pts.append(tg.positions[:2, step])
            if agents:
                for ag in agents:
                    all_pts.append(ag.positions[:2, step])

        if all_pts:
            P = _np.vstack([p.reshape(1, 2) for p in all_pts])
            mins = P.min(axis=0); maxs = P.max(axis=0)
            span = _np.maximum(maxs - mins, 1e-9)
            pad = 0.10 * float(_np.max(span))
            xlim = (mins[0] - pad, maxs[0] + pad)
            ylim = (mins[1] - pad, maxs[1] + pad)
        else:
            xlim = (-1, 1); ylim = (-1, 1)

        # ---------- Draw each subplot ----------
        for ax, step in zip(axes, steps):
            # Current XY positions
            T_now = [tg.positions[:2, step] for tg in targets] if targets else []
            A_now = [ag.positions[:2, step] for ag in agents]  if agents  else []

            # Targets (points only)
            if targets:
                for p in T_now:
                    ax.scatter(p[0], p[1], s=90, marker='X', color='red',
                            edgecolor='k', linewidth=0.7, zorder=7)

            # Agents (points only, per-agent colors)
            if 'square_group' in locals() and square_group:
                for i in square_group:
                    p = A_now[i]
                    ax.scatter(p[0], p[1], s=70, marker='o',
                            color=agent_colors[i],
                            edgecolor='k', linewidth=0.6, zorder=6)
=======
        # ---------- SCATTER TARGETS (points only) ----------
        # Plotted as red markers for readability (legend gives distinct colors per target name)
        if targets:
            for j, p in enumerate(T_now):
                ax.scatter(p[0], p[1], s=90, marker='X', color='red',
                        edgecolor='k', linewidth=0.7, zorder=7)
                
        # ---------- SCATTER AGENTS (points only, per-agent colors) ----------
        if 'square_group' in locals() and square_group:
            for i in square_group:
                p = A_now[i]
                ax.scatter(
                    p[0], p[1],
                    s=70, marker='o',
                    color=agent_colors[i],         # ← unique color per agent
                    edgecolor='k', linewidth=0.6, zorder=6
                )

        if 'tri_groups' in locals() and tri_groups:
            for gi, g in enumerate(tri_groups):
                for i in g:
                    p = A_now[i]
                    ax.scatter(
                        p[0], p[1],
                        s=65, marker='o',
                        color=agent_colors[i],     # ← unique color per agent
                        edgecolor='k', linewidth=0.6, zorder=6
                    )

        # ---------- LINKS ----------
        # Target↔Target (one color)
        if targets and "target_edge_set" in base_config:
            for i, j in base_config["target_edge_set"]:
                i0, j0 = i - 1, j - 1
                if 0 <= i0 < len(T_now) and 0 <= j0 < len(T_now):
                    _line2d(T_now[i0], T_now[j0], colors["tt_link"], lw=2.2)

        # Agent→Target (pinning) (one color)
        if agents and targets:
            for i, ag in enumerate(agents):
                if getattr(ag, "pin_row", _np.zeros(0)).size:
                    for t_idx, w in enumerate(ag.pin_row):
                        if w != 0.0 and 0 <= t_idx < len(T_now):
                            _line2d(A_now[i], T_now[t_idx], colors["at_link"], lw=1.8)

        # Intra-formation Agent↔Agent (formation color)
        if agents:
            # Triangles
>>>>>>> faa253b9248770939428bc40fcd34801a7ec8faf
            if 'tri_groups' in locals() and tri_groups:
                for gi, g in enumerate(tri_groups):
                    for i in g:
                        p = A_now[i]
                        ax.scatter(p[0], p[1], s=65, marker='o',
                                color=agent_colors[i],
                                edgecolor='k', linewidth=0.6, zorder=6)

            # Target↔Target links
            if targets and "target_edge_set" in base_config:
                for i, j in base_config["target_edge_set"]:
                    i0, j0 = i - 1, j - 1
                    if 0 <= i0 < len(T_now) and 0 <= j0 < len(T_now):
                        _line2d(ax, T_now[i0], T_now[j0], colors["tt_link"], lw=2.2)

            # Agent→Target (pinning)
            if agents and targets:
                for i, ag in enumerate(agents):
                    row = getattr(ag, "pin_row", _np.zeros(0))
                    if row.size:
                        for t_idx, w in enumerate(row):
                            if w != 0.0 and 0 <= t_idx < len(T_now):
                                _line2d(ax, A_now[i], T_now[t_idx], colors["at_link"], lw=1.8)

            # Intra-formation Agent↔Agent
            if agents:
                # Triangles
                if 'tri_groups' in locals() and tri_groups:
                    for gi, g in enumerate(tri_groups):
                        col = colors["tris"][gi % len(colors["tris"])]
                        for i in g:
                            for nb in agents[i].neighbors:
                                try:
                                    j = agents.index(nb)
                                except ValueError:
                                    continue
                                if j <= i or j not in g:
                                    continue
                                _line2d(ax, A_now[i], A_now[j], col, lw=1.8)
                # Square
                if 'square_group' in locals() and square_group:
                    col = colors["square"]
                    for i in square_group:
                        for nb in agents[i].neighbors:
                            try:
                                j = agents.index(nb)
                            except ValueError:
                                continue
                            if j <= i or j not in square_group:
                                continue
                            _line2d(ax, A_now[i], A_now[j], col, lw=1.8)

            # Cross-formation Agent↔Agent
            if agents:
                for i, ag in enumerate(agents):
                    for nb in ag.neighbors:
                        try:
                            j = agents.index(nb)
                        except ValueError:
                            continue
                        if j <= i:
                            continue
                        li = agent_label.get(i, None); lj = agent_label.get(j, None)
                        if li is not None and lj is not None and li != lj:
                            _line2d(ax, A_now[i], A_now[j], colors["inter_form_link"], lw=1.6)

            # Axis limits & style (no titles, no per-axes legends)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.grid(True)

        # ---------- Figure-level legend at bottom (four link types only) ----------
        # Representative color for "within formation" (use square color as exemplar)
        within_form_color = colors["square"]

        legend_handles = [
            Line2D([0], [0], color=colors["tt_link"],   lw=2.2, label='Target–Target connection'),
            Line2D([0], [0], color=colors["at_link"],   lw=1.8, label='Agent→Target connection'),
            Line2D([0], [0], color=within_form_color,   lw=1.8, label='Agent–Agent (within formation)'),
            Line2D([0], [0], color=colors["inter_form_link"], lw=1.6, label='Agent–Agent (out of formation)'),
        ]

        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=2,
            frameon=True,
            fontsize='small',
            bbox_to_anchor=(0.5, -0.02)  # slightly below the subplots
        )
        # Make room for the bottom legend
        fig.subplots_adjust(bottom=0.18)

        plt.show(block=False)

    # Call once (two subplots: t=0 and t=Tf, with bottom legend)
    _plot_snapshots_2d_subplots()



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
