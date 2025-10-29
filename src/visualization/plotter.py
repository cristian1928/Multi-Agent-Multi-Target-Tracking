import os
from typing import Any, Dict, List, Tuple, cast
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # type: ignore
import re

_num_re = re.compile(r'(\d+)\D*$')

def _num_key(filename: str) -> int:
    base = os.path.basename(filename)
    # quita el sufijo estándar de estado si está presente
    name = base.replace(STATE_DATA_SUFFIX, '')
    m = _num_re.search(name)
    return int(m.group(1)) if m else 10**9

# Constants for data access
DATA_DIR = 'simulation_data'
AGENT_DATA_DIR = os.path.join(DATA_DIR, 'agent_data')
TARGET_DATA_DIR = os.path.join(DATA_DIR, 'target_data')
STATE_DATA_SUFFIX = '_state_data.csv'
NN_DATA_SUFFIX = '_nn_data.csv'

# def configure_plot() -> None:
#     plt.style.use(['science', 'ieee'])
#     plt.rcParams['figure.dpi'] = 100
#     plt.rcParams["font.family"] = "serif"
#     plt.rcParams["axes.labelsize"] = 12
#     plt.rcParams["axes.titlesize"] = 16
#     plt.rcParams["xtick.labelsize"] = 12
#     plt.rcParams["ytick.labelsize"] = 12
#     plt.rcParams.update({
#         'lines.linewidth': 1.5,
#         'axes.linewidth': 0.5,
#         'legend.frameon': True,
#         'legend.edgecolor': 'black',
#     })

def get_simulation_data() -> Tuple[List[str], List[pd.DataFrame], List[str], List[pd.DataFrame]]:
    agent_state_files = sorted(
        [f for f in os.listdir(AGENT_DATA_DIR) if f.endswith(STATE_DATA_SUFFIX)],
        key=_num_key
    ) if os.path.isdir(AGENT_DATA_DIR) else []

    target_state_files = sorted(
        [f for f in os.listdir(TARGET_DATA_DIR) if f.endswith(STATE_DATA_SUFFIX)],
        key=_num_key
    ) if os.path.isdir(TARGET_DATA_DIR) else []

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
    # configure_plot()
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
        axis_error.set_ylabel('Synchronization Error Norm $\| \eta \|$ $(m)$')

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

# adding 3d trajectories

            # ====== CARGA DE CONFIG Y RECONSTRUCCIÓN DE EDGES ======
    import json
    from pathlib import Path

    def _load_merged_config():
        cfg_dir = Path("configurations")
        baseline = {}
        base_file = cfg_dir / "config_common.json"
        if base_file.exists():
            with open(base_file, "r") as f:
                baseline = json.load(f)
        # coge el primer config “concreto” como hace tu main
        others = sorted([p for p in cfg_dir.glob("*.json") if p.name != "config_common.json"])
        merged = baseline.copy()
        if others:
            with open(others[0], "r") as f:
                merged.update(json.load(f))
        return merged if merged else None

    cfg = _load_merged_config()

    # Helper para dibujar una línea 2D
    def _line2d(ax, p, q, c, lw=1.8, a=0.95, ls='-'):
        ax.plot([p[0], q[0]], [p[1], q[1]],
                linestyle=ls, color=c, alpha=a, linewidth=lw, zorder=5)

    # Reconstruye grupos y vecindarios de agentes como en main
    square_group, tri_groups, agent_neighbors = [], [], []
    target_edge_set = []
    pinning_matrix = None

    if cfg:
        # target edges
        target_edge_set = cfg.get("target_edge_set", []) or []

        # pinning
        pm = cfg.get("pinning_matrix", None)
        if pm is not None:
            pinning_matrix = np.array(pm, dtype=float)

        # agent neighbors
        N_agents = len(agent_names)
        agent_neighbors = [[] for _ in range(N_agents)]
        for i, j in cfg.get("agent_edge_set", []):
            i0, j0 = i - 1, j - 1
            if 0 <= i0 < N_agents and 0 <= j0 < N_agents:
                agent_neighbors[i0].append(j0)
                agent_neighbors[j0].append(i0)

        # grupos: últimos 4 al cuadrado; anteriores en tríos
        k_square = 4
        if N_agents >= k_square:
            square_group = list(range(N_agents - k_square, N_agents))
            tri_pool = list(range(0, N_agents - k_square))
            tri_groups = [tri_pool[i:i+3] for i in range(0, len(tri_pool), 3) if len(tri_pool[i:i+3]) == 3]

    def pos_at(df: pd.DataFrame, step: int) -> Tuple[float, float]:
        s = max(0, min(step, len(df) - 1))
        return float(df.iloc[s]['Position X']), float(df.iloc[s]['Position Y'])

    # Mapas rápidos: posiciones XY en un paso (ya tienes pos_at)
    def _T_now(step):
        return [pos_at(df, step) for df in targets_state_data]

    def _A_now(step):
        return [pos_at(df, step) for df in agents_state_data]

    # --- 2D snapshots desde CSV (evita depender de objetos runtime como agents/targets)x

    # Figura con dos subplots: t=0 y t=tf
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)

    # ---------- ESQUEMA DE COLORES (mismos nombres que usabas) ----------
    colors = {
        "square": 'tab:blue',
        "tris":   ['tab:purple', 'tab:green', 'tab:orange', 'tab:brown'],
        "tt_link": 'tab:red',
        "at_link": 'tab:cyan',
        "inter_form_link": 'tab:olive'
    }
    link_alpha = 0.95  # (no se usa, pero lo dejamos para coherencia)

    # Ya tienes estos mapas arriba; los reusamos aquí:
    #   agent_color_map = get_color_map(agent_names, cmap_name='tab20')
    #   target_color_map = get_color_map(target_names, cmap_name='tab10')

    # ---------- Pasos t0 y tf a partir de la longitud de los CSV ----------
    max_len = 0
    for df in agents_state_data + targets_state_data:
        max_len = max(max_len, len(df))
    if max_len == 0:
        # No hay datos; nada que dibujar
        plt.show(block=False)
        # (seguimos sin devolver para que el resto del flujo de tu función continue)
    else:
        t0 = 0
        tf = max_len - 1
        steps = [t0, tf]

        # ---------- Límites globales que cubran ambos pasos y todos los puntos ----------
        all_pts = []
        for step in steps:
            for df in targets_state_data:
                all_pts.append(pos_at(df, step))
            for df in agents_state_data:
                all_pts.append(pos_at(df, step))

        if all_pts:
            P = np.array(all_pts, dtype=float).reshape(-1, 2)
            mins = P.min(axis=0)
            maxs = P.max(axis=0)
            span = np.maximum(maxs - mins, 1e-9)
            pad = 0.10 * float(np.max(span))
            xlim = (mins[0] - pad, maxs[0] + pad)
            ylim = (mins[1] - pad, maxs[1] + pad)
        else:
            xlim = (-1.0, 1.0)
            ylim = (-1.0, 1.0)

        # ---------- Dibujo por subplot ----------
        for ax, step in zip(axes, steps):
            # Targets (X)
            for name, df in zip(target_names, targets_state_data):
                x, y = pos_at(df, step)
                c = target_color_map.get(name, 'red')
                ax.scatter(x, y, s=90, marker='X', color=c, edgecolor='k',
                           linewidth=0.7, zorder=7)

            # Agents (o)
            for name, df in zip(agent_names, agents_state_data):
                x, y = pos_at(df, step)
                c = agent_color_map.get(name, 'gray')
                ax.scatter(x, y, s=70, marker='o', color=c, edgecolor='k',
                           linewidth=0.6, zorder=6)

            # Estilo de ejes
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("X $(m)$")
            ax.set_ylabel("Y $(m)")
            ax.set_axisbelow(True)
            ax.grid(True, linestyle='--', alpha=0.5)
            # ax.set_title(f"t = {'0' if step == t0 else 't_f'}")

                    # ----- Target↔Target (TT) -----
            if target_edge_set and target_names:
                Tnow = _T_now(step)
                for i, j in target_edge_set:
                    i0, j0 = i - 1, j - 1
                    if 0 <= i0 < len(Tnow) and 0 <= j0 < len(Tnow):
                        _line2d(ax, Tnow[i0], Tnow[j0], colors["tt_link"], lw=2.2)

            # ----- Agent→Target (AT) -----
            if pinning_matrix is not None and len(agents_state_data) == pinning_matrix.shape[0]:
                Anow = _A_now(step)
                Tnow = _T_now(step)
                for ai in range(pinning_matrix.shape[0]):
                    row = pinning_matrix[ai, :]
                    for tj, w in enumerate(row):
                        if w != 0.0 and 0 <= tj < len(Tnow):
                            _line2d(ax, Anow[ai], Tnow[tj], colors["at_link"], lw=1.6, ls=':', a=0.9)

            # ----- Agent↔Agent (AA) -----
            if agent_neighbors:
                Anow = _A_now(step)

                # Triángulos (línea sólida por grupo)
                for gi, g in enumerate(tri_groups or []):
                    col = colors["tris"][gi % len(colors["tris"])]
                    for i in g:
                        for j in agent_neighbors[i]:
                            if j <= i or j not in g:
                                continue
                            _line2d(ax, Anow[i], Anow[j], col, lw=1.8)

                # Cuadrado (línea sólida)
                if square_group:
                    col = colors["square"]
                    for i in square_group:
                        for j in agent_neighbors[i]:
                            if j <= i or j not in square_group:
                                continue
                            _line2d(ax, Anow[i], Anow[j], col, lw=1.8)

                # Inter-formación (punteada)
                # etiqueta de grupo por índice de agente
                agent_label = {}
                for idx in square_group or []:
                    agent_label[idx] = 'square'
                for k, g in enumerate(tri_groups or []):
                    for idx in g:
                        agent_label[idx] = f'tri{k}'

                for i in range(len(Anow)):
                    for j in agent_neighbors[i]:
                        if j <= i:
                            continue
                        li = agent_label.get(i)
                        lj = agent_label.get(j)
                        if li is not None and lj is not None and li != lj:
                            _line2d(ax, Anow[i], Anow[j], colors["inter_form_link"], lw=1.6, ls=':', a=0.9)


        # ---------- Leyenda inferior de tipos de enlace (informativa) ----------
        legend_handles = [
            Line2D([0], [0], color=colors["tt_link"],   lw=2.2, label='Target-Target'),
            Line2D([0], [0], color=colors["at_link"],   lw=1.6, linestyle=':', label='Agent-Target'),
            Line2D([0], [0], color=colors["square"],    lw=1.8, label='Agent-Agent (within formation)'),
            Line2D([0], [0], color=colors["inter_form_link"], lw=1.6, linestyle=':', label='Agent-Agent (out of formation)'),
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=2,
            frameon=True,
            fontsize='small',
            bbox_to_anchor=(0.5, 0.06)
        )
        fig.subplots_adjust(bottom=0.22)

    # mantenemos tu show no-bloqueante
    plt.show(block=False)


#--
    plt.show()

def results() -> None:
    plot_from_csv()

if __name__ == "__main__":
    results()
