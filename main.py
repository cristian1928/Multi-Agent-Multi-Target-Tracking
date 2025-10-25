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

#---------------------------------------------------------------

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

#---------------------------------------------------------------

def construct_undirected_neighborhood_set(entities: list[Any], edge_set: list[list[int]]) -> None:
    for i, j in edge_set:
        a, b = entities[i - 1], entities[j - 1]
        if b not in a.neighbors: a.neighbors.append(b)
        if a not in b.neighbors: b.neighbors.append(a)

#---------------------------------------------------------------

def make_offsets(agent_specs: list[tuple[np.ndarray, dict]],
                 agent_offsets: np.ndarray,
                 target_specs: list[tuple[np.ndarray, dict]],
                 time_steps: int,
                 base_config: dict[str, Any],
                 neighbors: list[list[int]]) -> np.ndarray:

    nd = int(agent_specs[0][1]['num_states'])
    N = len(agent_specs)
    agent_offsets = np.zeros((nd, N))

    if target_specs and N >= 3:
        target_pos = target_specs[0][0]
        agent_positions = np.array([spec[0] for spec in agent_specs])
        
        # Use first two agents to determine side length l
        l = np.linalg.norm(agent_positions[1] - agent_positions[0])
        
        # Define relative formation vectors δ(i,j) for triangle
        triangle_vertices = [
            np.array([0,     l/np.sqrt(3),     0]),       # top vertex relative to triangle center
            np.array([l/2,  -l*np.sqrt(3)/6,   0]),      # bottom right relative to triangle center
            np.array([-l/2, -l*np.sqrt(3)/6,   0])       # bottom left relative to triangle center
        ]
        
        # Center the triangle so geometric center is at the target
        triangle_vertices = np.array(triangle_vertices)
        centroid = np.mean(triangle_vertices, axis=0)
        triangle_vertices_centered = triangle_vertices - centroid
        desired_positions = [target_pos + vertex for vertex in triangle_vertices_centered]

        # Desired absolute positions around target
        desired_positions = [target_pos + v for v in triangle_vertices_centered]
        
        # Calculate offsets (same as before)
        for i in range(N):
            for j in neighbors[i]:
                if i < j:
                    delta_ij = desired_positions[j] - desired_positions[i]
                    agent_offsets[:, i] += delta_ij
                    agent_offsets[:, j] -= delta_ij

        print("=== Inter-Agent Distances ===")
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(desired_positions[i] - desired_positions[j])
                print(f"Distance A{i}-A{j}: {dist:.3f}")
        print("=============================")

    return agent_offsets

#---------------------------------------------------------------

def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]

    final_time: float = base_config["final_time"]
    time_step_delta: float = base_config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    np.random.seed(base_config["seed"])

    # Build specs
    target_specs = build_entity_specs(base_config, section_key="targets")
    agent_specs = build_entity_specs(base_config, section_key="agents")

    # number of states and agents
    nd = int(agent_specs[0][1]['num_states'])
    N = len(agent_specs)

    # construct neighbors list (0-indexed)
    neighbors: list[list[int]] = [[] for _ in range(N)]
    for i, j in base_config["agent_edge_set"]:
        neighbors[i - 1].append(j - 1)
        neighbors[j - 1].append(i - 1)

    # compute offsets

    agent_offsets = np.zeros((nd, N))
    agent_offsets = make_offsets(agent_specs, agent_offsets, target_specs, time_steps, base_config, neighbors)

    # create targets

    targets: list[Target] = [Target(initial_position=pos,
                                time_steps=time_steps,
                                config=conf)
                                for (pos, conf) in target_specs]

    pinning_matrix: NDArray[np.float64] = np.array(base_config["pinning_matrix"], dtype=float)

    # create agents

    agents: list[Agent] = [
    Agent(
        initial_position=pos,
        time_steps=time_steps,
        offset=agent_offsets[:, i],  
        config=conf,
        targets=targets,
        pin_row=pinning_matrix[i, :]
    )
    for i, (pos, conf) in enumerate(agent_specs)
]
    for index, agent in enumerate(agents):
        agent.targets = targets
        agent.pin_row = pinning_matrix[index, :].copy()


    # Construct neighborhoods for agents and targets
    construct_undirected_neighborhood_set(agents, base_config["agent_edge_set"])
    construct_undirected_neighborhood_set(targets, base_config.get("target_edge_set", []))

#---------------------------------------------------------------

    for step in range(1, time_steps):
        for agent in agents: 
            agent.compute_control_output(step)
        for target in targets: 
            target.compute_control_output(step)

        for agent in agents: 
            agent.update_dynamics(step)
        for target in targets: 
            target.update_dynamics(step)

        simulation_time: float = step * time_step_delta
        save_state_to_csv(step, simulation_time, agents, targets)

        print(f"Progress: {step / time_steps * 100:6.2f}%", end="\r", flush=True)

    print("\nSimulation completed.")
    close_all_files()

#---------------------------------------------------------------

    agent_ids  = [conf.get("id", f"A{i+1}") for (_, conf) in agent_specs]
    target_ids = [conf.get("id", f"T{j+1}") for (_, conf) in target_specs]

    A_xy = np.array([ag.positions[:2, -1] for ag in agents]) if agents else np.empty((0, 2))
    T_xy = np.array([tg.positions[:2, -1] for tg in targets]) if targets else np.empty((0, 2))

    fig1, ax1 = plt.subplots(figsize=(6,6))
    if A_xy.size:
        ax1.scatter(A_xy[:, 0], A_xy[:, 1], s=100, label="Agents", marker="o", color="blue")
        for (x, y), lab in zip(A_xy, agent_ids):
            ax1.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points")
    if T_xy.size:
        ax1.scatter(T_xy[:, 0], T_xy[:, 1], s=100, label="Target", marker="o", color="red")
        for (x, y), lab in zip(T_xy, target_ids):
            ax1.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points")
    ax1.axis("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Agent Formation Around Target")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    plt.show(block=False)

#---------------------------------------------------------------

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

#---------------------------------------------------------------

def run_batch_simulation_with_results() -> None:
    configs = load_configurations()
    run_simulation_from_configs(configs)
    results()

if __name__ == "__main__":
    run_batch_simulation_with_results()