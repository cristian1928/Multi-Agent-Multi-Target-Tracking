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
        l = np.linalg.norm(agent_positions[1] - agent_positions[0])
        
        triangle_vertices = [
            np.array([0,     l/np.sqrt(3),     0]),   
            np.array([l/2,  -l*np.sqrt(3)/6,   0]),      
            np.array([-l/2, -l*np.sqrt(3)/6,   0])       
        ]
        
        desired_positions = [target_pos + vertex for vertex in triangle_vertices]
    
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



    fig1, ax1 = plt.subplots(figsize=(8, 8))
    step_skip = 10  # adjust for smoother or more discrete visualization

    # --- Plot agent trajectories ---
    for i, agent in enumerate(agents):
        traj = agent.positions[:2, :]
        x, y = traj[0, ::step_skip], traj[1, ::step_skip]
        color = "black"  # All agents in black

        # Discrete points + matching dashed path
        ax1.plot(x, y, linestyle="--", color=color, alpha=0.8, linewidth=1.8)
        ax1.scatter(x, y, s=25, color=color, alpha=0.6, label=f"Agent {i+1}" if i == 0 else "")

        # Start and end points
        ax1.scatter(traj[0, 0], traj[1, 0], s=70, marker="s", color=color, edgecolor="darkgray", zorder=5)
        ax1.scatter(traj[0, -1], traj[1, -1], s=100, marker="o", color=color, edgecolor="darkgray", zorder=6)
        ax1.text(traj[0, -1] + 0.1, traj[1, -1] + 0.1, f"A{i+1}", fontsize=9, color=color)

    # --- Plot target trajectories ---
    for j, target in enumerate(targets):
        traj = target.positions[:2, :]
        x, y = traj[0, ::step_skip], traj[1, ::step_skip]
        t_color = "red"  # All targets in red

        ax1.plot(x, y, linestyle="--", color=t_color, linewidth=2.0, alpha=0.8)
        ax1.scatter(x, y, s=35, color=t_color, alpha=0.7, label=f"Target {j+1}" if j == 0 else "")

        # Start and end
        ax1.scatter(traj[0, 0], traj[1, 0], s=90, marker="P", color=t_color, edgecolor="darkred", zorder=6)
        ax1.scatter(traj[0, -1], traj[1, -1], s=140, marker="o", color=t_color, edgecolor="darkred", zorder=7)
        ax1.text(traj[0, -1] + 0.1, traj[1, -1] + 0.1, f"T{j+1}", fontsize=10, color=t_color)

    # --- Formatting ---
    ax1.axis("equal")
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Agent Formation", fontsize=14, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Add legend entries for agents and targets (only one entry each)
    '''ax1.plot([], [], linestyle="--", color="black", linewidth=1.8, label="Agents")
    ax1.plot([], [], linestyle="--", color="red", linewidth=2.0, label="Targets")
    ax1.legend(loc="upper right", fontsize=9)'''
    
    plt.tight_layout()
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