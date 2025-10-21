from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any,  List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.entity import Agent, Target
from src.io.data_manager import close_all_files, save_state_to_csv
from src.visualization.plotter import results
from src.simulation.dynamics import get_initial_conditions

# Helper functions
def build_entity_specs(base_config: dict[str, Any], section_key: str) -> List[Tuple[NDArray[np.float64], dict[str, Any]]]:
    specs: List[Tuple[NDArray[np.float64], dict[str, Any]]] = []
    for index, item in enumerate(base_config[section_key]):
        dynamics_type = item["dynamics_type"]
        initial_position = np.array(item.get("initial_position", get_initial_conditions(dynamics_type)), dtype=float)
        merged_config = {**base_config, **item, "dynamics_type": dynamics_type}
        specs.append((initial_position, merged_config))
    return specs

def construct_undirected_neighborhood_set(entities: list[Any], edge_set: list[list[int]]) -> None:
    for i, j in edge_set:
        a, b = entities[i - 1], entities[j - 1]
        if b not in a.neighbors: a.neighbors.append(b)
        if a not in b.neighbors: b.neighbors.append(a)

# Main
def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]

    final_time: float = base_config["final_time"]
    time_step_delta: float = base_config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    np.random.seed(base_config["seed"])

    target_specs = build_entity_specs(base_config, section_key="targets")
    agent_specs  = build_entity_specs(base_config, section_key="agents")

    # Create targets
    targets: list[Target] = [Target(initial_position=pos, time_steps=time_steps, config=conf) for (pos, conf) in target_specs]

    # Create agents
    agents: list[Agent] = [Agent(initial_position=pos, time_steps=time_steps, config=conf, targets=[], pin_row=None) for (pos, conf) in agent_specs]

    # Attach targets and pinning
    pinning_matrix: NDArray[np.float64] = np.array(base_config["pinning_matrix"], dtype=float)
    for index, agent in enumerate(agents):
        agent.targets = targets
        agent.pin_row = pinning_matrix[index, :].copy()

    # Construct neighborhoods
    construct_undirected_neighborhood_set(agents,  base_config["agent_edge_set"])
    construct_undirected_neighborhood_set(targets, base_config.get("target_edge_set", []))

    # Simulation loop
    for step in range(1, time_steps):
        for agent in agents: agent.compute_control_output(step)
        for target in targets: target.compute_control_output(step)

        for agent in agents: agent.update_dynamics(step)
        for target in targets: target.update_dynamics(step)

        simulation_time: float = step * time_step_delta
        save_state_to_csv(step, simulation_time, agents, targets)

        print(f"Progress: {step / time_steps * 100:6.2f}%", end="\r", flush=True)

    print("\nSimulation completed.")
    close_all_files()

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

def run_batch_simulation_with_results() -> None:
    configs = load_configurations()
    run_simulation_from_configs(configs)
    results()

if __name__ == "__main__":
    run_batch_simulation_with_results()
