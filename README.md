# Multi-Agent, Multi-Target Consensus Simulator

A Python simulator for nonlinear dynamical systems with multiple agents and multiple targets. Agents use a proportional consensus-and-pinning controller. Targets currently have a zero controller. Both agents and targets evolve under **ẋ = f(x) + u** where `f(x)` is the chosen dynamics model and `u` is the control input.

## Quickstart

### Prerequisites
- Python 3.11 or higher
- Dependencies in `requirements.txt` (and `requirements-dev.txt` if you want dev tooling)

### Install
```bash
git clone https://github.com/cristian1928/Multi-Agent-Multi-Target-Tracking.git
cd Multi-Agent-Multi-Target-Tracking

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Run the simulation
```bash
python3 main.py
```
This will:
- Load `configurations/config_common.json` and merge any other JSONs in `configurations/`
- Build targets and agents from their arrays (IDs come from the config)
- Run dynamics with **ẋ = f(x) + u** for both agents and targets
- Use a **proportional** agent controller (consensus + multi-target pinning)
- Use a **zero** controller for targets (controller stub is present)
- Save data under:
  - `simulation_data/agent_data/<AgentID>_state_data.csv`
  - `simulation_data/target_data/<TargetID>_state_data.csv`

### Visualize results
```bash
python3 src/visualization/plotter.py
```
- Plots agent error norms and 3D trajectories
- Overlays all targets with their names from the config

## Overview

### What it does
- Simulates nonlinear systems for multiple agents and multiple targets
- Uses per-entity dynamics and per-entity state dimension (`num_states`)
- Supports undirected communication graphs for agents and for targets
- Supports a many-to-many pinning map between agents and targets

### Core pieces
- **Entity**: shared base with positions, velocities, time step, intrinsic dynamics, control vectors, and `ẋ = f(x) + u` integration
- **Agent**: proportional controller  
  \( u_i = k_{\text{agents}} \Big(\sum_{j\in\mathcal N_i}(x_j-x_i) + \sum_{m} p_{i,m}\,(x^{(m)}_{\text{target}}-x_i)\Big) \)
- **Target**: controller set to zero; integrates the same way as agents
- **Graphs**: undirected, 1-based indices
- **Data**: buffered CSV logging; separate folders for agents and targets
- **Plots**: IEEE-style formatting

### Built-in dynamics
- Attitude MRP
- Chua circuit
- Trophic dynamics
- Custom (zero derivative placeholder)

## Configuration

All simulation settings live in JSON files under `configurations/`. The baseline is `config_common.json`. You can add additional JSONs that override parts of the baseline (merged at load).

### Schema (summary)
```json
{
  "final_time": 120,
  "time_step_delta": 0.01,
  "seed": 0,

  "agents": [
    { "id": "A1", "dynamics_type": "trophic_dynamics", "initial_position": [-8.0, 3.0, 1.5], "num_states": 3 }
  ],

  "targets": [
    { "id": "T1", "dynamics_type": "trophic_dynamics", "initial_position": [40.0, 9.0, 2.0], "num_states": 3 }
  ],

  "agent_edge_set": [[1,2],[2,3]],
  "target_edge_set": [[1,2],[2,1]],

  "pinning_matrix": [
    [0,1]
  ],

  "agents_proportional_gain": 1.0,
  "targets_proportional_gain": 1.0
}
```

### Notes
- **IDs**: Used for filenames and plots. Keep them unique.
- **Dimensions**: `num_states` is per-entity. Interacting pairs must match dimensions:
  - Agent–Agent neighbors must share `num_states`
  - Agent–Target pairs with nonzero pin entries must share `num_states`
- **Pinning matrix**: Must be `len(agents) × len(targets)`.
- **Graphs**: `agent_edge_set` and `target_edge_set` are undirected; each `[i,j]` adds both directions.

## Data outputs

- `simulation_data/agent_data/<AgentID>_state_data.csv`  
  Columns: `Time, Position X, Position Y, Position Z, Synchronization Error Norm`
- `simulation_data/target_data/<TargetID>_state_data.csv`  
  Columns: `Time, Position X, Position Y, Position Z`

## Code quality

Type check:
```bash
mypy --strict
```

Tests:
```bash
python -m pytest -v tests/
```

## License
AGPL-3.0. See [LICENSE](LICENSE).

## Contact
- **Cristian Nino**
- **Email:** cristian1928@ufl.edu
- **GitHub:** @cristian1928
