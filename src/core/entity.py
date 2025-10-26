from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..simulation import dynamics
from ..simulation.integrate import integrate_step


class Entity:
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        self.id: str = config['id']
        self.num_states: int = config['num_states']
        self.time_step_delta: float = config['time_step_delta']
        self.positions: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.velocities: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position
        dynamics_type = config['dynamics_type']
        self.dynamics_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = dynamics.get_dynamics_function(dynamics_type)
        self.control_output: NDArray[np.float64] = np.zeros(self.num_states)
        self.synchronization_error: NDArray[np.float64] = np.zeros(self.num_states)
        self.neighbors: List["Entity"] = []
        self.offsets: NDArray[np.float64] = np.zeros(self.num_states)  # Add offsets to base class

    def update_dynamics(self, step: int) -> None:
        def dynamics_with_control(t: float, pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.dynamics_function(pos) + self.control_output
        self.velocities[:, step] = self.dynamics_function(self.positions[:, step - 1]) + self.control_output
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, dynamics_with_control)
        self.positions[:, step] = result


class Agent(Entity):
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any], targets: List["Target"], pin_row: Optional[NDArray[np.float64]] = None, offset: Optional[NDArray[np.float64]] = None) -> None:
        super().__init__(initial_position, time_steps, config)
        self.targets: List["Target"] = targets
        self.pin_row: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.k1: float = config['agents_proportional_gain']
        self.offsets: NDArray[np.float64] = offset if offset is not None else np.zeros(self.num_states)

    def compute_control_output(self, step: int) -> None:
        position = self.positions[:, step - 1]
    
    # Formation-aware neighborhood consensus: ∑[(q_j + δ_j) - (q_i + δ_i)]
        neighborhood_consensus_term = np.zeros(self.num_states)
        for neighbor in self.neighbors:
            if isinstance(neighbor, Agent):
                neighbor_total = neighbor.positions[:, step - 1] + neighbor.offsets
                self_total = position + self.offsets
                neighborhood_consensus_term += neighbor_total - self_total

    # MOVING TARGET TRACKING: Drive (q_i + δ_i) → q_0(t)
        target_consensus_term = np.zeros(self.num_states)
        if len(self.targets) and self.pin_row.size:
            for index, weight in enumerate(self.pin_row):
                if weight != 0.0:
                # Current target position q_0(t)
                    current_target_pos = self.targets[index].positions[:, step - 1]
                # Drive (position + offset) → current_target_pos
                    self_total = position + self.offsets
                    target_consensus_term += weight * (current_target_pos - self_total)

        self.synchronization_error = target_consensus_term + neighborhood_consensus_term
        self.control_output = self.k1 * self.synchronization_error


class Target(Entity):
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        self.k1: float = config['targets_proportional_gain']
        self.is_centroid: bool = bool(
            config.get('is_centroid', False) or config.get('tracking_type') == 'f8_dynamics'
        )

    def compute_control_output(self, step: int) -> None:
        time = step * self.time_step_delta

        # Only the centroid gets a desired velocity
        desired_velocity = np.zeros(self.num_states)
        if self.is_centroid and hasattr(dynamics, 'f8_dynamics'):
            desired_velocity = dynamics.f8_dynamics(time)  


        neighborhood_consensus_term = np.zeros(self.num_states)
        for neighbor in self.neighbors:
            if isinstance(neighbor, Target):
                neighbor_total = neighbor.positions[:, step - 1] + neighbor.offsets
                self_total     = self.positions[:, step - 1] + self.offsets
                neighborhood_consensus_term += neighbor_total - self_total

        self.synchronization_error = neighborhood_consensus_term
        self.control_output = self.k1 * self.synchronization_error + desired_velocity

