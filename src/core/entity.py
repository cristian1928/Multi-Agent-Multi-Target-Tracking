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

        # rows are state dimensions, columns are discrete time steps
            # self.positions[:, 0] is initial position at time 0
            # self.positions[:, 1] is position at time step 1, etc.
            # self.positions[:, step - 1] is position at prebious time step

        self.positions: NDArray[np.float64] = np.zeros((self.num_states, time_steps))

        self.velocities: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        dynamics_type = config['dynamics_type']
        self.dynamics_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = dynamics.get_dynamics_function(dynamics_type)
        self.control_output: NDArray[np.float64] = np.zeros(self.num_states)
        self.synchronization_error: NDArray[np.float64] = np.zeros(self.num_states)
        self.neighbors: List["Entity"] = []
        self.offsets: NDArray[np.float64] = np.zeros_like(initial_position)
        self.positions[:, 0] = initial_position + self.offsets

    def update_dynamics(self, step: int) -> None:
        def dynamics_with_control(t: float, pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.dynamics_function(pos) + self.control_output
        self.velocities[:, step] = self.dynamics_function(self.positions[:, step - 1]) + self.control_output
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, dynamics_with_control)
        self.positions[:, step] = result


class Agent(Entity):
    def __init__(self, initial_position: NDArray[np.float64], 
                 time_steps: int, 
                 offset, 
                 config: dict[str, Any], 
                 targets: List["Target"], 
                 pin_row: Optional[NDArray[np.float64]] = None) -> None:
        super().__init__(initial_position, time_steps, config)
        self.targets: List["Target"] = targets
        self.pin_row: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.k1: float = config['agents_proportional_gain']
        self.offsets = offset if offset is not None else np.zeros_like(initial_position)

    def compute_control_output(self, step: int) -> None:
        position = self.positions[:, step - 1] # current position state of agent i

        # neighborhood consensus term pulls agents i toward its neighbors' positions
        # affects all agents

        neighborhood_consensus_term = np.zeros(self.num_states)

        # Ni is the set of neighbors for agent i
        # for agent 2, Ni (N2) is agents 1 and 3
        # for each neighbor j in neighborhood for agent i, compute position difference between neighbor j and agent i
        # sum these vectors to get neighborhood consensus term

        for neighbor in self.neighbors:
            neighborhood_consensus_term += (neighbor.positions[:, step - 1] - position) + neighbor.offsets # computes position difference with each neighbor, i.e. the neighborhood consensus error
        #target_consensus_term = np.zeros(self.num_states)

        # self.pin.row is a row vector from pinning matrix, e.g. [0, 1, 1] for agent 2
        # each 1 means the agent is connected to that target, so for agent 2, it is connected to target 1 and target 2
        # if weight is not 0, that means this agent is connected to that target
            # for agent 2, for index = 0, weight = 0.0, so skip
            # for index = 1, weight = 1.0, so compute target 1 position - agent position
                # when youre computing target consenus term, you only know all states up to step t-1
                # you havnt yet integrated forward to compiyte positions for step t, thats purpose of target consensus term
                # "self.targets[index].positions[:, step - 1]" represents the position vector of the target at the previous time step
                # target consensus term measure show far an agent is from its pinned targets at the previous time step and in what direction it needs to move to reduce that difference
                    # pulls agents i toward the pinned target

        target_consensus_term = np.zeros(self.num_states)
        if len(self.targets) and self.pin_row.size: 
            for index, weight in enumerate(self.pin_row[:len(self.targets)]):
                if weight != 0.0: 
                    target_consensus_term += weight * (self.targets[index].positions[:, step - 1] - position) + neighbor.offsets

        # neighborhood consensus term ensures agents stay together 
        # target consensus term ensures agents follow targets
        # total error between target consensus term and neighborhood consensus term ensures agents stay in consensus with its neighbords and follow targets

        self.synchronization_error = neighborhood_consensus_term + target_consensus_term
        self.control_output = self.k1 * self.synchronization_error

class Target(Entity):
    def __init__(self, initial_position: NDArray[np.float64], 
                 time_steps: int, 
                 offset, 
                 config: dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        self.k1: float = config['targets_proportional_gain']
        self.offsets = offset if offset is not None else np.zeros_like(initial_position)

    def compute_control_output(self, step: int) -> None:
        time = step * self.time_step_delta
        desired_velocity = dynamics.f8_dynamics(time) 

        position = self.positions[:, step - 1] 
        neighborhood_consensus_term = np.zeros(self.num_states)

        for neighbor in self.neighbors:
            neighborhood_consensus_term += (neighbor.positions[:, step - 1] - position) + neighbor.offsets

        self.synchronization_error = neighborhood_consensus_term 
        self.control_output = self.k1 * self.synchronization_error + desired_velocity
