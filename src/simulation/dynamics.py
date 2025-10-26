from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------

def attitude_mrp(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rigid-body attitude kinematics in Modified Rodrigues Parameters.

    State vector
        r : np.ndarray, shape (3,)  -- Modified Rodrigues Parameters, unitless

    Returns
        r_dot : np.ndarray, shape (3,)  -- time derivative of r, 1/s
    """

    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the 3x3 skew-symmetric matrix of vector v (rad/s)."""
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    r: NDArray[np.float64] = state
    r_sq: float = float(np.dot(r, r))
    b_mat: NDArray[np.float64] = (1.0 - r_sq) * np.eye(3) + 2.0 * _skew(r) + 2.0 * np.outer(r, r)

    j_inertia: NDArray[np.float64] = np.diag([2.0, 1.2, 1.6])             # kg·m^2
    tau_body: NDArray[np.float64] = np.array([0.0, 0.15, 0.0])            # N·m
    omega_body: NDArray[np.float64] = np.linalg.inv(j_inertia) @ tau_body  # rad/s

    r_dot: NDArray[np.float64] = 0.5 * b_mat @ omega_body
    return r_dot

# ---------------------------------------------------------------------

def chua(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Dimensionless Chua double-scroll circuit.

    State vector
        x : capacitor voltage proxy, unitless
        y : capacitor voltage proxy, unitless
        z : inductor current proxy, unitless
    """

    x, y, z = state
    alpha: float = 15.6          # unitless
    beta: float = 28.0           # unitless
    m0: float = -1.143           # unitless
    m1: float = -0.714           # unitless

    g: float = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))

    x_dot: float = alpha * (y - x - g)
    y_dot: float = x - y + z
    z_dot: float = -beta * y
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)

# ---------------------------------------------------------------------

def trophic_dynamics(state: NDArray[np.float64]) -> NDArray[np.float64]:

    """
    Three-tier ecological food chain.

    State vector
        H : prey (herbivore) population, individuals
        P : predator population, individuals
        T : top-predator population, individuals
    """

    h_pop, p_pop, t_pop = state
    r_h: float = 0.6               # 1/day
    k_cap: float = 100.0           # individuals
    a_hp: float = 0.02             # 1/(individual·day)
    a_pt: float = 0.01             # 1/(individual·day)
    d_p: float = 0.3               # 1/day
    d_t: float = 0.1               # 1/day

    h_dot: float = r_h * h_pop * (1.0 - h_pop / k_cap) - a_hp * h_pop * p_pop
    p_dot: float = -d_p * p_pop + a_hp * h_pop * p_pop - a_pt * p_pop * t_pop
    t_dot: float = -d_t * t_pop + a_pt * p_pop * t_pop
    return np.array([h_dot, p_dot, t_dot], dtype=np.float64)

# ---------------------------------------------------------------------

def custom(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Placeholder user-defined dynamics.
    Returns a zero derivative of the same shape.
    """
    return np.zeros_like(state, dtype=np.float64)


######## def figure 8 ##########

def f8_dynamics(time: float) -> NDArray[np.float64]:

    '''Figure-8 trajectory dynamics'''

    A: float     = 10     # Amplitude in x-direction
    B: float     = 5        # Amplitude in y-direction
    a: float     = 1      # Frequency in x-direction
    b: float     = 2        # Frequency in y-direction
    delta: float = np.pi/2  # Phase shift

    xdot: float = A*a*np.cos(a*time + delta)
    ydot: float = B*b*np.cos(b*time)
    zdot: float = 0 # t = step*delta(t), a cos(wt)
    return np.array([xdot, ydot, zdot], dtype=np.float64)

# ---------------------------------------------------------------------

def none(time: float) -> NDArray[np.float64]:
    return np.zeros(3, dtype=np.float64)

#---------------------------------------------------------------------

def get_dynamics_function(dynamics_type: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return the dynamics function associated with `dynamics_type`."""
    dynamics_map: Dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "custom": custom,
        "none": none,
    }
    return dynamics_map[dynamics_type]

# ---------------------------------------------------------------------
def get_initial_conditions(dynamics_type: str) -> List[float]:
    """Return a list of reasonable initial conditions for the chosen model."""
    initial_conditions_map: Dict[str, List[float]] = {
        "attitude_mrp": [0.25, 0.10, -0.30],      # unitless
        "chua": [0.2, 0.0, 0.0],                  # unitless
        "trophic_dynamics": [40.0, 9.0, 2.0],     # individuals
        "custom": [0.0, 0.0, 0.0],
        "none": [0.0, 0.0, 0.0],
    }
    return initial_conditions_map[dynamics_type]


