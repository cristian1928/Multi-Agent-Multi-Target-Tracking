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

    j_inertia: NDArray[np.float64] = np.diag([2.0, 1.2, 1.6])              # kg·m^2
    tau_body: NDArray[np.float64] = np.array([0.0, 0.15, 0.0])             # N·m
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

# ---------------------------------------------------------------------

def current_dynamics(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Softer 2-D current: reduced magnitudes, widened/lengthened channels,
    and speed saturation to keep trajectories clean. No z motion (z_dot = 0).
    """

    x, y, z = state
    pos = np.array([x, y], dtype=np.float64)

    # ---------------- TUNING (gentle defaults) ----------------
    FLOW_GAIN    = 0.30   # scales ALL XY flow components
    SRC_GAIN     = 0.80   # extra attenuation for sources/sinks
    VORTEX_GAIN  = 0.50   # extra attenuation for vortices
    CHAN_GAIN    = 0.45   # extra attenuation for channel push
    EPS          = 1.20   # bigger => smoother/less spiky near centers
    W_SCALE      = 2.00   # widen channels (across)
    ELL_SCALE    = 2.50   # lengthen channels (along)
    V_SOFT       = 0.70   # soft-saturation knee for XY speed
    V_MAX        = 1.00   # hard clamp for XY speed (absolute)
    # ----------------------------------------------------------

    # Background drift (softened)
    v0: NDArray[np.float64] = FLOW_GAIN * np.array([0.12, 0.03], dtype=np.float64)

    # Sources / sinks: (center, strength)
    sources: List[tuple[NDArray[np.float64], float]] = [
        (np.array([-2.0,  1.0], dtype=np.float64), +1.2),
        (np.array([ 1.5, -1.0], dtype=np.float64), -1.0),
        (np.array([ 0.0,  2.0], dtype=np.float64), +0.6),
    ]

    # Vortices / eddies: (center, circulation)
    vortices: List[tuple[NDArray[np.float64], float]] = [
        (np.array([-1.0, -0.5], dtype=np.float64), +0.8),  # CCW
        (np.array([ 2.0,  1.5], dtype=np.float64), -0.6),  # CW
    ]

    # Channels: (center c, tangent d (unit), width w, along-length scale ell, strength beta)
    channels: List[tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]] = [
        (np.array([-3.0,  0.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64), 0.4, 4.0, 0.9),
        (np.array([ 0.0, -2.0], dtype=np.float64), (1.0/np.sqrt(2.0))*np.array([1.0, 1.0], dtype=np.float64), 0.5, 3.0, 0.6),
    ]

    def rot90(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-v[1], v[0]], dtype=np.float64)

    v_xy: NDArray[np.float64] = v0.copy()

    # Sources/sinks (potential flow, softened + bigger EPS)
    for center, alpha in sources:
        r = pos - center
        denom = float(r @ r) + EPS**2
        v_xy += FLOW_GAIN * SRC_GAIN * alpha * r / denom

    # Vortices (rotational flow, softened + bigger EPS)
    for center, gamma in vortices:
        r = pos - center
        denom = float(r @ r) + EPS**2
        v_xy += FLOW_GAIN * VORTEX_GAIN * gamma * rot90(r) / denom

    # Channels (Gaussian profile, widened/lengthened + clipped exponent)
    for c, d, w, ell, beta in channels:
        d = d / np.linalg.norm(d)
        n = rot90(d)
        xi = pos - c
        s_along  = float(d @ xi)
        n_across = float(n @ xi)

        w_eff   = max(W_SCALE  * float(w),   1e-6)
        ell_eff = max(ELL_SCALE * float(ell), 1e-6)
        arg = -0.5 * ((n_across / w_eff) ** 2 + (s_along / ell_eff) ** 2)
        arg = np.clip(arg, -700.0, 0.0)  # numeric safety
        weight = np.exp(arg)

        v_xy += FLOW_GAIN * CHAN_GAIN * beta * weight * d

    # # Soft saturation (keeps shape but compresses extremes)
    # n_xy = np.linalg.norm(v_xy)
    # if n_xy > 1e-12:
    #     v_xy = (np.tanh(n_xy / V_SOFT) * (V_SOFT / n_xy)) * v_xy

    # # Hard clamp (absolute cap)
    # n_xy = np.linalg.norm(v_xy)
    # if n_xy > V_MAX:
    #     v_xy = (V_MAX / n_xy) * v_xy

    # No vertical motion for entities using this dynamics
    z_dot: float = 0.0
    return np.array([v_xy[0], v_xy[1], z_dot], dtype=np.float64)


# ---------------------------------------------------------------------

def f8_dynamics(time: float) -> NDArray[np.float64]:
    """
    Figure-8 trajectory dynamics with vertical z-axis oscillation.

    Returns
        [x_dot, y_dot, z_dot] : instantaneous velocity along a figure-8 in x-y
                                and sinusoidal oscillation in z.
    """

    # In-plane figure-8 (Lissajous-style)
    A: float     = 2.0         # Amplitude in x-direction
    B: float     = 3.0           # Amplitude in y-direction
    a: float     = 1.0           # Frequency in x-direction
    b: float     = 2.0           # Frequency in y-direction
    delta: float = np.pi / 2.0   # Phase shift (x)

    # Vertical oscillation parameters
    C: float       = 3        # Amplitude in z-direction
    c: float       = 5         # Frequency in z-direction
    delta_z: float = 0         # Phase shift (z)

    xdot: float = A * a * np.cos(a * time + delta)
    ydot: float = B * b * np.cos(b * time)
    zdot: float = C * c * np.cos(c * time + delta_z)

    return np.array([xdot, ydot, zdot], dtype=np.float64)

# ---------------------------------------------------------------------

def none(time: float) -> NDArray[np.float64]:
    return np.zeros(3, dtype=np.float64)

# ---------------------------------------------------------------------

def get_dynamics_function(dynamics_type: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return the dynamics function associated with `dynamics_type`."""
    dynamics_map: Dict[str, Callable[..., NDArray[np.float64]]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "custom": custom,
        "current": current_dynamics,     # new current flow field
        "f8_dynamics": f8_dynamics,      # figure-8 with z oscillation
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
        "current": [0.0, 0.0, 0.0],               # position (x,y,z); z unused by flow
        "f8_dynamics": [0.0, 0.0, 0.0],           # position placeholder if needed
        "none": [0.0, 0.0, 0.0],
    }
    return initial_conditions_map[dynamics_type]
