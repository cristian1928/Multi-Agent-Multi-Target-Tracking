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

#---------------------------------------------------------------------

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

#---------------------------------------------------------------------

def custom(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Placeholder user-defined dynamics.
    Returns a zero derivative of the same shape.
    """
    return np.zeros_like(state, dtype=np.float64)

# ---------------------------------------------------------------------

def current_dynamics_agent(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Steady 2-D 'current' field with sources/sinks, vortices (eddies),
    and Gaussian-profile channels.

    Parameters
    ----------
    state : np.ndarray, shape (>=2,)
        Interpreted as position [x, y, (z ...)], z ignored.

    Returns
    -------
    np.ndarray, shape (3,)
        Velocity [x_dot, y_dot, z_dot]; z_dot = 0.
    """
    x: float = float(state[0])
    y: float = float(state[1])

    # Small background drift (uniform flow)
    vx: float = 0.20
    vy: float = 0.05

    # Smooth singularities to keep the ODE well-behaved
    eps: float = 0.15

    # ---------------- Sources / Sinks ----------------
    # rows = [px, py, alpha]; alpha>0 source, alpha<0 sink
    SS = np.array([
        [-2.2,  1.1, +1.2],
        [ 1.6, -1.1, -1.0],
        [ 0.2,  2.0, +0.6],
    ], dtype=np.float64)

    if SS.size:
        dx = x - SS[:, 0]
        dy = y - SS[:, 1]
        r2 = dx * dx + dy * dy + eps * eps
        vx += np.sum(SS[:, 2] * dx / r2)
        vy += np.sum(SS[:, 2] * dy / r2)

    # ---------------- Vortices / Eddies ----------------
    # rows = [qx, qy, gamma]; gamma>0 CCW, gamma<0 CW
    VV = np.array([
        [-1.1, -0.6, +0.8],
        [ 2.1,  1.4, -0.6],
    ], dtype=np.float64)

    if VV.size:
        dx = x - VV[:, 0]
        dy = y - VV[:, 1]
        r2 = dx * dx + dy * dy + eps * eps
        # rotate (dx, dy) by +90° => (-dy, dx)
        vx += np.sum(VV[:, 2] * (-dy) / r2)
        vy += np.sum(VV[:, 2] * ( dx) / r2)

    # ---------------- Channels (Gaussian strips) ----------------
    # Each channel k: flow ~ beta * exp(- (n·r)^2 / (2 w^2)) * exp(- (d·r)^2 / (2 ell^2)) * d
    def add_channel(center: tuple[float, float],
                    d: NDArray[np.float64],
                    n: NDArray[np.float64],
                    beta: float,
                    width: float,
                    ell: float) -> None:
        nonlocal vx, vy
        rx = x - center[0]
        ry = y - center[1]
        nproj = rx * float(n[0]) + ry * float(n[1])
        tproj = rx * float(d[0]) + ry * float(d[1])
        gN = np.exp(-(nproj * nproj) / (2.0 * width * width))
        gT = np.exp(-(tproj * tproj) / (2.0 * ell * ell))
        s = beta * gN * gT
        vx += s * float(d[0])
        vy += s * float(d[1])

    # Unit tangents and normals
    d1 = np.array([1.0, 0.0])                      # horizontal channel
    n1 = np.array([0.0, 1.0])

    invsqrt2 = 1.0 / np.sqrt(2.0)
    d2 = np.array([invsqrt2, invsqrt2])            # NE-tilted channel
    n2 = np.array([-d2[1], d2[0]])

    # Channel 1: left→right river across y≈0
    add_channel(center=(-3.0, 0.0), d=d1, n=n1, beta=0.9, width=0.45, ell=4.2)

    # Channel 2: NE chute near y≈-2
    add_channel(center=(0.0, -2.1), d=d2, n=n2, beta=0.6, width=0.50, ell=3.4)

    return np.array([vx, vy, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------

def current_dynamics_target(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Steady 3-D 'current' field with sources/sinks, swirlers (axis-aligned eddies),
    and Gaussian-profile flow tubes (channels).

    Parameters
    ----------
    state : np.ndarray, shape (>=3,)
        Interpreted as position [x, y, z].

    Returns
    -------
    np.ndarray, shape (3,)
        Velocity [x_dot, y_dot, z_dot].
    """
    # Position (allow shorter vectors defensively)
    x: float = float(state[0]) if state.shape[0] > 0 else 0.0
    y: float = float(state[1]) if state.shape[0] > 1 else 0.0
    z: float = float(state[2]) if state.shape[0] > 2 else 0.0

    # Background drift (uniform flow)
    v0 = np.array([0.20, 0.05, 0.02], dtype=np.float64)
    vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])

    # Small smoothing to prevent singularities
    eps: float = 0.15

    # ---------------- Sources / Sinks (3D) ----------------SS
    # columns: [px, py, pz, alpha]; alpha>0 = source, alpha<0 = sink
    
    SS = np.array([
        [-2.0,  1.0,  0.8, +1.2],
        [ 1.6, -1.2, -0.5, -1.0],
        [ 0.2,  2.0,  1.6, +0.6],
    ], dtype=np.float64)

    if SS.size:
        R = np.stack([x - SS[:, 0], y - SS[:, 1], z - SS[:, 2]], axis=1)        # (N,3)
        r2 = np.sum(R * R, axis=1) + eps * eps                                  # (N,)
        invr3 = 1.0 / np.power(r2, 1.5)                                          # ~ 1/r^3
        contrib = (SS[:, 3] * invr3)[:, None] * R                                # (N,3)
        v_src = contrib.sum(axis=0)
        vx += float(v_src[0]); vy += float(v_src[1]); vz += float(v_src[2])

    # ---------------- Swirlers / Eddies (3D) ----------------
    # columns: [qx, qy, qz, ax, ay, az, Gamma]
    # Flow ~ (Gamma * (a × r)) / (|r|^2 + eps^2)
    VV = np.array([
        [-1.2, -0.6,  0.0,   0.0, 0.0, 1.0,  +0.9],   # swirl around +Z near (-1.2,-0.6,0)
        [ 2.0,  1.3,  0.7,   0.0, 1.0, 0.0,  -0.7],   # swirl around +Y near (2.0,1.3,0.7)
    ], dtype=np.float64)

    if VV.size:
        Q = VV[:, 0:3]                             # centers (N,3)
        A = VV[:, 3:6]                             # axis vectors (N,3)
        # normalize axes
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        R = np.stack([x - Q[:, 0], y - Q[:, 1], z - Q[:, 2]], axis=1)            # (N,3)
        r2 = np.sum(R * R, axis=1) + eps * eps
        cross = np.cross(An, R)                                                    # (N,3)
        # 1/r^2 decay (softer than sources); you can switch to r^(3/2) if you prefer
        invr2 = 1.0 / r2
        contrib = (VV[:, 6] * invr2)[:, None] * cross
        v_vor = contrib.sum(axis=0)
        vx += float(v_vor[0]); vy += float(v_vor[1]); vz += float(v_vor[2])

    # ---------------- Channels (Gaussian tubes) ----------------
    # Each tube k: center ck, direction dk (unit), flow ~ beta * exp(-||n||^2/(2 w^2)) * exp(-(t^2)/(2 ℓ^2)) * dk
    def add_channel(center: tuple[float, float, float],
                    d: NDArray[np.float64],
                    beta: float,
                    width: float,
                    ell: float) -> None:
        nonlocal vx, vy, vz
        cx, cy, cz = center
        r = np.array([x - cx, y - cy, z - cz], dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)

        t = float(np.dot(r, d))              # axial projection
        n_vec = r - t * d                    # normal component to the tube axis
        n2 = float(np.dot(n_vec, n_vec))

        gN = float(np.exp(-n2 / (2.0 * width * width)))
        gT = float(np.exp(-(t * t) / (2.0 * ell * ell)))
        s = beta * gN * gT
        vx += s * float(d[0]); vy += s * float(d[1]); vz += s * float(d[2])

    # Define a couple of tubes
    add_channel(center=(-3.0, 0.0, 0.0), d=np.array([1.0, 0.0, 0.1]), beta=0.9, width=0.8, ell=5)   # slightly rising x-directed river
    add_channel(center=( 0.0,-2.0, 1.0), d=np.array([0.7, 0.7, 0.0]), beta=0.6, width=0.55, ell=3.2)  # NE tube in x–y

    return np.array([vx, vy, vz], dtype=np.float64)

# ---------------------------------------------------------------------

def f8_dynamics(time: float) -> NDArray[np.float64]:
    """
    Figure-8 trajectory dynamics with vertical z-axis oscillation.

    Returns
        [x_dot, y_dot, z_dot] : instantaneous velocity along a figure-8 in x-y
                                and sinusoidal oscillation in z.
    """

    # In-plane figure-8 (Lissajous-style)
    A: float     = 25      # Amplitude in x-direction
    B: float     = 15      # Amplitude in y-direction
    a: float     = 1.0           # Frequency in x-direction
    b: float     = 2.0           # Frequency in y-direction
    delta: float = np.pi / 2.0   # Phase shift (x)

    # Vertical oscillation parameters
    C: float       = 15# Amplitude in z-direction
    c: float       = 1         # Frequency in z-direction
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
        "current_agent": current_dynamics_agent,     # new current flow field
        "current_target": current_dynamics_target,     # new current flow field
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
        "current_agent": [0.0, 0.0, 0.0],               # position (x,y,z); z unused by flow
        "current_target": [0.0, 0.0, 0.0],
        "f8_dynamics": [0.0, 0.0, 0.0],           # position placeholder if needed
        "none": [0.0, 0.0, 0.0],
    }
    # print initial_conditions_map of current_agent and current_target
    print("Initial conditions for current_agent:", initial_conditions_map["current_agent"])
    print("Initial conditions for current_target:", initial_conditions_map["current_target"])
    return initial_conditions_map[dynamics_type]
