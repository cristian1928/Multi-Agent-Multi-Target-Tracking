# import data from runs/flow_simulation_data/ic_xy_data/A1_state_data_i.csv files and plot

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path('runs/flow_simulation_data/ic_xy_data')
data_dir.mkdir(parents=True, exist_ok=True)

n = 1

# desired time axis: 3.0 .. 5.0 s in 0.001 s increments
t_start = 0
t_end = 1
dt = 0.001
# include endpoint
desired_t = np.arange(t_start, t_end + dt/2, dt)

# create 3D figure/axis once
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.set_zlabel('Time (s)')
ax.grid(True)

for i in range(1, n + 1):
    print(i)
    p = data_dir / f'A1_state_data_{i}.csv'
    if not p.exists():
        print(f"Warning: {p} not found, skipping")
        continue

    A1_state_data = pd.read_csv(p)

    # Extract numeric position/time columns safely
    x_col = A1_state_data.get("Position X") if "Position X" in A1_state_data.columns else None
    y_col = A1_state_data.get("Position Y") if "Position Y" in A1_state_data.columns else None
    t_col = A1_state_data.get("Time") if "Time" in A1_state_data.columns else None

    if x_col is None or y_col is None:
        print(f"Warning: {p} missing Position X/Y columns, skipping")
        continue

    x_vals = pd.to_numeric(x_col, errors='coerce').to_numpy()
    y_vals = pd.to_numeric(y_col, errors='coerce').to_numpy()

    # build original time vector
    if t_col is not None:
        orig_t = pd.to_numeric(t_col, errors='coerce').to_numpy()
        mask = (~np.isnan(orig_t)) & (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        orig_t = orig_t[mask]
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
    else:
        # no Time column: assume data spans t_start..t_end and distribute samples evenly
        if len(x_vals) == 0:
            print(f"Warning: {p} has no data, skipping")
            continue
        orig_t = np.linspace(t_start, t_end, num=len(x_vals))
        mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        orig_t = orig_t[mask]
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

    if orig_t.size == 0 or x_vals.size == 0 or y_vals.size == 0:
        print(f"Warning: {p} has no valid numeric samples, skipping")
        continue

    # sort by time
    sort_idx = np.argsort(orig_t)
    orig_t = orig_t[sort_idx]
    x_vals = x_vals[sort_idx]
    y_vals = y_vals[sort_idx]

    # skip if original data doesn't overlap requested window at all
    t_min, t_max = orig_t[0], orig_t[-1]
    if t_max < t_start or t_min > t_end:
        print(f"Warning: {p} time range ({t_min:.3f},{t_max:.3f}) s does not overlap requested [{t_start:.3f},{t_end:.3f}] s, skipping")
        continue

    # Interpolate (or fill) onto desired_t clipped to original range (endpoints repeated)
    if orig_t.size == 1:
        # single sample: repeat value across desired_t
        x_interp = np.full_like(desired_t, x_vals[0], dtype=float)
        y_interp = np.full_like(desired_t, y_vals[0], dtype=float)
    else:
        t_for_interp = np.clip(desired_t, t_min, t_max)
        x_interp = np.interp(t_for_interp, orig_t, x_vals)
        y_interp = np.interp(t_for_interp, orig_t, y_vals)

    trace_label = f'A1 State {i}'
    color = plt.cm.tab10.colors[(i - 1) % 10]

    ax.plot3D(x_interp, y_interp, desired_t, label=trace_label, color=color)
    ax.scatter(x_interp[0], y_interp[0], desired_t[0], color=color, marker='s', s=60, zorder=5)

ax.view_init(elev=25, azim=-60)
ax.legend()
plt.tight_layout()
plt.show(block=True)



