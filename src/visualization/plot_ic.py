# import data from runs/flow_simulation_data/ic_xy_data/A1_state_data_i.csv files and plot

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path('runs/flow_simulation_data/ic_xy_data/')
data_dir.mkdir(parents=True, exist_ok=True)

n = 6

plt.xlabel('X Position')
plt.ylabel('Y Position')

for i in range(1, n + 1):
    # print(i)
    p = data_dir / f'A1_state_data_{i}.csv'
    A1_state_data = pd.read_csv(p)
    A1_x = A1_state_data["Position X"]
    A1_y = A1_state_data["Position Y"]
    # print(A1_x)
    # print(A1_y)

    # ensure all traces go to the same axes (keep spatial scaling)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plot x and y data from A1_state_data files to the same figure
    plt.plot(A1_x, A1_y, label=f'A1 State {i}')

# Plot arbitrary initial conditions for most disconnected agent, showing convergence to desired position
    # for x, y in zip(A1_x, A1_y):
    #     plt.scatter(x, y)


plt.show()



