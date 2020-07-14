import os

import numpy as np

curr_latest_version = max(map(int, [name.split('_')[1] for name in os.listdir("lightning_logs")]))
eps = 1e-6
print(curr_latest_version)
with open("exp.csv", 'w') as f:
    f.truncate()
    for beta in np.linspace(0 + eps, 0.15, 5):
        curr_latest_version += 1
        os.system("python3.8 launch.py train -p SimpleGaussian -a const -b " + str(beta))
        f.write(f"version_{curr_latest_version}, beta_value: {beta}")
