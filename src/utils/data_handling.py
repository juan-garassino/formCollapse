import numpy as np
import os

def save_data(results, output_dir):
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for name, data in results.items():
        np.savetxt(os.path.join(data_dir, f"{name}.csv"), data, delimiter=",")
    print(f"Saved raw data in {data_dir}")