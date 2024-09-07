import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def preprocess_input(dim=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(dim)

def save_png(fig, filename, output_dir):
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    full_path = os.path.join(png_dir, f"{filename}.png")
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot as {full_path}")

def plot_attractors(results, output_dir):
    for name, data in results.items():
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
        ax.set_title(f'{name} Attractor (3D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save as PNG
        save_png(fig, f"{name}_attractor_3d", output_dir)
        plt.close(fig)
        
        # 2D projections
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.plot(data[:, 0], data[:, 1], lw=0.5)
        ax1.set_title(f'{name} (XY Projection)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        ax2.plot(data[:, 0], data[:, 2], lw=0.5)
        ax2.set_title(f'{name} (XZ Projection)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        
        ax3.plot(data[:, 1], data[:, 2], lw=0.5)
        ax3.set_title(f'{name} (YZ Projection)')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        
        plt.tight_layout()
        
        # Save as PNG
        save_png(fig, f"{name}_attractor_2d_projections", output_dir)
        
        plt.close(fig)

import matplotlib.pyplot as plt
import math

def create_summary_plot(results, output_dir):
    num_attractors = len(results)
    rows = math.ceil(math.sqrt(num_attractors))
    cols = math.ceil(num_attractors / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), subplot_kw={'projection': '3d'})
    
    if num_attractors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_png(fig, "summary_plot", output_dir)
    plt.close(fig)

def save_data(results, output_dir):
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for name, data in results.items():
        np.savetxt(os.path.join(data_dir, f"{name}.csv"), data, delimiter=",")
    print(f"Saved raw data in {data_dir}")