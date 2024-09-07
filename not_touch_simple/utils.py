import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import svgwrite
import os

def preprocess_input(dim=3):
    np.random.seed(42)
    return np.random.randn(dim)

def save_png(fig, filename):
    fig.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    print(f"Saved plot as {filename}.png")

def save_svg(points, filename):
    dwg = svgwrite.Drawing(f"{filename}.svg", profile='tiny', size=('100%', '100%'))
    dwg.viewbox(0, 0, 1000, 1000)
    
    # Normalize points to fit the SVG viewbox
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * 1000
    
    # Draw the polyline in SVG
    dwg.add(dwg.polyline(points, fill='none', stroke='black', stroke_width=1))
    dwg.save()
    
    print(f"Saved plot as {filename}.svg")

def plot_attractors(results):
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
        save_png(fig, f"{name}_attractor_3d")
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
        save_png(fig, f"{name}_attractor_2d_projections")
        
        # Save as SVG (using XY projection)
        save_svg(data[:, :2], f"{name}_attractor_xy_projection")
        
        plt.close(fig)

def create_summary_plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    save_png(fig, "summary_plot")
    plt.close(fig)

def save_data(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, data in results.items():
        np.savetxt(os.path.join(output_dir, f"{name}.csv"), data, delimiter=",")
    print(f"Saved raw data in {output_dir}")