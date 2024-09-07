import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from typing import Dict, Any
import logging
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

def save_png(fig, filename, output_dir):
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    full_path = os.path.join(png_dir, f"{filename}.png")
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved plot as {full_path}")

def smooth_data(data: np.ndarray, smooth: bool = False, resolution: int = 10000) -> np.ndarray:
    if not smooth:
        return data
    
    x, y, z = data.T
    
    try:
        # Try spline interpolation first with increased resolution
        tck, u = splprep([x, y, z], s=0, k=3)
        u_new = np.linspace(0, 1, resolution)
        new_points = splev(u_new, tck)
        return np.array(new_points).T
    except Exception as e:
        logger.warning(f"Spline interpolation failed: {str(e)}. Using Savitzky-Golay filter instead.")
        
        # Fallback to Savitzky-Golay filter with adjusted parameters
        window_length = min(len(x) // 20 * 2 + 1, 101)  # Increased window length, must be odd
        poly_order = min(3, window_length - 1)  # Must be less than window_length
        
        smoothed_x = savgol_filter(x, window_length, poly_order)
        smoothed_y = savgol_filter(y, window_length, poly_order)
        smoothed_z = savgol_filter(z, window_length, poly_order)
        
        # Interpolate the Savitzky-Golay filtered data to increase resolution
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, resolution)
        smoothed_x = np.interp(t_new, t, smoothed_x)
        smoothed_y = np.interp(t_new, t, smoothed_y)
        smoothed_z = np.interp(t_new, t, smoothed_z)
        
        return np.column_stack((smoothed_x, smoothed_y, smoothed_z))

def plot_attractor(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot a single attractor in 3D and 2D projections."""
    logger.info(f"Plotting attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
    except Exception as e:
        logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
        smooth_data_points = data

    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
    ax.set_title(f'{name} Attractor (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    save_png(fig, f"{name}_attractor_3d", output_dir)
    plt.close(fig)

    # 2D projections
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], lw=0.5)
    ax1.set_title(f'{name} (XY Projection)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.plot(smooth_data_points[:, 0], smooth_data_points[:, 2], lw=0.5)
    ax2.set_title(f'{name} (XZ Projection)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    ax3.plot(smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
    ax3.set_title(f'{name} (YZ Projection)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    
    plt.tight_layout()
    save_png(fig, f"{name}_attractor_2d_projections", output_dir)
    plt.close(fig)

def create_summary_plot(results: Dict[str, np.ndarray], output_dir: str, smooth: bool = False) -> None:
    """Create a summary plot of all attractors."""
    logger.info("Creating summary plot")
    num_attractors = len(results)
    
    if num_attractors == 0:
        logger.warning("No successful simulations to plot. Skipping summary plot.")
        return
    
    rows = int(np.ceil(np.sqrt(num_attractors)))
    cols = int(np.ceil(num_attractors / rows))

    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    for i, (name, data) in enumerate(results.items()):
        try:
            smooth_data_points = smooth_data(data, smooth)
        except Exception as e:
            logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
            smooth_data_points = data
        
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    save_png(fig, "summary_plot", output_dir)
    plt.close(fig)

# You can add more visualization functions here if needed