import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter, welch
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

logger = logging.getLogger(__name__)

# Set up Seaborn style
sns.set_theme(style="white")
plt.rcParams['axes.edgecolor'] = '0.2'
plt.rcParams['axes.linewidth'] = 0.5

def save_png(fig: plt.Figure, filename: str, output_dir: str) -> None:
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    full_path = os.path.join(png_dir, f"{filename}.png")
    try:
        fig.savefig(full_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved plot as {full_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {str(e)}")

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

def remove_top_right_axes(ax: plt.Axes) -> None:
    """Remove the top and right axes from a given Axes object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_attractor(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot a single attractor in 3D and 2D projections."""
    logger.info(f"Plotting attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
    except Exception as e:
        logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
        smooth_data_points = data

    try:
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
        ax.set_title(f'{name} Attractor (3D)', fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.grid(False)
        save_png(fig, f"{name}_attractor_3d", output_dir)
        plt.close(fig)

        # 2D projections
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], lw=0.5)
        ax1.set_title(f'{name} (XY Projection)', fontsize=14)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        remove_top_right_axes(ax1)
        
        ax2.plot(smooth_data_points[:, 0], smooth_data_points[:, 2], lw=0.5)
        ax2.set_title(f'{name} (XZ Projection)', fontsize=14)
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Z', fontsize=12)
        remove_top_right_axes(ax2)
        
        ax3.plot(smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
        ax3.set_title(f'{name} (YZ Projection)', fontsize=14)
        ax3.set_xlabel('Y', fontsize=12)
        ax3.set_ylabel('Z', fontsize=12)
        remove_top_right_axes(ax3)
        
        plt.tight_layout()
        save_png(fig, f"{name}_attractor_2d_projections", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot attractor {name}: {str(e)}")

def create_summary_plot(results: Dict[str, np.ndarray], output_dir: str, smooth: bool = False) -> None:
    """Create a summary plot of all attractors."""
    logger.info("Creating summary plot")
    num_attractors = len(results)
    
    if num_attractors == 0:
        logger.warning("No successful simulations to plot. Skipping summary plot.")
        return
    
    rows = int(np.ceil(np.sqrt(num_attractors)))
    cols = int(np.ceil(num_attractors / rows))

    try:
        fig = plt.figure(figsize=(5*cols, 5*rows))
        
        for i, (name, data) in enumerate(results.items()):
            try:
                smooth_data_points = smooth_data(data, smooth)
            except Exception as e:
                logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
                smooth_data_points = data
            
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            ax.plot(smooth_data_points[:, 0], smooth_data_points[:, 1], smooth_data_points[:, 2], lw=0.5)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(False)

        plt.tight_layout()
        save_png(fig, "summary_plot", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create summary plot: {str(e)}")

def plot_phase_space(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot the phase space of an attractor."""
    logger.info(f"Plotting phase space for attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
    except Exception as e:
        logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
        smooth_data_points = data

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(x=smooth_data_points[:, 0], y=smooth_data_points[:, 1], cmap="YlGnBu", fill=True, cbar=True)
        ax.set_title(f'{name} Attractor Phase Space', fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        remove_top_right_axes(ax)
        
        plt.tight_layout()
        save_png(fig, f"{name}_phase_space", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot phase space for attractor {name}: {str(e)}")

def plot_time_series(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot the time series of an attractor."""
    logger.info(f"Plotting time series for attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
    except Exception as e:
        logger.error(f"Smoothing failed for {name}: {str(e)}. Using original data.")
        smooth_data_points = data

    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        time = np.arange(len(smooth_data_points))
        
        ax1.plot(time, smooth_data_points[:, 0], lw=1)
        ax1.set_ylabel('X', fontsize=12)
        remove_top_right_axes(ax1)
        
        ax2.plot(time, smooth_data_points[:, 1], lw=1)
        ax2.set_ylabel('Y', fontsize=12)
        remove_top_right_axes(ax2)
        
        ax3.plot(time, smooth_data_points[:, 2], lw=1)
        ax3.set_ylabel('Z', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        remove_top_right_axes(ax3)
        
        fig.suptitle(f'{name} Attractor Time Series', fontsize=16)
        
        plt.tight_layout()
        save_png(fig, f"{name}_time_series", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot time series for attractor {name}: {str(e)}")

def animate_3d(name: str, data: np.ndarray, output_dir: str) -> None:
    """Create a 3D animation of the attractor with proper axis limits."""
    logger.info(f"Creating 3D animation for attractor: {name}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the limits for each dimension
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    z_min, z_max = data[:, 2].min(), data[:, 2].max()

    # Add some padding to the limits (e.g., 10% of the range)
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

    line, = ax.plot([], [], [], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{name} Attractor 3D Animation')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def animate(i):
        line.set_data(data[:i, 0], data[:i, 1])
        line.set_3d_properties(data[:i, 2])
        return line,

    # Adjust the number of frames to reduce file size and rendering time
    num_frames = min(len(data), 500)  # Limit to 500 frames
    frame_interval = len(data) // num_frames
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                         frames=range(0, len(data), frame_interval), 
                         interval=20, blit=True)

    animation_dir = os.path.join(output_dir, 'animations')
    os.makedirs(animation_dir, exist_ok=True)
    
    # Save with a lower dpi and fps to reduce file size
    anim.save(os.path.join(animation_dir, f'{name}_3d_animation.mp4'), 
              writer='ffmpeg', fps=30, dpi=150)
    
    plt.close(fig)
    logger.info(f"3D animation for {name} saved successfully")

def plot_poincare_section(name: str, data: np.ndarray, output_dir: str, plane: str = 'xy', threshold: float = 0) -> None:
    """Plot Poincaré section of the attractor."""
    logger.info(f"Plotting Poincaré section for attractor: {name}")

    if plane == 'xy':
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        crossing_indices = np.where(np.diff(np.sign(z - threshold)))[0]
    elif plane == 'yz':
        x, y, z = data[:, 1], data[:, 2], data[:, 0]
        crossing_indices = np.where(np.diff(np.sign(z - threshold)))[0]
    elif plane == 'xz':
        x, y, z = data[:, 0], data[:, 2], data[:, 1]
        crossing_indices = np.where(np.diff(np.sign(z - threshold)))[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x[crossing_indices], y[crossing_indices], s=1, alpha=0.5)
    ax.set_title(f'{name} Attractor Poincaré Section ({plane.upper()} plane)', fontsize=16)
    ax.set_xlabel(plane[0].upper(), fontsize=12)
    ax.set_ylabel(plane[1].upper(), fontsize=12)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_poincare_section_{plane}", output_dir)
    plt.close(fig)

def plot_bifurcation(name: str, system_func: callable, param_range: np.ndarray, param_name: str, output_dir: str) -> None:
    """Plot bifurcation diagram for a given parameter."""
    logger.info(f"Plotting bifurcation diagram for attractor: {name}")

    results = []
    for param in param_range:
        params = {param_name: param}
        trajectory = system_func(np.random.rand(3), np.linspace(0, 100, 1000), params)
        results.extend([(param, x) for x in trajectory[500:, 0]])  # Use x-coordinate and discard transients

    results = np.array(results)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(results[:, 0], results[:, 1], ',k', alpha=0.1, markersize=0.1)
    ax.set_title(f'{name} Attractor Bifurcation Diagram', fontsize=16)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_bifurcation_diagram", output_dir)
    plt.close(fig)

def plot_lyapunov_exponent(name: str, lyap_exp: np.ndarray, output_dir: str) -> None:
    """Plot Lyapunov exponent spectrum."""
    logger.info(f"Plotting Lyapunov exponent spectrum for attractor: {name}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(lyap_exp) + 1), lyap_exp, 'o-')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f'{name} Attractor Lyapunov Exponent Spectrum', fontsize=16)
    ax.set_xlabel('Exponent Index', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_lyapunov_spectrum", output_dir)
    plt.close(fig)

def plot_power_spectrum(name: str, data: np.ndarray, output_dir: str) -> None:
    """Plot power spectrum of the attractor."""
    logger.info(f"Plotting power spectrum for attractor: {name}")

    # Compute power spectrum for each dimension
    f, Pxx_den = welch(data, fs=1, nperseg=1024)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(f, Pxx_den[:, 0], label='X')
    ax.semilogy(f, Pxx_den[:, 1], label='Y')
    ax.semilogy(f, Pxx_den[:, 2], label='Z')
    ax.set_title(f'{name} Attractor Power Spectrum', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.legend()
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_power_spectrum", output_dir)
    plt.close(fig)