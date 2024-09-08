import os
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

from src.utils.config import get_config, generate_system_params, generate_initial_condition
from src.attractors.simulators import adaptive_simulation
from src.gan.training import train_gan_on_results, generate_samples, preprocess_data
from src.utils.visualization import plot_attractor, create_summary_plot, plot_phase_space, plot_time_series, animate_3d, plot_poincare_section, plot_power_spectrum, plot_bifurcation, plot_lyapunov_exponent
from src.utils.data_handling import save_data
from src.utils.svg_gcode import save_svg, generate_gcode

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate strange attractors and create visualizations.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulations per attractor.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results.')
    parser.add_argument('--create_svg_gcode', action='store_true', help='Flag to create SVG and G-code files.')
    parser.add_argument('--create_advanced_viz', action='store_true', help='Flag to create advanced visualizations.')
    parser.add_argument('--train_gan', action='store_true', help='Flag to train GAN.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_file', type=str, help='File to save logs to.')
    return parser.parse_args()

def run_simulation(system_name: str, system_config: Dict[str, Any], output_dir: str, create_svg_gcode: bool, create_advanced_viz: bool, max_time: float) -> Tuple[Optional[str], Optional[np.ndarray]]:
    logger = logging.getLogger(f"simulation.{system_name}")
    logger.info(f"Starting simulation for {system_name}")
    
    params = generate_system_params(system_config)
    initial_condition = generate_initial_condition()
    
    success, data, message = adaptive_simulation(
        system_name,
        system_config['func'],
        params,
        initial_condition,
        system_config['sim_time'],
        system_config['sim_steps'],
        max_attempts=5,
        max_time=max_time
    )
    
    if not success:
        logger.error(f"Failed to simulate {system_name}: {message}")
        return None, None
    
    logger.info(f"Successful simulation of {system_name}: {message}")
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    key = f'{system_name}_{timestamp}'
    
    logger.info(f"Plotting attractor for {key}")
    plot_attractor(key, data, output_dir, smooth=True)
    plot_phase_space(key, data, output_dir, smooth=True)
    plot_time_series(key, data, output_dir, smooth=True)
    
    if create_advanced_viz:
        logger.info(f"Creating advanced visualizations for {key}")
        
        # 3D Animation
        animate_3d(key, data, output_dir)
        
        # Poincaré Section
        for plane in ['xy', 'yz', 'xz']:
            plot_poincare_section(key, data, output_dir, plane=plane)
        
        # Power Spectrum
        plot_power_spectrum(key, data, output_dir)
        
        # Bifurcation Diagram
        if 'bifurcation_param' in system_config:
            param_name = system_config['bifurcation_param']['name']
            param_range = np.linspace(
                system_config['bifurcation_param']['start'],
                system_config['bifurcation_param']['stop'],
                system_config['bifurcation_param']['num']
            )
            plot_bifurcation(key, SYSTEM_FUNCTIONS[system_config['func']], param_range, param_name, output_dir)
        else:
            logger.info(f"Skipping bifurcation diagram for {key}: No bifurcation parameter specified in config")
        
        # Lyapunov Exponent
        # Note: This requires a function to compute Lyapunov exponents, which is not provided in the current setup
        # If you have such a function, you can uncomment the following lines:
        # lyap_exp = compute_lyapunov_exponents(system_config['func'], params, initial_condition, system_config['sim_time'], system_config['sim_steps'])
        # plot_lyapunov_exponent(key, lyap_exp, output_dir)
        
    if create_svg_gcode:
        logger.info(f"Creating SVG and G-code for {key}")
        save_svg(data[:, :2], f"{key}_svg", output_dir)
        generate_gcode(data, f"{key}_gcode", output_dir)
    
    logger.info(f"Simulation for {key} completed")
    return key, data

def process_system(system_name: str, system_config: Dict[str, Any], args: argparse.Namespace, max_time: float) -> Dict[str, np.ndarray]:
    logger = logging.getLogger(f"process.{system_name}")
    logger.info(f"Processing system: {system_name}")
    
    if 'func' not in system_config:
        logger.error(f"Missing 'func' in configuration for {system_name}. Skipping.")
        return {}
    
    system_results = {}
    for i in range(args.num_simulations):
        logger.info(f"Running simulation {i+1}/{args.num_simulations} for {system_name}")
        simulation_result = run_simulation(
            system_name,
            system_config,
            args.output_dir,
            args.create_svg_gcode,
            args.create_advanced_viz,
            max_time
        )
        if simulation_result[0]:
            key, data = simulation_result
            system_results[key] = data
        else:
            logger.warning(f"Simulation {i+1} for {system_name} failed.")
    
    if system_results:
        logger.info(f"Saving data for {system_name}")
        save_data(system_results, args.output_dir)
        logger.info(f"All simulations for {system_name} completed and saved")
    else:
        logger.warning(f"No successful simulations for {system_name}")
    
    return system_results

def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")
    
    logger.info("Loading configuration")
    config = get_config(args.config)
    
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Configured attractors:")
    for system_name in config['systems']:
        logger.info(f"  - {system_name}")

    results = {}
    for system_name, system_config in config['systems'].items():
        max_time = 60.0 if system_name.lower() == 'three_scroll_system' else 30.0
        system_results = process_system(system_name, system_config, args, max_time)
        results.update(system_results)

    logger.info(f"Total number of successful simulations: {len(results)}")
    logger.info("Simulated attractors:")
    for key in results:
        logger.info(f"  - {key}")

    if results:
        logger.info("Creating summary plot")
        create_summary_plot(results, args.output_dir, smooth=True)
        logger.info(f"All attractor simulations saved in the '{args.output_dir}' folder")

        if args.create_svg_gcode:
            logger.info(f"SVG and G-code files for attractors saved in the '{args.output_dir}' folder")

        if args.train_gan:
            logger.info("Starting GAN training on simulation results")
            train_gan_on_results(results, device, config, args.output_dir)
    else:
        logger.warning("No successful simulations. Skipping summary plot, SVG/G-code creation, and GAN training.")

    logger.info("Program execution completed")

if __name__ == "__main__":
    main()