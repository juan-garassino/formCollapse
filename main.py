import os
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

from src.utils.config import get_config, generate_system_params
from src.attractors.simulators import adaptive_simulation, SYSTEM_FUNCTIONS
from src.gan.training import train_gan_on_results
from src.utils.visualization import (
    plot_attractor, create_summary_plot, plot_phase_space, 
    plot_time_series, plot_poincare_section, plot_power_spectrum, 
    plot_bifurcation
)
from src.utils.data_handling import save_data
from src.utils.svg_gcode import save_svg, generate_gcode
from src.utils.animations import animate_3d

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
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file.')
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulations per attractor.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results.')
    parser.add_argument('--create_svg_gcode', action='store_true', help='Flag to create SVG and G-code files.')
    parser.add_argument('--create_advanced_viz', action='store_true', help='Flag to create advanced visualizations.')
    parser.add_argument('--train_gan', action='store_true', help='Flag to train GAN.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_file', type=str, help='File to save logs to.')
    return parser.parse_args()

def run_simulation(system_name: str, system_config: Dict[str, Any], params: Dict[str, Any], max_time: float) -> Tuple[bool, Optional[np.ndarray], str]:
    logger = logging.getLogger(f"simulation.{system_name}")
    logger.info(f"Starting simulation for {system_name}")
    
    success, data, message = adaptive_simulation(
        system_name,
        system_config['func'],
        params,
        params['sim_time'],
        params['sim_steps'],
        max_attempts=10,
        max_time=max_time,
        scale=system_config['scale']
    )
    
    if not success:
        logger.error(f"Failed to simulate {system_name}: {message}")
    else:
        logger.info(f"Successful simulation of {system_name}: {message}")
    
    return success, data, message

def process_system(system_name: str, system_config: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace, max_time: float) -> Dict[str, np.ndarray]:
    logger = logging.getLogger(f"process.{system_name}")
    logger.info(f"Processing system: {system_name}")
    
    results = {}
    for i in range(args.num_simulations):
        logger.info(f"Running simulation {i+1}/{args.num_simulations} for {system_name}")
        params = generate_system_params(config, system_name)
        success, data, message = run_simulation(system_name, system_config, params, max_time)
        
        if success:
            key = f'{system_name}_{datetime.now().strftime("%m%d_%H%M%S")}'
            results[key] = data
            logger.info(f"Simulation {i+1} for {system_name} succeeded: {message}")
            
            # Plotting
            logger.info(f"Plotting attractor for {key}")
            plot_attractor(key, data, args.output_dir, smooth=True)
            plot_phase_space(key, data, args.output_dir, smooth=True)
            plot_time_series(key, data, args.output_dir, smooth=True)
            
            if args.create_advanced_viz:
                logger.info(f"Creating advanced visualizations for {key}")
                animate_3d(key, data, args.output_dir)
                for plane in ['xy', 'yz', 'xz']:
                    plot_poincare_section(key, data, args.output_dir, plane=plane)
                plot_power_spectrum(key, data, args.output_dir)
                
                if 'bifurcation_param' in system_config:
                    param_name = system_config['bifurcation_param']['name']
                    param_range = np.linspace(
                        system_config['bifurcation_param']['start'],
                        system_config['bifurcation_param']['stop'],
                        system_config['bifurcation_param']['num']
                    )
                    plot_bifurcation(key, SYSTEM_FUNCTIONS[system_config['func']], param_range, param_name, args.output_dir)
                else:
                    logger.info(f"Skipping bifurcation diagram for {key}: No bifurcation parameter specified in config")
            
            if args.create_svg_gcode:
                logger.info(f"Creating SVG and G-code for {key}")
                save_svg(data[:, :2], f"{key}_svg", args.output_dir)
                generate_gcode(data, f"{key}_gcode", args.output_dir)
        else:
            logger.warning(f"Simulation {i+1} for {system_name} failed: {message}")
    
    if results:
        logger.info(f"Saving data for {system_name}")
        save_data(results, args.output_dir)
        logger.info(f"All simulations for {system_name} completed and saved")
    else:
        logger.warning(f"No successful simulations for {system_name}")
    
    return results

def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")
    
    logger.info("Loading configuration")
    try:
        config = get_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return

    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Configured attractors:")
    for system_name in config['systems']:
        logger.info(f"  - {system_name}")

    results = {}
    for system_name, system_config in config['systems'].items():
        if 'func' not in system_config:
            logger.error(f"Missing 'func' in configuration for {system_name}. Skipping.")
            continue
        
        max_time = 60.0 if system_name.lower() == 'three_scroll_system' else 30.0
        system_results = process_system(system_name, system_config, config, args, max_time)
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