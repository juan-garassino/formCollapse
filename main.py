import os
import argparse
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

# Set Matplotlib's logger to WARNING level
matplotlib.set_loglevel("WARNING")

from src.utils.config import get_config, generate_system_params, generate_initial_condition
from src.attractors.simulators import adaptive_simulation
from src.gan.models import create_models
from src.gan.training import train_gan
from src.utils.visualization import plot_attractor, create_summary_plot
from src.utils.data_handling import save_data
from src.utils.svg_gcode import save_svg, generate_gcode

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set logging level for Matplotlib to WARNING
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simulate strange attractors and optionally train GAN.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulations per attractor.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results.')
    parser.add_argument('--train_gan', action='store_true', help='Flag to train GAN.')
    parser.add_argument('--create_svg_gcode', action='store_true', help='Flag to create SVG and G-code files.')
    parser.add_argument('--use_odeint', action='store_true', help='Use odeint for integration instead of solve_ivp.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_file', type=str, help='File to save logs to.')
    return parser.parse_args()

def train_gan_on_results(results: Dict[str, np.ndarray], device: torch.device, config: Dict[str, Any], output_dir: str) -> None:
    """Train GAN on the simulation results and generate new data."""
    logger = logging.getLogger("gan_training")
    logger.info("Starting GAN training")
    
    all_data = np.concatenate(list(results.values()), axis=0)
    tensor_data = torch.FloatTensor(all_data).to(device)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=config['gan_params']['batch_size'], shuffle=True)

    generator, discriminator = create_models(config['gan_params']['latent_dim'])
    generator.to(device)
    discriminator.to(device)

    logger.info("Training GAN")
    train_gan(generator, discriminator, dataloader, 
              num_epochs=config['gan_params']['num_epochs'], 
              latent_dim=config['gan_params']['latent_dim'], 
              device=device)

    logger.info("Generating samples from trained GAN")
    num_samples = 1000
    with torch.no_grad():
        noise = torch.randn(num_samples, config['gan_params']['latent_dim'], device=device)
        generated_data = generator(noise).cpu().numpy().reshape(-1, 3)

    logger.info("Saving GAN-generated data")
    save_svg(generated_data[:, :2], "gan_generated_attractor", output_dir)
    generate_gcode(generated_data, "gan_generated_attractor", output_dir)

    logger.info(f"GAN-generated data saved in the '{output_dir}' folder")

def run_simulation(system_name: str, system_config: Dict[str, Any], output_dir: str, create_svg_gcode: bool, use_odeint: bool, max_time: float) -> Tuple[str, np.ndarray]:
    """Run a single simulation for a given system with adaptive simulation and time limit."""
    logger = logging.getLogger(f"simulation.{system_name}")
    logger.info(f"Starting simulation for {system_name}")
    
    params = generate_system_params(system_config)
    initial_condition = generate_initial_condition()
    
    logger.debug(f"System config: {system_config}")
    logger.debug(f"Generated params: {params}")
    logger.debug(f"Initial condition: {initial_condition}")
    
    success, data, message = adaptive_simulation(
        system_name,
        system_config['func'],
        params,
        initial_condition,
        system_config['sim_time'],
        system_config['sim_steps'],
        use_odeint,
        max_attempts=5,
        max_time=max_time
    )
    
    if not success:
        logger.error(f"Failed to simulate {system_name}: {message}")
        return None
    
    logger.info(f"Successful simulation of {system_name}: {message}")
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    key = f'{system_name}_{timestamp}'
    
    logger.info(f"Plotting attractor for {key}")
    plot_attractor(key, data, output_dir, smooth=True)
    
    if create_svg_gcode:
        logger.info(f"Creating SVG and G-code for {key}")
        save_svg(data[:, :2], f"{key}_svg", output_dir)
        generate_gcode(data, f"{key}_gcode", output_dir)
    
    logger.info(f"Simulation for {key} completed")
    logger.debug(f"Output shape: {data.shape}")
    logger.debug(f"Output standard deviation: {np.std(data, axis=0)}")

    return key, data

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
        logger.info(f"Processing system: {system_name}")
        logger.debug(f"System config: {system_config}")
        
        if 'func' not in system_config:
            logger.error(f"Missing 'func' in configuration for {system_name}. Skipping.")
            continue
        
        system_results = {}
        max_time = 60.0 if system_name.lower() == 'three_scroll_system' else 30.0
        for i in range(args.num_simulations):
            logger.info(f"Running simulation {i+1}/{args.num_simulations} for {system_name}")
            simulation_result = run_simulation(system_name, system_config, args.output_dir, args.create_svg_gcode, args.use_odeint, max_time)
            if simulation_result:
                key, data = simulation_result
                system_results[key] = data
            else:
                logger.warning(f"Simulation {i+1} for {system_name} failed.")
        
        if system_results:
            results.update(system_results)
            logger.info(f"Saving data for {system_name}")
            save_data({k: v for k, v in results.items() if k.startswith(system_name)}, args.output_dir)
            logger.info(f"All simulations for {system_name} completed and saved")
        else:
            logger.warning(f"No successful simulations for {system_name}")

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
            train_gan_on_results(results, device, config, args.output_dir)
    else:
        logger.warning("No successful simulations. Skipping summary plot, SVG/G-code creation, and GAN training.")

    logger.info("Program execution completed")

if __name__ == "__main__":
    main()