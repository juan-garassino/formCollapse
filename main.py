import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import odeint
from datetime import datetime
import hashlib

# Import all attractors from a single file
from src.attractors import lorenz_system, aizawa_system, rabinovich_fabrikant_system, three_scroll_system

# Import GAN-related functions
from src.gan.models import Generator, Discriminator
from src.gan.training import train_gan

# Import utility functions
from src.utils.general import preprocess_input, plot_attractors, create_summary_plot, save_data
from src.utils.svg_gcode import save_svg, generate_gcode

def simulate_system(system_func, X0, t, params):
    """Simulate the system using SciPy's odeint."""
    return odeint(system_func, X0, t, args=(params,))

def generate_seed(system_name, index):
    """Generate a valid seed for NumPy's random number generator."""
    unique_string = f"{system_name}_{index}"
    hash_object = hashlib.sha256(unique_string.encode())
    hash_digest = hash_object.digest()
    seed = int.from_bytes(hash_digest[:4], byteorder='big')
    return seed % (2**32 - 1)

def generate_initial_condition(rng):
    """Generate a more diverse initial condition."""
    return rng.uniform(-10, 10, size=3)

def generate_system_params(system_name, rng):
    """Generate system-specific parameters."""
    if system_name == 'Lorenz':
        return {
            'sigma': rng.uniform(9, 11),
            'beta': rng.uniform(2, 3),
            'rho': rng.uniform(20, 30)
        }
    elif system_name == 'Aizawa':
        return {
            'a': rng.uniform(0.7, 1.0),
            'b': rng.uniform(0.6, 0.8),
            'c': rng.uniform(0.3, 0.7),
            'd': rng.uniform(3.0, 4.0),
            'e': rng.uniform(0.2, 0.3),
            'f': rng.uniform(0.05, 0.15)
        }
    elif system_name == 'Rabinovich-Fabrikant':
        return {
            'alpha': rng.uniform(0.1, 0.2),
            'gamma': rng.uniform(0.05, 0.15)
        }
    elif system_name == 'Three-Scroll':
        return {
            'a': rng.uniform(35, 45),
            'b': rng.uniform(50, 60),
            'c': rng.uniform(1.5, 2.0)
        }
    else:
        return {}

def main(num_simulations, output_dir, train_gan_flag, create_svg_gcode):
    # Set up PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Time range for generating the attractor points
    t = np.linspace(0, 100, 10000)

    # Initialize a dictionary to hold results for each attractor
    results = {}

    # Define the systems to simulate
    systems = {
        'Lorenz': lorenz_system,
        'Aizawa': aizawa_system,
        'Rabinovich-Fabrikant': rabinovich_fabrikant_system,
        'Three-Scroll': three_scroll_system
    }

    # Simulate attractors
    for system_name, system_func in systems.items():
        for i in range(num_simulations):
            seed = generate_seed(system_name, i)
            rng = np.random.default_rng(seed)
            
            X0 = generate_initial_condition(rng)
            params = generate_system_params(system_name, rng)
            
            data = simulate_system(system_func, X0, t, params)
            
            # Check if the output is too simple (e.g., a line)
            if np.all(np.std(data, axis=0) < 1e-6):
                print(f"Warning: Simple output detected for {system_name} simulation {i+1}. Regenerating...")
                continue  # Skip this iteration and try again
            
            timestamp = datetime.now().strftime("%m%d_%H%M")
            key = f'{system_name}_{i+1:02d}_{timestamp}'
            results[key] = data
            
            if create_svg_gcode:
                # Generate SVG and G-code for each simulation
                save_svg(data[:, :2], f"{key}_svg", output_dir)
                generate_gcode(data, f"{key}_gcode", output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save attractor results
    plot_attractors(results, output_dir)

    # Create and save summary plot
    create_summary_plot(results, output_dir)

    # Save raw data
    save_data(results, output_dir)

    print(f"Attractor simulations saved in the '{output_dir}' folder.")

    if create_svg_gcode:
        print(f"SVG and G-code files for attractors saved in the '{output_dir}' folder.")

    if train_gan_flag:
        # Prepare data for GAN
        all_data = np.concatenate(list(results.values()), axis=0)
        tensor_data = torch.FloatTensor(all_data).to(device)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Set up GAN
        latent_dim = 100
        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)

        # Train GAN
        train_gan(generator, discriminator, dataloader, num_epochs=50, latent_dim=latent_dim, device=device)

        # Generate new data using GAN
        num_samples = 1000
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim, device=device)
            generated_data = generator(noise).cpu().numpy().reshape(-1, 3)

        # Save GAN-generated data as SVG and G-code
        save_svg(generated_data[:, :2], "gan_generated_attractor", output_dir)
        generate_gcode(generated_data, "gan_generated_attractor", output_dir)

        print(f"GAN-generated data saved in the '{output_dir}' folder.")
    else:
        print("GAN training skipped. Only attractor samples are generated and plotted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate strange attractors and optionally train GAN to generate SVG and G-code for pen plotting.")
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulations to run for each attractor.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results.')
    parser.add_argument('--train_gan', action='store_true', help='Flag to train GAN. If not set, only attractor samples will be generated.')
    parser.add_argument('--create_svg_gcode', action='store_true', help='Flag to create SVG and G-code files for each attractor simulation.')
    args = parser.parse_args()

    main(args.num_simulations, args.output_dir, args.train_gan, args.create_svg_gcode)