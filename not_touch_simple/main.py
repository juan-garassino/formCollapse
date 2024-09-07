import os
import numpy as np
import argparse
from attractors.attractors import lorenz_system, aizawa_system, rabinovich_fabrikant_system, three_scroll_system, simulate_system
from utils import preprocess_input, plot_attractors, create_summary_plot, save_data

def main(num_simulations, output_dir):
    # Time range for generating the attractor points
    t = np.linspace(0, 100, 20000)

    # Initialize a dictionary to hold results for each attractor
    results = {}

    # Define the systems to simulate
    systems = {
        'Lorenz': lorenz_system,
        'Aizawa': aizawa_system,
        'Rabinovich-Fabrikant': rabinovich_fabrikant_system,
        'Three-Scroll': three_scroll_system
    }

    for system_name, system_func in systems.items():
        for i in range(num_simulations):
            # Preprocess input (high-dimensional sample for each simulation)
            X0 = preprocess_input(dim=3)

            # Simulate the system
            data = simulate_system(system_func, X0, t, {})
            results[f'{system_name}_{i + 1}'] = data

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Plot and save attractor results
    plot_attractors(results)

    # Create and save summary plot
    create_summary_plot(results)

    # Save raw data
    save_data(results, "raw_data")

    print(f"All {num_simulations} simulations for each attractor saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate and plot strange attractors.")
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulations to run for each attractor.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results.')
    args = parser.parse_args()

    main(args.num_simulations, args.output_dir)