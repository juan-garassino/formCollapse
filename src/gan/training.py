import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple
import logging
from .models import create_models
from ..utils.visualization import plot_attractor, plot_phase_space, plot_time_series
from ..utils.svg_gcode import save_svg, generate_gcode
from ..utils.data_handling import save_data

logger = logging.getLogger(__name__)

def preprocess_data(results: Dict[str, np.ndarray]) -> torch.Tensor:
    """Preprocess the simulation results for GAN training."""
    all_data = np.concatenate(list(results.values()), axis=0)
    # Normalize the data
    data_mean = np.mean(all_data, axis=0)
    data_std = np.std(all_data, axis=0)
    normalized_data = (all_data - data_mean) / data_std
    return torch.FloatTensor(normalized_data)

def train_gan(generator: nn.Module, discriminator: nn.Module, dataloader: DataLoader, 
              num_epochs: int, latent_dim: int, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """Train the GAN and return the trained generator and discriminator."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    logger.info(f"Starting GAN training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            label = torch.full((batch_size,), 1, device=device)
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(noise)
            label.fill_(0)
            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            label.fill_(1)
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizer_G.step()

        logger.info(f'[Epoch {epoch+1}/{num_epochs}] Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}')

    logger.info("GAN training completed")
    return generator, discriminator

def generate_samples(generator: nn.Module, latent_dim: int, num_samples: int, device: torch.device) -> np.ndarray:
    """Generate samples using the trained generator."""
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_data = generator(noise).cpu().numpy()
    return generated_data

def train_gan_on_results(results: Dict[str, np.ndarray], device: torch.device, config: Dict[str, Any], output_dir: str) -> None:
    """Train GAN on the simulation results and generate new data."""
    logger.info("Starting GAN training on simulation results")
    
    # Preprocess data
    tensor_data = preprocess_data(results)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=config['gan_params']['batch_size'], shuffle=True)

    # Create and train GAN
    generator, discriminator = create_models(config['gan_params']['latent_dim'])
    generator.to(device)
    discriminator.to(device)

    generator, discriminator = train_gan(generator, discriminator, dataloader, 
                                         num_epochs=config['gan_params']['num_epochs'], 
                                         latent_dim=config['gan_params']['latent_dim'], 
                                         device=device)

    # Generate samples
    num_samples = 1000
    generated_data = generate_samples(generator, config['gan_params']['latent_dim'], num_samples, device)

    # Denormalize the generated data
    data_mean = np.mean(np.concatenate(list(results.values()), axis=0), axis=0)
    data_std = np.std(np.concatenate(list(results.values()), axis=0), axis=0)
    denormalized_data = generated_data * data_std + data_mean

    # Visualize and save generated data
    logger.info("Saving GAN-generated data visualizations")
    plot_attractor("gan_generated", denormalized_data, output_dir, smooth=True)
    plot_phase_space("gan_generated", denormalized_data, output_dir, smooth=True)
    plot_time_series("gan_generated", denormalized_data, output_dir, smooth=True)

    # Save SVG and G-code
    save_svg(denormalized_data[:, :2], "gan_generated_attractor", output_dir)
    generate_gcode(denormalized_data, "gan_generated_attractor", output_dir)

    logger.info(f"GAN-generated data saved in the '{output_dir}' folder")