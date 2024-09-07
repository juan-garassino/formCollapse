import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 3*8),
            nn.BatchNorm1d(3*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (3, 8)),
            nn.ConvTranspose1d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
        logger.info(f"Initialized Generator with latent_dim: {latent_dim}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 3, 1)
        )
        logger.info("Initialized Discriminator")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def create_models(latent_dim: int) -> Tuple[nn.Module, nn.Module]:
    """Create and return instances of Generator and Discriminator."""
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    logger.info(f"Created Generator and Discriminator models with latent_dim: {latent_dim}")
    return generator, discriminator