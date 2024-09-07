import yaml
import logging
from typing import Dict, Any
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
        return DEFAULT_CONFIG
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            logger.warning("Using default configuration.")
            return DEFAULT_CONFIG

def generate_system_params(system_config: Dict[str, Any]) -> Dict[str, float]:
    """Generate random parameters for a system based on its configuration."""
    params = {}
    for k, v in system_config['params'].items():
        if isinstance(v, dict) and 'type' in v and 'range' in v:
            if v['type'] == 'uniform':
                params[k] = np.random.uniform(*v['range'])
            else:
                params[k] = v['value']
        else:
            # If it's not a dict or doesn't have 'type' and 'range', assume it's a direct value
            params[k] = v
    return params

def generate_initial_condition(dim: int = 3) -> np.ndarray:
    """Generate a random initial condition."""
    return np.random.randn(dim)

# Default configuration
DEFAULT_CONFIG = {
    'systems': {
        'Lorenz': {
            'func': 'lorenz_system',
            'params': {
                'sigma': {'type': 'uniform', 'range': [9, 11]},
                'beta': {'type': 'uniform', 'range': [2, 3]},
                'rho': {'type': 'uniform', 'range': [20, 30]}
            },
            'sim_time': 100,
            'sim_steps': 10000
        },
        'Aizawa': {
            'func': 'aizawa_system',
            'params': {
                'a': {'type': 'uniform', 'range': [0.7, 1.0]},
                'b': {'type': 'uniform', 'range': [0.6, 0.8]},
                'c': {'type': 'uniform', 'range': [0.3, 0.7]},
                'd': {'type': 'uniform', 'range': [3.0, 4.0]},
                'e': {'type': 'uniform', 'range': [0.2, 0.3]},
                'f': {'type': 'uniform', 'range': [0.05, 0.15]}
            },
            'sim_time': 100,
            'sim_steps': 10000
        },
        'Rabinovich-Fabrikant': {
            'func': 'rabinovich_fabrikant_system',
            'params': {
                'alpha': {'type': 'uniform', 'range': [0.1, 0.3]},
                'gamma': {'type': 'uniform', 'range': [0.05, 0.25]}
            },
            'sim_time': 50,
            'sim_steps': 5000
        },
        'Three-Scroll': {
            'func': 'three_scroll_system',
            'params': {
                'a': {'type': 'uniform', 'range': [32, 48]},
                'b': {'type': 'uniform', 'range': [45, 65]},
                'c': {'type': 'uniform', 'range': [1.5, 2.2]}
            },
            'sim_time': 50,
            'sim_steps': 5000
        }
    },
    'sim_params': {
        'method': 'RK45',
        'rtol': 1e-6,
        'atol': 1e-9
    },
    'gan_params': {
        'latent_dim': 100,
        'batch_size': 64,
        'num_epochs': 50
    }
}

def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    Get the configuration, either from a file or the default.
    If a file is provided, it will be merged with the default configuration.
    """
    if config_path:
        user_config = load_config(config_path)
        # Merge user config with default config
        for key, value in user_config.items():
            if isinstance(value, dict) and key in DEFAULT_CONFIG:
                DEFAULT_CONFIG[key].update(value)
            else:
                DEFAULT_CONFIG[key] = value
    return DEFAULT_CONFIG