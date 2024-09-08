import yaml
import logging
from typing import Dict, Any, Tuple
import os
import numpy as np

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise

def generate_system_params(config: Dict[str, Any], system_name: str) -> Dict[str, Any]:
    system_config = config['systems'][system_name]
    params = {}
    for param_name, param_config in system_config['params'].items():
        if isinstance(param_config, dict) and 'range' in param_config:
            params[param_name] = np.random.uniform(*param_config['range'])
        else:
            params[param_name] = param_config
    
    # Add simulation time and steps, using global defaults if not specified
    params['sim_time'] = system_config.get('sim_time', config['global']['default_sim_time'])
    params['sim_steps'] = system_config.get('sim_steps', config['global']['default_sim_steps'])
    params['scale'] = system_config['scale']
    
    return params

# def generate_initial_condition(system_config: Dict[str, Any], attempt: int) -> np.ndarray:
#     """Generate an initial condition based on the system scale and attempt number."""
#     scale = system_config.get('scale', 1.0)
#     if attempt == 0:
#         # First attempt: use a small perturbation from origin
#         return np.random.randn(3) * 0.1 * scale
#     else:
#         # Subsequent attempts: use full scale with increasing randomness
#         return np.random.randn(3) * scale * (1 + attempt * 0.2)

def generate_initial_condition(scale: float, attempt: int) -> np.ndarray:
    """Generate an initial condition based on the system scale and attempt number."""
    if attempt == 0:
        # First attempt: use a small perturbation from origin
        return np.random.randn(3) * 0.1 * scale
    else:
        # Subsequent attempts: use full scale with increasing randomness
        return np.random.randn(3) * scale * (1 + attempt * 0.2)

def get_config(config_path: str) -> Dict[str, Any]:
    """
    Get the configuration from the specified YAML file.
    """
    if not config_path:
        raise ValueError("Configuration file path must be provided.")
    return load_config(config_path)