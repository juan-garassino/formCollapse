import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Callable, Tuple, Optional, List
import logging
import time
from .attractors import *

logger = logging.getLogger(__name__)

SYSTEM_FUNCTIONS: Dict[str, Callable] = {
    'lorenz_system': lorenz_system, # ESTE
    'aizawa_system': aizawa_system, # ESTE
    'rabinovich_fabrikant_system': rabinovich_fabrikant_system, # ESTE
    # #'three_scroll_system': three_scroll_system,
    # 'rossler_system': rossler_system,
    'chen_system': chen_system, # ESTE
    'halvorsen_system': halvorsen_system, # ESTE
    # 'anishchenko_system': anishchenko_system,
    # 'arnold_system': arnold_system,
    # #'burke_shaw_system': burke_shaw_system,
    # 'chen_celikovsky_system': chen_celikovsky_system,
    # 'finance_system': finance_system,
    'newton_leipnik_system': newton_leipnik_system, # ESTE
    # 'qi_chen_system': qi_chen_system,
    # 'rayleigh_benard_system': rayleigh_benard_system,
    # 'tsucs1_system': tsucs1_system,
    # 'liu_chen_system': liu_chen_system
}

SYSTEM_SCALES: Dict[str, Dict[str, Any]] = {
    'lorenz_system': {'scale': 20.0, 'params': {'sigma': (9, 11), 'beta': (2, 3), 'rho': (20, 30)}},
    'aizawa_system': {'scale': 1.0, 'params': {'a': (0.7, 0.95), 'b': (0.7, 0.85), 'c': (0.4, 0.6), 'd': (3.0, 3.5), 'e': (0.25, 0.3), 'f': (0.07, 0.12)}},
    'rabinovich_fabrikant_system': {'scale': 1.0, 'params': {'alpha': (0.14, 0.17), 'gamma': (0.98, 1.02)}},
    'three_scroll_system': {'scale': 20.0, 'params': {'a': (38, 42), 'b': (52, 58), 'c': (1.7, 2.0)}},
    'rossler_system': {'scale': 10.0, 'params': {'a': (0.1, 0.2), 'b': (0.1, 0.2), 'c': (5.0, 5.7)}},
    'chen_system': {'scale': 20.0, 'params': {'a': (35, 36), 'b': (3, 3.5), 'c': (20, 28)}},
    'halvorsen_system': {'scale': 5.0, 'params': {'a': (1.27, 1.3)}},
    'anishchenko_system': {'scale': 10.0, 'params': {'a': (1.0, 1.2), 'b': (0.5, 0.7), 'c': (0.7, 0.9)}},
    'burke_shaw_system': {'scale': 10.0, 'params': {'a': (10, 11), 'b': (4, 4.5), 'c': (2.7, 3.0)}},
    'chen_celikovsky_system': {'scale': 20.0, 'params': {'a': (35, 36), 'c': (27, 28), 'd': (1.8, 2.0)}},
    'finance_system': {'scale': 1.0, 'params': {'a': (0.95, 1.0), 'b': (0.2, 0.3), 'c': (1.0, 1.1)}},
    'newton_leipnik_system': {'scale': 0.5, 'params': {'a': (0.3, 0.5), 'b': (0.1, 0.2)}},
    'qi_chen_system': {'scale': 15.0, 'params': {'a': (35, 36), 'b': (3, 3.5), 'c': (20, 28)}},
    'rayleigh_benard_system': {'scale': 10.0, 'params': {'a': (9, 10), 'b': (5, 6), 'c': (12, 13)}},
    'tsucs1_system': {'scale': 1.0, 'params': {'a': (40, 41), 'b': (0.16, 0.17), 'c': (0.95, 1.05)}},
    'liu_chen_system': {'scale': 20.0, 'params': {'a': (5, 5.5), 'b': (-10, -9.5), 'c': (-3.8, -3.6), 'd': (1, 1.1)}}
}

def generate_initial_condition(scale: float, attempt: int) -> np.ndarray:
    """Generate an initial condition based on the system scale and attempt number."""
    if attempt == 0:
        # First attempt: use a small perturbation from origin
        return np.random.randn(3) * 0.1 * scale
    else:
        # Subsequent attempts: use full scale with increasing randomness
        return np.random.randn(3) * scale * (1 + attempt * 0.2)

def adaptive_simulation(
    system_name: str,
    system_func: str,
    params: Dict[str, float],
    sim_time: float,
    sim_steps: int,
    max_attempts: int = 10,
    max_time: float = 30.0,
    scale: float = 1.0
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Attempt to simulate the system multiple times with different initial conditions."""
    start_time = time.time()
    system_function = SYSTEM_FUNCTIONS.get(system_func)
    if system_function is None:
        raise ValueError(f"Unknown system function: {system_func}")

    for attempt in range(max_attempts):
        initial_condition = generate_initial_condition(scale, attempt)
        try:
            # Try different ODE solvers
            for method in ['RK45', 'BDF', 'Radau']:
                solution = solve_ivp(
                    lambda t, y: system_function(y, t, params),
                    (0, sim_time),
                    initial_condition,
                    method=method,
                    t_eval=np.linspace(0, sim_time, sim_steps),
                    rtol=1e-6,
                    atol=1e-9
                )
                
                if solution.success:
                    data = solution.y.T
                    valid, message = check_simulation_validity(data, sim_steps)
                    if valid:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_time:
                            return True, data[:int(len(data) * max_time / elapsed_time)], f"Successful simulation (truncated due to time limit) using {method}"
                        return True, data, f"Successful simulation using {method}"
                
                logger.info(f"Attempt {attempt + 1} with {method} failed: {solution.message}")
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}")
    
    return False, None, f"Failed to produce valid simulation after {max_attempts} attempts"

def check_simulation_validity(data: np.ndarray, sim_steps: int) -> Tuple[bool, str]:
    """Check if the simulation results are valid."""
    if data is None or len(data) < sim_steps // 2:
        return False, "Simulation terminated too early"
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False, "Simulation produced NaN or Inf values"
    if np.all(np.std(data, axis=0) < 1e-3):
        return False, "Simulation produced simple output"
    
    # Check for chaotic behavior
    lyapunov_exp = estimate_lyapunov_exponent(data)
    if lyapunov_exp <= 0:
        return False, f"System might not be chaotic. Estimated Lyapunov exponent: {lyapunov_exp:.4f}"
    
    return True, f"Simulation valid. Estimated Lyapunov exponent: {lyapunov_exp:.4f}"

def estimate_lyapunov_exponent(data: np.ndarray, dt: float = 1.0) -> float:
    """Estimate the largest Lyapunov exponent."""
    n_steps = len(data)
    d0 = np.linalg.norm(data[1] - data[0])
    d1 = np.linalg.norm(data[-1] - data[-2])
    return np.log(d1 / d0) / ((n_steps - 1) * dt)

# Additional utility functions can be added here if needed

# Example of a utility function that could be useful:
def analyze_simulation_results(data: np.ndarray) -> Dict[str, Any]:
    """Analyze the simulation results and return various metrics."""
    metrics = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'lyapunov_exponent': estimate_lyapunov_exponent(data)
    }
    return metrics

# You can add more functions here as needed for your specific requirements