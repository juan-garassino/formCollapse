import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Callable, Tuple, Optional, List
import logging
import time
from .attractors import *

logger = logging.getLogger(__name__)

SYSTEM_FUNCTIONS: Dict[str, Callable] = {
    'lorenz_system': lorenz_system,
    'aizawa_system': aizawa_system,
    'rabinovich_fabrikant_system': rabinovich_fabrikant_system,
    'three_scroll_system': three_scroll_system,
    'rossler_system': rossler_system,
    'chen_system': chen_system,
    'halvorsen_system': halvorsen_system,
    'anishchenko_system': anishchenko_system,
    'burke_shaw_system': burke_shaw_system,
    'chen_celikovsky_system': chen_celikovsky_system,
    'finance_system': finance_system,
    'newton_leipnik_system': newton_leipnik_system,
    'qi_chen_system': qi_chen_system,
    'rayleigh_benard_system': rayleigh_benard_system,
    'tsucs1_system': tsucs1_system,
    'liu_chen_system': liu_chen_system
}

# Define typical scales for each system
SYSTEM_SCALES: Dict[str, float] = {
    'lorenz_system': 20.0,
    'aizawa_system': 1.0,
    'rabinovich_fabrikant_system': 1.0,
    'three_scroll_system': 20.0,
    'rossler_system': 10.0,
    'chen_system': 20.0,
    'halvorsen_system': 5.0,
    'anishchenko_system': 10.0,
    'burke_shaw_system': 10.0,
    'chen_celikovsky_system': 20.0,
    'finance_system': 1.0,
    'newton_leipnik_system': 0.5,
    'qi_chen_system': 15.0,
    'rayleigh_benard_system': 10.0,
    'tsucs1_system': 1.0,
    'liu_chen_system': 20.0
}

def generate_initial_condition(system_name: str, attempt: int) -> np.ndarray:
    """Generate an initial condition based on the system and attempt number."""
    scale = SYSTEM_SCALES.get(system_name, 10.0)
    
    if attempt == 0:
        # First attempt: use a small perturbation from origin
        return np.random.randn(3) * 0.1 * scale
    elif attempt == 1:
        # Second attempt: use full scale
        return np.random.randn(3) * scale
    else:
        # Subsequent attempts: gradually increase the scale
        return np.random.randn(3) * scale * (1 + attempt * 0.5)

def rk45_step(func: Callable, y: np.ndarray, t: float, dt: float, params: Dict[str, float]) -> Tuple[np.ndarray, float, float]:
    """Perform a single step of the Runge-Kutta-Fehlberg (RKF45) method."""
    k1 = dt * func(y, t, params)
    k2 = dt * func(y + k1/4, t + dt/4, params)
    k3 = dt * func(y + 3*k1/32 + 9*k2/32, t + 3*dt/8, params)
    k4 = dt * func(y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197, t + 12*dt/13, params)
    k5 = dt * func(y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104, t + dt, params)
    k6 = dt * func(y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40, t + dt/2, params)

    y_new = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
    y_err = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5

    error = np.max(np.abs(y_new - y_err))
    return y_new, error

def simulate_system(system_func: Callable, params: Dict[str, float], 
                    initial_condition: np.ndarray, sim_time: float, 
                    max_steps: int, tol: float = 1e-6, 
                    min_dt: float = 1e-6, max_dt: float = 0.1) -> np.ndarray:
    """Simulate the system using adaptive RK45 method."""
    t = 0
    y = initial_condition
    results = [y]
    times = [t]
    dt = min(sim_time / 1000, max_dt)
    
    while t < sim_time and len(results) < max_steps:
        y_new, error = rk45_step(system_func, y, t, dt, params)
        
        if error <= tol:
            t += dt
            y = y_new
            results.append(y)
            times.append(t)
            dt = min(max(0.9 * dt * (tol / error)**0.2, min_dt), max_dt)
        else:
            dt = max(0.9 * dt * (tol / error)**0.2, min_dt)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            logger.warning(f"NaN or Inf values detected at t={t}. Stopping simulation.")
            break
    
    return np.array(results)

def simulate_attractor(
    system_name: str,
    system_func: str,
    params: Dict[str, float],
    initial_condition: np.ndarray,
    sim_time: float,
    sim_steps: int,
    max_time: float = 30.0
) -> Optional[np.ndarray]:
    """Simulate the attractor system with a time limit."""
    logger.info(f"Simulating {system_name} attractor")
    logger.debug(f"Parameters: {params}")
    logger.debug(f"Initial condition: {initial_condition}")
    logger.debug(f"Simulation time: {sim_time}, steps: {sim_steps}")

    start_time = time.time()
    
    system_function = SYSTEM_FUNCTIONS.get(system_func)
    if system_function is None:
        raise ValueError(f"Unknown system function: {system_func}")

    try:
        solution = simulate_system(system_function, params, initial_condition, sim_time, sim_steps)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            logger.warning(f"Simulation of {system_name} exceeded time limit. Elapsed time: {elapsed_time:.2f}s")
            return solution[:int(len(solution) * max_time / elapsed_time)]
        
        logger.info(f"Simulation of {system_name} completed successfully in {elapsed_time:.2f}s")
        return solution
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return None

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

def adaptive_simulation(
    system_name: str,
    system_func: str,
    params: Dict[str, float],
    sim_time: float,
    sim_steps: int,
    max_attempts: int = 10,
    max_time: float = 30.0
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Attempt to simulate the system multiple times with different initial conditions."""
    for attempt in range(max_attempts):
        initial_condition = generate_initial_condition(system_name, attempt)
        try:
            data = simulate_attractor(system_name, system_func, params, initial_condition, sim_time, sim_steps, max_time)
            valid, message = check_simulation_validity(data, sim_steps)
            if valid:
                return True, data, f"Successful simulation on attempt {attempt + 1}. {message}"
            logger.warning(f"Attempt {attempt + 1} failed: {message}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}")
    
    return False, None, f"Failed to produce valid simulation after {max_attempts} attempts"