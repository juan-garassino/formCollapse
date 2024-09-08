import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Callable, Tuple, Optional
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

def rk4_step(func: Callable, y: np.ndarray, t: float, dt: float, params: Dict[str, float]) -> np.ndarray:
    """Perform a single step of the RK4 method."""
    k1 = dt * func(y, t, params)
    k2 = dt * func(y + 0.5 * k1, t + 0.5 * dt, params)
    k3 = dt * func(y + 0.5 * k2, t + 0.5 * dt, params)
    k4 = dt * func(y + k3, t + dt, params)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def adaptive_rk4_step(func: Callable, y: np.ndarray, t: float, dt: float, params: Dict[str, float], tol: float) -> Tuple[np.ndarray, float, float]:
    """Perform an adaptive step of the RK4 method with error estimation."""
    y_full = rk4_step(func, y, t, dt, params)
    y_half1 = rk4_step(func, y, t, dt/2, params)
    y_half2 = rk4_step(func, y_half1, t + dt/2, dt/2, params)
    
    error = np.max(np.abs(y_full - y_half2))
    new_dt = 0.9 * dt * (tol / error)**0.2 if error > 0 else dt * 2
    
    return y_half2, new_dt, error

def simulate_system(system_func: Callable, params: Dict[str, float], 
                    initial_condition: np.ndarray, sim_time: float, 
                    max_steps: int, tol: float = 1e-6, 
                    min_dt: float = 1e-6, max_dt: float = 0.1) -> np.ndarray:
    """Simulate the system using adaptive RK4 method."""
    t = 0
    y = initial_condition
    results = [y]
    times = [t]
    dt = min(sim_time / 1000, max_dt)
    
    while t < sim_time and len(results) < max_steps:
        y_new, new_dt, error = adaptive_rk4_step(system_func, y, t, dt, params, tol)
        
        if error <= tol:
            t += dt
            y = y_new
            results.append(y)
            times.append(t)
            dt = min(max(new_dt, min_dt), max_dt)
        else:
            dt = max(new_dt, min_dt)
        
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
        
        if time.time() - start_time > max_time:
            logger.warning(f"Simulation of {system_name} exceeded time limit")
            return solution[:int(len(solution) * max_time / (time.time() - start_time))]
        
        logger.info(f"Simulation of {system_name} completed successfully")
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
    return True, "Simulation valid"

def adaptive_simulation(
    system_name: str,
    system_func: str,
    params: Dict[str, float],
    initial_condition: np.ndarray,
    sim_time: float,
    sim_steps: int,
    max_attempts: int = 5,
    max_time: float = 30.0
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Attempt to simulate the system multiple times with different parameters if necessary."""
    for attempt in range(max_attempts):
        try:
            data = simulate_attractor(system_name, system_func, params, initial_condition, sim_time, sim_steps, max_time)
            valid, message = check_simulation_validity(data, sim_steps)
            if valid:
                return True, data, f"Successful simulation on attempt {attempt + 1}"
            logger.warning(f"Attempt {attempt + 1} failed: {message}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}")
        
        # Adjust parameters for the next attempt
        params = {k: v * (0.9 + 0.2 * np.random.random()) for k, v in params.items()}
        initial_condition = np.random.randn(3)  # Generate new initial condition
    
    return False, None, f"Failed to produce valid simulation after {max_attempts} attempts"