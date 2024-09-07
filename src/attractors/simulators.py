import numpy as np
from scipy.integrate import solve_ivp, odeint
from typing import Dict, Any, Callable, Tuple
import logging
from .attractors import *  # Import all attractor systems from attractors.py
from .robust_rf import simulate_rabinovich_fabrikant_robust
import time

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
    'arnold_system': arnold_system,
    'burke_shaw_system': burke_shaw_system,
    'chen_celikovsky_system': chen_celikovsky_system,
    'finance_system': finance_system,
    'newton_leipnik_system': newton_leipnik_system,
    'qi_chen_system': qi_chen_system,
    'rayleigh_benard_system': rayleigh_benard_system,
    'tsucs1_system': tsucs1_system,
    'liu_chen_system': liu_chen_system
}

def simulate_attractor(system_name: str, system_func: str, params: Dict[str, float], 
                       initial_condition: np.ndarray, sim_time: float, sim_steps: int,
                       use_odeint: bool = False, max_time: float = 30.0) -> np.ndarray:
    """Simulate the attractor system with a time limit."""
    logger.info(f"Simulating {system_name} attractor")
    logger.debug(f"Parameters: {params}")
    logger.debug(f"Initial condition: {initial_condition}")
    logger.debug(f"Simulation time: {sim_time}, steps: {sim_steps}")
    logger.debug(f"Using odeint: {use_odeint}")

    start_time = time.time()

    if system_func == 'three_scroll_system':
        return simulate_three_scroll(params, initial_condition, sim_time, sim_steps, max_time)

    t = np.linspace(0, sim_time, sim_steps)
    
    system_function = SYSTEM_FUNCTIONS[system_func]

    if use_odeint:
        try:
            solution = odeint(lambda X, t: system_function(X, t, params), initial_condition, t)
            if time.time() - start_time > max_time:
                logger.warning(f"Simulation of {system_name} exceeded time limit")
                return solution[:int(len(solution) * max_time / (time.time() - start_time))]
            logger.info(f"Simulation of {system_name} completed successfully using odeint")
            return solution
        except Exception as e:
            logger.error(f"odeint integration failed: {str(e)}")
            raise RuntimeError("odeint integration failed")
    else:
        methods = ['RK45', 'BDF', 'Radau']

        for method in methods:
            try:
                solution = solve_ivp(
                    lambda t, y: system_function(y, t, params),
                    [t[0], t[-1]], 
                    initial_condition, 
                    t_eval=t,
                    method=method,
                    rtol=1e-6,
                    atol=1e-6,
                    max_step=sim_time / 1000
                )

                if solution.success:
                    if time.time() - start_time > max_time:
                        logger.warning(f"Simulation of {system_name} exceeded time limit")
                        return solution.y.T[:int(len(solution.y.T) * max_time / (time.time() - start_time))]
                    logger.info(f"Simulation of {system_name} completed successfully using {method} method")
                    return solution.y.T

                logger.warning(f"Integration with {method} method failed. Message: {solution.message}")

            except Exception as e:
                logger.warning(f"Error occurred with {method} method: {str(e)}")

        logger.error("All integration methods failed")
        raise RuntimeError("Unable to simulate the system with any method")

def simulate_three_scroll(params: Dict[str, float], initial_condition: np.ndarray, 
                          sim_time: float, sim_steps: int, max_time: float) -> np.ndarray:
    """Optimized simulation for the Three-Scroll system."""
    logger.info("Using optimized method for Three-Scroll system")
    
    t = np.linspace(0, sim_time, sim_steps)
    y = np.zeros((sim_steps, 3))
    y[0] = initial_condition

    start_time = time.time()
    dt = sim_time / sim_steps

    for i in range(1, sim_steps):
        if time.time() - start_time > max_time:
            logger.warning("Three-Scroll simulation exceeded time limit")
            return y[:i]
        
        k1 = three_scroll_system(y[i-1], t[i-1], params)
        k2 = three_scroll_system(y[i-1] + 0.5 * dt * k1, t[i-1] + 0.5 * dt, params)
        k3 = three_scroll_system(y[i-1] + 0.5 * dt * k2, t[i-1] + 0.5 * dt, params)
        k4 = three_scroll_system(y[i-1] + dt * k3, t[i-1] + dt, params)
        
        y[i] = y[i-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    logger.info("Three-Scroll simulation completed successfully")
    return y

def simulate_attractor_robust(system_name: str, system_func: str, params: Dict[str, float], 
                              initial_condition: np.ndarray, sim_time: float, sim_steps: int,
                              use_odeint: bool = False) -> np.ndarray:
    """Attempt to simulate the attractor using different methods if the primary method fails."""
    try:
        return simulate_attractor(system_name, system_func, params, initial_condition, sim_time, sim_steps, use_odeint)
    except RuntimeError:
        logger.warning("Primary simulation method failed. Attempting custom RK4 method.")
        return simulate_custom_rk4(system_name, system_func, params, initial_condition, sim_time, sim_steps)

def simulate_custom_rk4(system_name: str, system_func: str, params: Dict[str, float], 
                        initial_condition: np.ndarray, sim_time: float, sim_steps: int) -> np.ndarray:
    """Custom simulation using 4th order Runge-Kutta method."""
    logger.info(f"Using custom RK4 method for {system_name} system")
    
    t = np.linspace(0, sim_time, sim_steps)
    y = np.zeros((len(t), 3))
    y[0] = initial_condition

    system_function = SYSTEM_FUNCTIONS[system_func]

    for i in range(1, len(t)):
        y[i] = rk4_step(y[i-1], t[i-1], t[i] - t[i-1], params, system_function)
        
        if np.any(np.isnan(y[i])) or np.any(np.isinf(y[i])):
            logger.warning(f"NaN or Inf values detected at step {i}. Stopping simulation.")
            return y[:i]

    logger.info(f"{system_name} simulation completed successfully using custom RK4")
    return y

def rk4_step(y: np.ndarray, t: float, dt: float, params: Dict[str, float], system_func: Callable) -> np.ndarray:
    """Perform a single step of the RK4 method."""
    k1 = dt * system_func(y, t, params)
    k2 = dt * system_func(y + 0.5 * k1, t + 0.5 * dt, params)
    k3 = dt * system_func(y + 0.5 * k2, t + 0.5 * dt, params)
    k4 = dt * system_func(y + k3, t + dt, params)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Additional utility functions can be added here if needed

def check_simulation_validity(data: np.ndarray, sim_steps: int) -> Tuple[bool, str]:
    """Check if the simulation results are valid."""
    if len(data) < sim_steps // 2:
        return False, "Simulation terminated too early"
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False, "Simulation produced NaN or Inf values"
    if np.all(np.std(data, axis=0) < 1e-3):
        return False, "Simulation produced simple output"
    return True, "Simulation valid"

def adaptive_simulation(system_name: str, system_func: str, params: Dict[str, float],
                        initial_condition: np.ndarray, sim_time: float, sim_steps: int,
                        use_odeint: bool = False, max_attempts: int = 5, max_time: float = 30.0) -> Tuple[bool, np.ndarray, str]:
    """Attempt to simulate the system multiple times with different parameters if necessary."""
    for attempt in range(max_attempts):
        try:
            data = simulate_attractor(system_name, system_func, params, initial_condition, sim_time, sim_steps, use_odeint, max_time)
            valid, message = check_simulation_validity(data, sim_steps)
            if valid:
                return True, data, f"Successful simulation on attempt {attempt + 1}"
            logger.warning(f"Attempt {attempt + 1} failed: {message}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}")
        
        # Adjust parameters for the next attempt
        params = {k: v * (0.9 + 0.2 * np.random.random()) for k, v in params.items()}
        initial_condition = np.random.randn(3)  # Generate new initial condition
    
    return False, np.array([]), f"Failed to produce valid simulation after {max_attempts} attempts"