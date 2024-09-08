# import numpy as np
# from typing import Dict, Tuple
# import logging

# logger = logging.getLogger(__name__)

# def rabinovich_fabrikant_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
#     """
#     Compute the derivatives for the Rabinovich-Fabrikant system.
#     Takes the current state X, time t, and system parameters as input.
#     Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
#     """
#     x, y, z = X
#     alpha, gamma = params['alpha'], params['gamma']
#     dx = y * (z - 1 + x**2) + gamma * x
#     dy = x * (3 * z + 1 - x**2) + gamma * y
#     dz = -2 * z * (alpha + x * y)
#     return np.array([dx, dy, dz])

# def rk4_step_rf(y: np.ndarray, t: float, dt: float, params: Dict[str, float]) -> Tuple[np.ndarray, float]:
#     """Perform a single step of the RK4 method for Rabinovich-Fabrikant system with error estimation."""
#     k1 = dt * rabinovich_fabrikant_system(y, t, params)
#     k2 = dt * rabinovich_fabrikant_system(y + 0.5 * k1, t + 0.5 * dt, params)
#     k3 = dt * rabinovich_fabrikant_system(y + 0.5 * k2, t + 0.5 * dt, params)
#     k4 = dt * rabinovich_fabrikant_system(y + k3, t + dt, params)
    
#     # 4th order estimate
#     y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
#     # 5th order estimate for error calculation
#     k5 = dt * rabinovich_fabrikant_system(y_new, t + dt, params)
#     y_5th = y + (k1 + 4*k2 + k3 + 4*k4 + k5) / 10
    
#     error = np.linalg.norm(y_new - y_5th)
#     return y_new, error

# def simulate_rabinovich_fabrikant_robust(params: Dict[str, float], initial_condition: np.ndarray, 
#                                          sim_time: float, max_steps: int) -> np.ndarray:
#     """Robust simulation for Rabinovich-Fabrikant system using adaptive RK4 method."""
#     logger.info("Using robust adaptive RK4 method for Rabinovich-Fabrikant system")
    
#     t = 0
#     y = initial_condition
#     results = [y]
#     times = [t]
    
#     dt = sim_time / max_steps  # Initial step size
#     min_dt = 1e-10  # Minimum allowed step size
#     max_dt = sim_time / 100  # Maximum allowed step size
#     tolerance = 1e-6  # Error tolerance
    
#     while t < sim_time and len(results) < max_steps:
#         y_new, error = rk4_step_rf(y, t, dt, params)
        
#         if error > tolerance:
#             # Decrease step size
#             dt = max(0.9 * dt * (tolerance / error)**0.2, min_dt)
#             continue
        
#         # Step successful
#         t += dt
#         y = y_new
#         results.append(y)
#         times.append(t)
        
#         # Increase step size if error is small
#         if error < 0.5 * tolerance:
#             dt = min(1.1 * dt, max_dt)
        
#         # Check for NaN or Inf
#         if np.any(np.isnan(y)) or np.any(np.isinf(y)):
#             logger.warning(f"NaN or Inf values detected at t={t}. Stopping simulation.")
#             break
    
#     logger.info(f"Rabinovich-Fabrikant simulation completed with {len(results)} steps")
#     return np.array(results)