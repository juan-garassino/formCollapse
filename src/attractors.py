import numpy as np
from scipy.integrate import solve_ivp

def lorenz_system(X, t, params):
    x, y, z = X
    sigma = params.get('sigma', 10)
    beta = params.get('beta', 8/3)
    rho = params.get('rho', 28)
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def aizawa_system(X, t, params):
    x, y, z = X
    a = params.get('a', 0.95)
    b = params.get('b', 0.7)
    c = params.get('c', 0.6)
    d = params.get('d', 3.5)
    e = params.get('e', 0.25)
    f = params.get('f', 0.1)
    return [
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
    ]

def rabinovich_fabrikant_system(X, t, params):
    x, y, z = X
    alpha = params.get('alpha', 0.14)
    gamma = params.get('gamma', 0.1)
    return [
        y * (z - 1 + x**2) + gamma * x,
        x * (3 * z + 1 - x**2) + gamma * y,
        -2 * z * (alpha + x * y)
    ]

def three_scroll_system(X, t, params):
    x, y, z = X
    a = params.get('a', 40)
    b = params.get('b', 55)
    c = params.get('c', 1.833)
    return [
        a * (y - x),
        (b - a) * x - x * z + b * y,
        x * y - c * z
    ]

def simulate_system(system_func, X0, t, params):
    """
    Simulates the provided attractor system using solve_ivp.
    
    Args:
        system_func (function): The system function (e.g., lorenz_system).
        X0 (list or array): Initial condition [x, y, z].
        t (array): Array of time points for the simulation.
        params (dict): Parameters for the system.
    
    Returns:
        np.array: Simulation result with shape (len(t), 3) where 3 is for [x, y, z].
    """
    def func(t, y):
        return system_func(y, t, params)
    
    solution = solve_ivp(func, [t[0], t[-1]], X0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-9)
    
    if not solution.success:
        print(f"Warning: Integration failed. Message: {solution.message}")
    
    result = solution.y.T
    
    print(f"Simulation result shape: {result.shape}")
    
    return result