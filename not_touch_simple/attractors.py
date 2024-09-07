import numpy as np
from scipy.integrate import odeint

def lorenz_system(X0, t, params={'sigma': 10, 'beta': 8/3, 'rho': 28}):
    """
    Simulates the Lorenz system, which models atmospheric convection. It is a
    set of three coupled, first-order, nonlinear differential equations.

    Args:
        X0 (list or array): Initial condition [x, y, z] at time t=0.
        t (array): Array of time points for the simulation.
        params (dict): Parameters of the Lorenz system, containing:
            - sigma (float): Prandtl number, which controls the rate of convection.
            - beta (float): Geometric factor, related to the physical aspect ratio of the convective system.
            - rho (float): Rayleigh number, a measure of the temperature difference driving the convection.
    
    Returns:
        list: Derivatives [dx/dt, dy/dt, dz/dt] at each time step.
    """
    # Unpack initial conditions
    x, y, z = X0

    # Unpack parameters from the dictionary (with defaults)
    sigma = params.get('sigma', 10)
    beta = params.get('beta', 8/3)
    rho = params.get('rho', 28)

    # Lorenz equations
    dx_dt = sigma * (y - x)  # Convection term
    dy_dt = x * (rho - z) - y  # Horizontal temperature variation
    dz_dt = x * y - beta * z  # Vertical temperature variation

    # Return the system of differential equations
    return [dx_dt, dy_dt, dz_dt]

def aizawa_system(X0, t, params={'a': 0.95, 'b': 0.7, 'c': 0.6, 'd': 3.5, 'e': 0.25, 'f': 0.1}):
    """
    Simulates the Aizawa attractor system, a six-parameter chaotic attractor.
    This system is notable for its complex and aesthetically interesting chaotic behavior.

    Args:
        X0 (list or array): Initial condition [x, y, z] at time t=0.
        t (array): Array of time points for the simulation.
        params (dict): Parameters of the Aizawa system, containing:
            - a (float): Governs the growth rate of z.
            - b (float): Determines the scaling of x and y terms.
            - c (float): Governs linear growth of z.
            - d (float): Controls the oscillation in the x and y directions.
            - e (float): Nonlinear damping factor.
            - f (float): Scaling for cubic terms in the x-direction.
    
    Returns:
        list: Derivatives [dx/dt, dy/dt, dz/dt] at each time step.
    """
    # Unpack initial conditions
    x, y, z = X0

    # Unpack parameters from the dictionary
    a = params.get('a', 0.95)
    b = params.get('b', 0.7)
    c = params.get('c', 0.6)
    d = params.get('d', 3.5)
    e = params.get('e', 0.25)
    f = params.get('f', 0.1)

    # Aizawa system equations
    dx_dt = (z - b) * x - d * y  # Oscillation and interaction of x and z
    dy_dt = d * x + (z - b) * y  # Oscillation and interaction of y and z
    dz_dt = c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3  # Governs z's growth

    # Return the system of differential equations
    return [dx_dt, dy_dt, dz_dt]

def rabinovich_fabrikant_system(X0, t, params={'alpha': 0.14, 'gamma': 0.1}):
    """
    Simulates the Rabinovich-Fabrikant system, a model of plasma oscillations,
    which exhibits chaotic behavior depending on the parameters alpha and gamma.

    Args:
        X0 (list or array): Initial condition [x, y, z] at time t=0.
        t (array): Array of time points for the simulation.
        params (dict): Parameters of the Rabinovich-Fabrikant system, containing:
            - alpha (float): Controls the chaotic nature and strength of the interaction between variables.
            - gamma (float): Damping factor, affecting the system's stability.
    
    Returns:
        list: Derivatives [dx/dt, dy/dt, dz/dt] at each time step.
    """
    # Unpack initial conditions
    x, y, z = X0

    # Unpack parameters from the dictionary
    alpha = params.get('alpha', 0.14)
    gamma = params.get('gamma', 0.1)

    # Rabinovich-Fabrikant system equations
    dx_dt = y * (z - 1 + x**2) + gamma * x  # x's equation
    dy_dt = x * (3 * z + 1 - x**2) + gamma * y  # y's equation
    dz_dt = -2 * z * (alpha + x * y)  # z's equation, highly nonlinear

    # Return the system of differential equations
    return [dx_dt, dy_dt, dz_dt]

def three_scroll_system(X0, t, params={'a': 40, 'b': 55, 'c': 1.833}):
    """
    Simulates the Three-Scroll Unified Chaotic System, a well-known chaotic system
    that generates intricate scroll-like structures in 3D space.

    Args:
        X0 (list or array): Initial condition [x, y, z] at time t=0.
        t (array): Array of time points for the simulation.
        params (dict): Parameters of the Three-Scroll system, containing:
            - a (float): Scaling factor for the rate of change in x and y.
            - b (float): Influences the rate of change in y.
            - c (float): Governs the scroll generation in z.
    
    Returns:
        list: Derivatives [dx/dt, dy/dt, dz/dt] at each time step.
    """
    # Unpack initial conditions
    x, y, z = X0

    # Unpack parameters from the dictionary
    a = params.get('a', 40)
    b = params.get('b', 55)
    c = params.get('c', 1.833)

    # Three-Scroll system equations
    dx_dt = a * (y - x)  # Rate of change in x, creating scroll-like behavior
    dy_dt = (b - a) * x - x * z + b * y  # Rate of change in y, interaction with z
    dz_dt = x * y - c * z  # Rate of change in z, generating the three-scroll effect

    # Return the system of differential equations
    return [dx_dt, dy_dt, dz_dt]

def simulate_system(system_func, X0, t, params):
    """
    Simulates the provided attractor system.
    
    Args:
        system_func (function): The system function (e.g., lorenz_system).
        X0 (list or array): Initial condition [x, y, z].
        t (array): Array of time points for the simulation.
        params (dict): Parameters for the system.
    
    Returns:
        np.array: Simulation result with shape (len(t), 3) where 3 is for [x, y, z].
    """
    # Simulate the system using scipy's odeint
    result = odeint(system_func, X0, t, args=(params,))
    
    # Check the shape of the result (it should be (len(t), 3))
    print(f"Simulation result shape: {result.shape}")
    
    # Return the simulation data
    return result