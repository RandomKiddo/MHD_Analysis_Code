import numpy as np
import warnings


G = 6.6743e-8
PARAMS = {
    'M': 2.7839e33,
    'R': 1.2e6, 
    'a': 3e9,
    'alpha': 6.28318530718,
    'B': 813473751950763.0*np.sqrt(4*np.pi),
    'rho': 316094286571.96594
}

C1 = (8 * np.pi * PARAMS['rho'] * PARAMS['a']**2) / (PARAMS['B']**2)
C2 = (-G * PARAMS['M']) / (PARAMS['R'] * PARAMS['a']**2)
C3 = (PARAMS['alpha']**2 * PARAMS['R']**2) / (2* PARAMS['a']**2)


def override_params(new_params: dict) -> bool:
    global PARAMS, C1, C2, C3

    required_keys = ['M', 'R', 'a', 'alpha', 'B', 'rho']
    if all(_ in new_params for _ in required_keys):
        PARAMS = new_params

        C1 = (8 * np.pi * PARAMS['rho'] * PARAMS['a']**2) / (PARAMS['B']**2)
        C2 = (-G * PARAMS['M']) / (PARAMS['R'] * PARAMS['a']**2)
        C3 = (PARAMS['alpha']**2 * PARAMS['R']**2) / (2* PARAMS['a']**2)
    else:
        warnings.warn('Unable to override parameters due to missing keys. Defaulting to hard-coded parameters.')
        return False


def f(y: float) -> float:
    if y <= 0:
        return np.inf
    
    exp1 = np.exp(C2 * (1 - y))
    exp2 = np.exp(C3 * (y**-2 - y))

    return y**6 - C1 * exp1 * exp2


def df(y: float) -> float:
    if y <= 0:
        return np.inf
    
    term1 = 6 * y**5
    
    term2 = C1 * np.exp(C2 * (1 - y)) * np.exp(C3 * (y**-2 - y))
    term2 *= (-C2 + C3 * (-1 - 2/(y**3)))

    return term1 - term2

