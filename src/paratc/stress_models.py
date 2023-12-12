'''
Functions for calculating surface wind stress. This module contains functions for both
the calculate of stress and the crag coefficients C_D.
'''

import numpy as np
from paratc import _const

def quadratic_stress_equation( wind_u = None, wind_v = None, windspeed = None, 
                               cd = 2.5e-3 ):
    '''
    Calculates wind stress as a quadratic function of windspeed, applying a 
    statistical model of the drag coefficient C_d. You must provide either windspeed
    or both of (wind_u, wind_v) to this function.

    Args:
        wind_u (float, np.ndarray): Wind U vector components (ms^-1)
        wind_v (float, np.ndarray): Wind V vector components (ms^-1)
        windspeed (float, np.ndarray): Windspeed (ms^-1)
        cd (float): The drag coefficient cd.
    Returns:
        Windstress (tau) if only windspeed is provided. If wind_u and wind_v are provided,
        also return tau_u and tau_v components of stress
    '''

    if windspeed is None:
        windspeed= np.sqrt(wind_u**2 + wind_v**2)

    tau = windspeed**2 * _const.rho * cd

    if wind_u is not None and wind_v is not None:
        tau_u = wind_u * windspeed * _const.rho * cd
        tau_v = wind_v * windspeed * _const.rho * cd
        return tau, tau_u, tau_v
    else:
        return tau

def cd_garratt77( windspeed ):
    ''' Wind stress vectors according to (Garratt, 1977) '''
    cd = 0.001*(0.75+0.067*windspeed)
    return cd
    
def cd_large_pond( windspeed ):
    ''' Wind stress vectors according to Large & Pond '''
    cd = 0.001*(0.49+0.065*windspeed)
    return cd

def cd_andreas( windspeed ):
    ''' Wind stress vectors according to Andreas '''
    ws1 = windspeed - 9.271
    ws2 = np.sqrt( 0.12*ws1**2 + 0.181 )
    ws = .239 + .0433*( ws1 + ws2 )
    cd = ( ws / windspeed )**2
    return cd

def cd_peng( windspeed ):
    ''' Wind stress vectors according to Peng '''
    a = 2.15e-6
    c = 2.797e-3
    cd = -a*(windspeed - 33)**2 + c
    return cd