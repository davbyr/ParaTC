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
    ''' Wind stress drag coefficient according to Garratt, (1977) '''
    cd = 0.001*(0.75+0.067*windspeed)
    return cd
    
def cd_large_pond82( windspeed ):
    ''' Wind stress drag coefficient according to Large & Pond (1982). 
    This is not quite what is presented in the paper. We also hold cd = 1.14e-3
    for windspeeds < 4 and cd = 2.18e-3 for windspeed >= 26.'''
    convert_to_float = False
    if not hasattr(windspeed, '__len__'):
        convert_to_float = True
        windspeed = np.array( [windspeed] )

    cd = np.zeros_like(windspeed)
    cd[ windspeed <= 10 ] = 1.14e-3
    gt_idx = np.logical_and( windspeed > 10, windspeed <=26 )
    cd[ gt_idx ] = (0.49+0.065*windspeed[gt_idx]) * 0.001
    cd[ windspeed > 26 ] = (0.49+0.065*26) * 0.001

    if convert_to_float:
        return cd[0]
    else:
        return cd

def cd_andreas12( windspeed ):
    ''' Wind stress drag coefficient according to Andreas et al., (2012) '''
    return 3.4e-3 * (1 - 4.17 / windspeed )**2

def cd_peng_li15( windspeed ):
    ''' Wind stress drag coefficient according to Peng & Li (2015) '''
    a = 2.15e-6
    c = 2.797e-3
    cd = -a*(windspeed - 33)**2 + c
    return cd