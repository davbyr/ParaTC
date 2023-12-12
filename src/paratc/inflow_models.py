'''
Models of inflow angle.
'''

import numpy as np

def nws( dist_cent, rmw ):
    '''
    Inflow angle determined using a piecewise function of storm center distance,
    as recommended by the National Weather Service (Bretschnedier, 1972)

    Args:
        dist_cent (float, np.ndarray): Distance from storm center (km).
        rmw (float): radius of maximum winds (km)

    Returns:
        inflow_angle( float, np.ndarray): Angles in degrees, in same format as dist_cent.
    '''
    inflow_angle = np.zeros_like(dist_cent)
    r_scaled = dist_cent / rmw

    # Close zone
    zone_angle = 10*(r_scaled + 1)
    zone_idx = dist_cent <= rmw
    inflow_angle[ zone_idx ] = zone_angle[zone_idx]

    # Middle zone
    zone_angle = 25*r_scaled - 5
    zone_idx = np.logical_and( dist_cent >= rmw, 
                               dist_cent <= 1.2*rmw )
    inflow_angle[ zone_idx ] = zone_angle[zone_idx]

    # Far zone
    inflow_angle[ dist_cent >= 1.2*rmw ] = 25
    return inflow_angle