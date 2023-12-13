'''
Module for background field models. These models describe the transient atmospheric flow to which the
tropical cyclone vectors are relative to. Generally, this background flow is scaled, rotated and added
to the rotating cyclone flow, creating asymmetry.

These functions should take the name of the model by citation, where possible. They should also
return a single value or a field of U and V vector components.
'''

import numpy as np
from paratc import _utils

def miyazaki61( dist_cent, utrans, vtrans, rmw ):
    ''' Background winds that decay exponentially with normalized distance from storm center.
    (Miyazaki, 1961). '''
    term = - (np.pi * dist_cent) / (10 * rmw  )
    u_bg = utrans * np.exp( term )
    v_bg = vtrans * np.exp( term )
    return u_bg, v_bg

def MN05(dist_cent, utrans, vtrans, rmw):
    ''' Background winds that decay at a rate reciprocal to the normalized distance from 
     storm center. Mouton, F. & Nordbeck, O. (2005).'''
    decay = np.fmin(1, rmw/dist_cent)
    return utrans*decay, vtrans*decay