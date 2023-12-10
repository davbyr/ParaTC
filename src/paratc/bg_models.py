import numpy as np
from paratc import _utils

def miyazaki61( dist_cent, utrans, vtrans, rmw ):
    term = - (np.pi * dist_cent) / (10 * rmw  )
    um = utrans * np.exp( term )
    vm = vtrans * np.exp( term )
    return um, vm

def MN05(dist_cent, utrans, vtrans, rmw):
    ''' Mouton, F. & Nordbeck, O. (2005) '''
    return

def uniform_flow( utrans, vtrans, 
                  alpha = .55 , beta = 20,
                  hemisphere = 'N' ):

    utrans_scaled = utrans * alpha
    vtrans_scaled = vtrans * alpha

    if hemisphere == 'N':
        u_rot, v_rot = _utils.rotate_vectors( utrans_scaled, 
                                              vtrans_scaled, beta, 
                                              radians=False )
    else:
        u_rot, v_rot = _utils.rotate_vectors( utrans_scaled, 
                                              vtrans_scaled, -beta, 
                                              radians=False )
    return u_rot, v_rot

def MN05(dist_cent, utrans, vtrans, rmw):
    ''' Mouton, F. & Nordbeck, O. (2005) '''