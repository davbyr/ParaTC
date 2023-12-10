import numpy as np
from paratc import _const

def make_stress( U, V, method='andreas', max_cd = 0.0035 ):

    s = np.sqrt(U**2 + V**2)
    
    if method == 'andreas':
        Cd = cd_andreas(s)
    elif method == 'large_pond':
        Cd = cd_large_pond(s)
    elif method == 'garratt':
        Cd = cd_garratt( s )
    elif method == 'peng':
        Cd = cd_peng( s )

    Cd = np.clip(Cd, 0, max_cd)
    tau = s**2 * _const.rho * Cd
    tau_u = U * s * _const.rho * Cd
    tau_v = V * s * _const.rho * Cd
    return tau, tau_u, tau_v

def cd_garratt( s ):
    ''' Wind stress vectors according to Garratt '''
    Cd = 0.001*(0.75+0.067*s)
    return Cd
    
def cd_large_pond( s ):
    ''' Wind stress vectors according to Large & Pond '''
    Cd = 0.001*(0.49+0.065*s)
    return Cd

def cd_andreas( s ):
    ''' Wind stress vectors according to Andreas '''
    ws1 = s - 9.271
    ws2 = np.sqrt( 0.12*ws1**2 + 0.181 )
    ws = .239 + .0433*( ws1 + ws2 )
    Cd = (ws / s)**2
    return Cd

def cd_peng( s ):
    a = 2.15e-6
    c = 2.797e-3
    Cd = -a*(s - 33)**2 + c
    return Cd