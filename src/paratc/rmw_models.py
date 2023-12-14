import numpy as np

def calculate_rmw( track, rmw_model = 'vickery08' ):
    ''' Calculate rmw from track dataframe

    Args:
        track (pd.Dataframe): Track dataframe with columns necessary for chosen rmw models
        rmw_model (str): rmw model to use from rmw_models module.

    Returns:
        rmw (np.ndarray): Radius of maximum winds in km
    '''

    if rmw_model == 'VW08':
        rmw = vickery08( track.pdelta, track.lat )
    elif rmw_model == 'climada':
        rmw = rmw_climada( track.pcen )

    return rmw

def VW08( pdelta, lat ):
    ''' Statistical model of rmw, taken from (Vickery & Wadhera, 2008).
    This is for all hurricanes analysed. Returns rmw in km for pdelta in mb.'''
    exponent = 3.015 - 6.291e-5*pdelta**2 + 0.0337*lat
    return np.exp(exponent)

def rmw_climada(pcen):
    """
    Statistical model for rmw (km), as in the Climada package.
    """
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]
    rmw = np.zeros_like
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        rmw[msk] += rmw_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - pcen[msk])
                                           - slope_1 * np.fmax(0, pcen[msk] - pres_l_i)))

    return np.where(rmw <= 0, np.nan, rmw)