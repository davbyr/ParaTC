import numpy as np

def vickery2008( pdelta, lat ):
    exponent = 3.015 - 6.291e-5*pdelta**2 + 0.0337*lat
    return np.exp(exponent)

def climada(pcen):
    """
    RMW in km
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