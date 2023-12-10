import numpy as np
from paratc import _const

def check_cols_in_dataframe( df, cols ):
    c_exists = [ cc in df for cc in cols ]
    print(c_exists)
    return np.sum(c_exists) == len(cols)
        

def get_hemisphere( lat ):
    if lat <=0:
        return 'S'
    else:
        return 'N'

def calculate_coriolis( lat ):
    
    lat_rad = np.radians(lat)
    return 2 * _const.earth_omega * np.sin( lat_rad )

def haversine(lon1, lat1, lon2, lat2, radians=True):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    if not radians:
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def rotate_vectors( u,v, angle, radians=True ):
    ''' Rotate wind vectors counter clockwise '''

    if not radians:
        angle = np.radians(angle)
    
    cosang = np.cos(angle)
    sinang = np.sin(angle)
    new_u = u * cosang - v * sinang
    new_v = u * sinang + v * cosang
    return new_u, new_v