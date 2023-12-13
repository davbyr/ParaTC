'''
Utility tools for manipulating 2D wind data.
'''

import numpy as np
from paratc import _utils, _const
from shapely.geometry import Polygon

def get_grid_poly( lon, lat ):
    '''
    For a rectangular grid defined by 2D lon, lat arrays, return a 
    Shapely Polygon object representing its bounds.
    '''

    c1 = (lon[0,0], lat[0,0])      # Top left
    c2 = (lon[0,-1],lat[0,-1])     # Top right
    p1 = (lon[-1,0], lat[-1,0])    # Bottom left
    p2 = (lon[-1,-1], lat[-1,-1])  # Bottom right
    pol = Polygon( (c1, c2, p2, p1) )
    return pol

def windspeed_to_vector( wspeed, inflow_angle,
                         lon, lat,
                         track_lon, track_lat):
    ''' Convert an array of windspeeds to U and V vector components by
    applying a storm inflow angle. This function builds arrays of U and V that
    are perpendicular to the storm center, then rotates them by the amount in
    inflow_angle. This rotation is counter clockwise in the northern hemisphere
    and clockwise in the southern hemisphere.

    Args:
        wspeed (np.ndarray): Array of windspeeds
        inflow_angle (np.ndarray): Array of inflow angles.
        lon (np.ndarray): Array of grid longitudes at which to calculate U, V
        lat (np.ndarray): Array of grid latitudes at which to calculate U, V
        track_lon (np.ndarray): Array of track longitudes
        track_lat (np.ndarray): Array of track latitudes

    Returns:
        u, v : U and V components of wind at grid_lon and grid_lat.
    '''

    # Get the wind vectors perpendicular to storm center
    u, v = get_storm_perpendicular( lon, lat, track_lon, track_lat )

    # Rotate perpendicular vectors by inflow angle
    if track_lat > 0:
        u, v = _utils.rotate_vectors( u, v, inflow_angle, radians=False )
    else:
        u, v = _utils.rotate_vectors( u, v, -inflow_angle, radians=False )

    # Scale by windspeed
    u = wspeed*u
    v = wspeed*v
    
    return u, v

def get_storm_perpendicular( lon, lat, track_lon, track_lat ):
    ''' For a set of longitude and latitude grid points, get the vector 
    that is perpendicular to the vector connecting the point to the storm center.
    In the northern hemisphere, the result is a circular windfield rotating
    counter-clockwise (clockwise in the southern hemisphere

    Args:
        lon (np.ndarray): Array of grid longitudes at which to calculate U, V
        lat (np.ndarray): Array of grid latitudes at which to calculate U, V
        track_lon (np.ndarray): Array of track longitudes
        track_lat (np.ndarray): Array of track latitudes
    Returns:
        u, v: U and V components of wind perpendicular to storm center.
            The magnitude of the vectors will be 1.
    '''

    # Get centre-parallel vectors
    diff_lon = lon - track_lon
    diff_lat = lat - track_lat

    # Get vector magnitude and normalize
    magnitude = np.sqrt( diff_lon**2 + diff_lat**2 )
    diff_lon = diff_lon / magnitude
    diff_lat = diff_lat / magnitude
    
    # Rotate to be perpendicular
    if track_lat > 0:
        diff_lon, diff_lat = _utils.rotate_vectors( diff_lon, diff_lat, 
                                             90, radians=False )
    else:
        diff_lon, diff_lat = _utils.rotate_vectors( diff_lon, diff_lat, 
                                             -90, radians=False )
    return diff_lon, diff_lat

def distance_from_storm_center( grid_lon, grid_lat, 
                                track_lon, track_lat ):
    ''' Transform an array of grid longitudes and latitudes into distances from
    storm center. Uses a haversine function.
    
    Args:
        grid_lon (np.ndarray): Grid longitudes at which to calculate distance
        grid_lat (np.ndarray): Grid latitudes at which to calculate distance
        track_lon (float): A single track longitude 
        track_lat (float): A single track latitude

    Returns:
        np.ndarray of distances or a float distance in km.
    
    '''
    return _utils.haversine( grid_lon, grid_lat, track_lon, track_lat, radians=False )