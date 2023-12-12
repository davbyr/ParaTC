import numpy as np
from paratc import _utils, _const

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

def get_translation_vector( track_lon, track_lat, track_timestep ):
    ''' Get storm translation vectors from track longitudes, latitudes and
    timestep. 

    Args:
        track_lon (np.ndarray):
        track_lat (np.ndarray):
        track_timestep (np.ndarray):
    '''

    utrans = np.zeros_like(track_lon)
    vtrans = np.zeros_like(track_lat)
    n_steps = len(track_lon)
    
    distances = np.zeros_like(track_lon)
    distances[1:] = [ _utils.haversine( track_lon[ii], track_lat[ii], 
                                        track_lon[ii-1], track_lat[ii-1], radians=False ) 
                                        for ii in range(1,n_steps) ]
    trans_speed = _const.kmh_to_ms * distances / track_timestep

    utrans[1:] = track_lon[1:] - track_lon[:-1]
    vtrans[1:] = track_lat[1:] - track_lat[:-1]
    vec_norm = np.sqrt( utrans**2 + vtrans**2 )
    utrans = trans_speed * utrans / vec_norm
    vtrans = trans_speed * vtrans / vec_norm
    utrans[0] = 0
    vtrans[0] = 0
    
    return trans_speed, utrans, vtrans