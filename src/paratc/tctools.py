import numpy as np
from paratc import _utils, _const

def windspeed_to_vector( wspeed, inflow_angle,
                         lon, lat,
                         track_lon, track_lat):

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
    return _utils.haversine( grid_lon, grid_lat, track_lon, track_lat, radians=False )

def get_translation_vector( track_lon, track_lat, track_timestep ):

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