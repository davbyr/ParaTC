import numpy as np
from paratc import _utils, _const
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from datetime import datetime, timedelta

def filter_bad_rmw( track_list ):

    output_list = []

    for ii, tr in enumerate(track_list):
        zero_sum = np.sum( tr.radius_max_wind.values != 0 )
        nan_sum = np.sum( ~np.isnan( tr.radius_max_wind.values ) )
        if zero_sum > 0 and nan_sum > 0:
            output_list.append( tr )
    return output_list

def interpolate_to_timestep( track, new_timestep, **kwargs):
    ''' Interpolate a track dataframe to a new timestep in hours '''

    time_0 = track.time.values
    new_time = pd.date_range( time_0[0], time_0[-1], freq=f'{new_timestep}H')

    # Convert to xarray for easier interpolation
    track = track.to_xarray().swap_dims({'index':'time'})
    track = track.interp( time = new_time, **kwargs )
    track = track.to_dataframe().reset_index()
    track['timestep'] = new_timestep
    return track.drop(columns='index')
    

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
    utrans[trans_speed == 0] = 0
    vtrans[trans_speed == 0] = 0
    
    return trans_speed, utrans, vtrans

def filter_tracks_by_column( track_list, col_name = 'vmax',
                             col_min = 33, col_max = np.inf):
    ''' Filter a list of track dataframes by a range of values in a specified
    column. By default, this will filter all tracks that never reach 64 m/s.
    
    Args:
        track_list (list): List of dataframes containing track info.
                           Must contain vmax column.
        col_name (str): Name of column to filter by (default = 'vmax')
        col_min (float): Minimum value of variable to search
        col_max (float): Maximum value of variable to search

    Returns:
        New filtered list of track dataframes.
    '''
    keep_idx = []
    for ii, tr in enumerate(track_list):
        var_over = tr[col_name].values > col_min
        var_under = tr[col_name].values < col_max
        if np.sum( np.logical_and( var_over, var_under ) ) > 0:
            keep_idx.append(ii)
    return keep_idx

def distance_track_to_poly( track_list, pol ):
    ''' Uses Shapely to check minimum proximity of a storm track to a polygon.
        This does not calculate geographical distances, result will be in 
        degrees. This function is used for determining if a track passes through
        a polygon.'''
    linestrings = track_to_linestring( track_list )
    return pol.distance(linestrings)

def clip_track_to_poly( track, poly, max_dist = 1, round_days=True ):
    '''
    Takes a track dataframe and clips it to a shapely polygon.

    Args:
        track (pd.dataframe): Pandas dataframe track with lon, lat columns
        poly (shapely Polygon): Poly to clip to in same crs
        max_dist (float): Maximum distance around poly to clip (degrees)
        round_days (bool): If true, round resulting poly to day start
    '''
    t_points = list(zip( track.lon, track.lat) )
    points = [Point(tc) for tc in t_points]
    dist = np.array( [poly.distance(pt) for pt in points] )
    keep_idx = np.where(dist <= max_dist)[0]
    track_clipped = track.iloc[np.min(keep_idx):np.max(keep_idx)]

    if round_days: 
        date0 = datetime(*pd.to_datetime(track_clipped.time.values[0]).timetuple()[:3])
        date1 = datetime(*pd.to_datetime(track_clipped.time.values[-1]).timetuple()[:3])
        date1 = date1 + timedelta(days=1)
        didx = np.logical_and( track.time >= date0, track.time <= date1 )
        track_clipped = track.iloc[ np.where(didx)[0]]

    return track_clipped

def track_to_linestring( track ):
    ''' Converts a track (or list of) dataframes into a list of shapely (lon,lat) LineStrings '''
    if type(track) is not list:
        track = [track]
    ls_list =[ LineString(list(zip( tr.lon.values, tr.lat.values) ) )  for tr in track ]

    if len(track) > 1:
        return ls_list
    else:
        return ls_list[0]

def subset_tracks_in_poly( track_list, pol, buffer = 0):
    ''' Identifies tracks (dataframe) in a list that pass through a specified
    polygon, with some added buffer. This is not an exact procedure and geographical distances
    are not used for the buffer. Instead, distance in degrees is used.
    
    Args:
        track_list (list): List of track pandas dataframes with lon, lat columns
        pol (shapely.geometry.Polygon): Polygon to compare
        buffer (float): Buffer around polygon to allow (in degrees)
    returns
        Indices of filtered tracks.
    '''
    distances = distance_track_to_poly( track_list, pol )
    keep_idx = np.where( distances <= buffer )[0]
    return keep_idx

def climada_to_dataframe( track, convert_units = True ):
    ''' Converts a climada track xarray dataset into an appropriate dataframe for ParaTC'''
    df_track = track.rename({'radius_max_wind':'rmw', 
                             'central_pressure':'pcen',
                             'environmental_pressure':'penv',
                             'max_sustained_wind':'vmax'})
    df_track = df_track.to_dataframe().reset_index()
    if convert_units:
        df_track['rmw'] = df_track['rmw']*1.852
        df_track['vmax'] = df_track['vmax']*0.51444 / 0.9
    return df_track

def subset_tracks_in_year( track_list, year ):
    ''' Subsets tracks into integer year '''
    year_list = [ pd.to_datetime(tr.time[0]).year for tr in track_list ]
    year_list = np.array(year_list)
    keep_idx = np.where(year_list == year)[0]
    return keep_idx