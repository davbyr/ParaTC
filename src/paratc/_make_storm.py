import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
from paratc import _utils, tctools, inflow_models, bg_models

def make_sample_data( ):
    
    track = pd.DataFrame()
    track['lon'] = [-81.4, -81.9, -82.4, -82.9, -83.3]
    track['lat'] = [23.8, 23.56, 23.4, 23.36, 23.4]
    track['pcen'] = [990, 970, 940, 980, 1005]
    track['penv'] = [1013, 1013, 1013, 1013, 1013]
    track['rmw'] = [50, 50, 75, 100, 0]
    track['time'] = pd.date_range( datetime( 2016, 8, 28, 18 ),
                                   datetime( 2016, 8, 29, 6 ),
                                   freq = '3H' )

    grid = xr.Dataset()
    lon2, lat2 = np.meshgrid( np.arange( -84, -79, 0.1 ), 
                              np.arange( 21, 25, 0.1 ) )
    return track, lon2, lat2

def make_storm( wind_model, 
                grid_lon, 
                grid_lat,
                bl_model = 'swrf', 
                bl_swrf = 0.9,
                wind_scaling = 1,
                inflow_model = 'wang20', 
                inflow_angle = 0,
                bg_model = 'constant', 
                bg_alpha = 0.55, 
                bg_beta = 20 ):

    track = wind_model.track
    
    # Initialize output array for looping
    n_time = len(track)
    pressure = np.zeros( (n_time, *grid_lon.shape) )
    wind_u = np.zeros( (n_time, *grid_lon.shape) )
    wind_v = np.zeros( (n_time, *grid_lon.shape) )

    for tii in range( n_time ):
        tr_ii = track.iloc[tii]
        dist_cent = tctools.distance_from_storm_center( grid_lon, grid_lat,
                                                        tr_ii.lon, tr_ii.lat )

        # Make pressure and gradient wind
        pressure[tii] = wind_model.calculate_pressure( dist_cent, tr_ii.rmw, 
                                                       tr_ii.B, tr_ii.penv,
                                                       tr_ii.pcen, tr_ii.lat )
        wind_g = wind_model.calculate_gradient_wind( dist_cent, tr_ii.pdelta, 
                                                     tr_ii.B, tr_ii.rmw, tr_ii.lat ) 
        
        # Make inflow angle
        if inflow_model == 'wang20':
            inflow_angle = inflow_models.wang20( dist_cent, tr_ii.rmw )
        elif inflow_model is None:
            inflow_angle = 0
        elif inflow_model is 'constant':
            pass
        else:
            raise Exception(f' Inflow angle model not found: {inflow_model}' )

        # Scale cyclone winds to 10m level
        if bl_model == 'swrf':
            wind_g = wind_g * bl_swrf
        else:
            raise Exception(f' Boundary Layer model unknown: {bl_model}')

        # Other scalings (e.g. 1-min to 10-min)
        wind_g = wind_g * wind_scaling

        # Expand wind speed into vectors using inflow angle
        u_ii, v_ii = tctools.windspeed_to_vector( wind_g, inflow_angle, 
                                                  grid_lon, grid_lat,
                                                  tr_ii.lon, tr_ii.lat )

        # Make background flow
        hemisphere = _utils.get_hemisphere( track.lat[0] )
        if bg_model == 'constant':
            u_bg, v_bg = tr_ii.utrans, tr_ii.vtrans
            
            u_bg, v_bg = bg_models.uniform_flow( tr_ii.utrans, tr_ii.vtrans,
                                                 bg_alpha, bg_beta, hemisphere)
        elif bg_model == 'miyazaki61':
            u_bg, v_bg = bg_models.miyazaki61( dist_cent, tr_ii.utrans, 
                                               tr_ii.vtrans, tr_ii.rmw )
            u_bg, v_bg = _utils.rotate_vectors( u_bg, v_bg, bg_beta, radians=False)
            u_bg, v_bg = bg_alpha * u_bg, bg_alpha * v_bg
        elif bg_model == None:
            u_bg, v_bg = 0, 0
        else:
            raise Exception(f' Background flow model not found: {bg_model}' )

        wind_u[tii] = u_ii + bg_alpha * u_bg
        wind_v[tii] = v_ii + bg_alpha * v_bg
    
    return pressure, wind_u, wind_v

def make_stress( wind_u, wind_v ):


    
    return
    