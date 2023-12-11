import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
from paratc import _utils, tctools, inflow_models, bg_models

def make_example_data( ):
    '''
    Generate a synthetic tropical cyclone track with 5 points.

    Returns:
        track: pandas dataframe with track information
        lon2: Grid longitudes for using TCModels
        lat2: Grid latitudes for using TCModels
    '''
    
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
    lon2, lat2 = np.meshgrid( np.arange( -89, -79, 0.1 ), 
                              np.arange( 21, 27, 0.1 ) )
    return track, lon2, lat2