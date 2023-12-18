import numpy as np
from paratc import _utils, _const, tctools
from paratc.tc_models import TCModel

class Holland1980( TCModel ):
    '''
    Class and functions for generating parametric tropical cyclones according to
    (Holland, 1980). Can be used either as a library to access pressure and wind
    functions, or as an instance of TCModel to automatically handle storm 
    generation.

    Initialize a Holland1980 tropical cyclone model. 
        
    Args:
        track (pd.DataFrame): Dataframe whose columns are track parameters. 
            This cyclone model requires at least: 
                'lon' : Longitude in degrees
                'lat' : Latitude in degrees
                'rmw' : Radius of max winds in km
                'pcen' : Central pressure in mb
                'penv' : Environmental pressure in mb
            Other columns may be required depending on B_model used. If rmw is
            missing, it will be generated using a statistical model. If 'B' is a
            columns in track, this will be used.
        grid_lon (np.ndarray): 2D grid longitudes
        grid_lat (np.ndarray): 2D grid latitudes
        B_model (str): The B_model to use. Default is powell05.
        B_min (float): Minimum value for B shape parameter.
        B_max (float): Maximum value for B shape parameter.
        **kwargs: Any extra keyword arguments are passed to TCModel()

    Attributes:
        data (xr.DataSet): Contains generated storm parameters such as windspeed
            and pressure. The grid is generated using the grid_lon and grid_lat
            parameters provided by the user. When first called, this class will
            generate pressure and gradient wind fields. The wind fields are
            unadjusted by background flow and winds are perpendicular to the
            storm center.
        track (pd.Dataframe): Processed track information. User input is stored
            to the instance and some extra columns are calculated (utrans, vtrans
            B, rmw, timestep).
    '''

    def __init__(self, track, grid_lon, grid_lat,
                 B_model = 'powell05',
                 B_min = 1, B_max = 2.5, 
                 **kwargs):
        
        # Check general parameters and make grid dataset
        super().__init__(track, grid_lon, grid_lat, **kwargs)
        data = self.data
        track = self.track
    
        # Make B if it isn't in track dataframe
        if B_model == 'powell05':
            track['B'] = self.B_powell05( track.rmw, track.lat )
        elif B_model == 'wr04':
            track['B'] = self.B_wr04( track.rmw, track.vmax, track.lat )
        elif B_model == 'VW08':
            track['B'] = self.B_VW08( track.rmw, track.lat )
        elif B_model == 'vickery00':
            track['B'] = self.B_vickery00( track.pdelta, track.rmw )
        elif B_model == 'holland80':
            track['B'] = self.B_holland80( track.pdelta, track.vmax )
        elif B_model is None and 'B' not in track:
            raise Exception(' Expected to find B column in track since B_model = None')
        else:
            raise Exception(f'B_model unknown: {B_model}')
        track['B'] = np.clip( track['B'], B_min, B_max )

        # Generate pressure and gradient wind speeds -- add penv to initial pressure
        n_time = len(track)
        pressure = np.zeros( (n_time, *grid_lon.shape) )
        pressure[0] = track.penv[0]
        wind_g = np.zeros( (n_time, *grid_lon.shape) )
    
        for tii in range( 1, n_time ):
            tr_ii = track.iloc[tii]
    
            # Make pressure and gradient wind
            dist_ii = data.dist_cent[tii].values
            pressure[tii] = self.pressure_equation( dist_ii, 
                                                    rmw = tr_ii.rmw, 
                                                    B = tr_ii.B, 
                                                    penv = tr_ii.penv,
                                                    pcen = tr_ii.pcen, 
                                                    lat = tr_ii.lat )
            wind_g[tii] = self.gradient_wind_equation( dist_ii, 
                                                       rmw = tr_ii.rmw, 
                                                       B = tr_ii.B, 
                                                       pdelta = tr_ii.pdelta, 
                                                       lat = tr_ii.lat ) 
        
        data['pressure'] = (['time','y','x'], pressure)
        data['pressure'].attrs = {'long_name':'Surface atmospheric pressure',
                                  'units':'millibar'}
        data['windspeed'] = (['time','y','x'], wind_g)
        data['windspeed'].attrs = {'long_name':'Magnitude of windspeed',
                                   'units':'m/s'}
        data.attrs['tc_model'] = 'Holland1980'
        data.attrs['tc_B_model'] = B_model

        # Expand windspeed into wind vectors with 0 inflow angle
        self.apply_inflow_angle( inflow_model = 'constant', inflow_angle = 0)

    @classmethod
    def pressure_equation(cls, dist_cent, rmw, B,
                          penv, pcen, lat ):
        """Tropical cyclone pressure model taken from (Holland, 1980).
        This models the pressure profile as a hyperbolic function of
        distance from storm center, radius of max winds, a shape parameter
        B, central pressure and latitude. The shape parameter B can be 
        estimated using one of the B_* functions inside this class.

        This function will calculate pressure for a single snapshot of a 
        tropical cyclone track, i.e. all track parameters should be floats.

        Args:
            dist_cent (np.ndarray): An array of distances to the storm center (km).
                This array can be any shape, e.g. a 1D profile or a 2D grid.
            rmw (float): Radius of maximum winds (km)
            B (float): The storm shape parameter (dimensionless). 
            penv (float): Environmental pressure (mb)
            pcen (float): Central pressure (mb)
            lat (float): Latitude (degrees)
    
        Returns:
            Pressure array of the same shape as dist_cent, units are mb.
        """

        # Convert units for the Holland equation
        penv = _const.mb_to_pa * penv
        pcen = _const.mb_to_pa * pcen
        dist_cent = _const.km_to_m * dist_cent
        rmw = _const.km_to_m * rmw

        # Pressure equation
        rmw_norm = (rmw / dist_cent)**B
        pressure = pcen + (penv - pcen)*np.exp( -rmw_norm )

        # Get locations where distances are very close to eye (approaches infinity)
        too_close = dist_cent < 1
        pressure[too_close] = pcen
        return pressure / _const.mb_to_pa
    
    @classmethod
    def gradient_wind_equation(cls, dist_cent, rmw, B, pdelta, lat ):
        """Tropical cyclone gradient wind model taken from (Holland, 1980).
        Calculates 1-min wind speeds at gradient level.

        This function will calculate pressure for a single snapshot of a 
        tropical cyclone track, i.e. all track parameters should be floats.

        Args:
            dist_cent (np.ndarray): An array of distances to the storm center (km).
                This array can be any shape, e.g. a 1D profile or a 2D grid.
            rmw (float): Radius of maximum winds (km)
            B (float): The storm shape parameter (dimensionless). 
            penv (float): Environmental pressure (mb)
            pcen (float): Central pressure (mb)
            lat (float): Latitude (degrees)
    
        Returns:
            Pressure array of the same shape as dist_cent, units are mb.
        """
        # Convert units
        pdelta = _const.mb_to_pa * pdelta
        dist_cent = _const.km_to_m * dist_cent
        rmw = _const.km_to_m * rmw
        
        # Calculate the various terms one by one
        f = _utils.calculate_coriolis( lat )
        rf = .5 * dist_cent * f
        rf2 = rf**2
        rmw_norm = (rmw / dist_cent)**B
    
        inside_sqrt = rmw_norm * (B / _const.rho) * pdelta * np.exp( -rmw_norm) + rf2
        Vg = np.sqrt( np.fmax(0, inside_sqrt) ) - rf
        Vg[ dist_cent < 0.1 ] = 0
        return Vg

    @classmethod
    def B_powell05(cls, rmw, lat ):
        ''' Statistical model of Holland B parameter according to (Powell, 2005).
        Returns B in same form as input arguments.
        
        Args:
            rmw (float): Radius of maximum winds (km)
            lat (float): Latitude (degrees)
        '''
        return 1.881 - 0.00557*rmw - 0.01295*lat

    @classmethod
    def B_wr04(cls, rmw, vmax, lat ):
        ''' Statistical model of Holland B parameter according to 
            (Willoughby & Rahn, 2004). Returns B in same form as input arguments.
        
        Args:
            rmw (float, np.ndarray): Radius of maximum winds (km)
            vmax (float, np.ndarray): Maximum sustained wind at rmax (ms-1)
            lat (float, np.ndarray): Latitude (degrees)
        '''
        B = 1.0036 + 0.0173*vmax + 0.0313*np.log(rmw) + 0.0087*lat
        return B 

    @classmethod
    def B_VW08(cls, rmw, lat ):
        ''' Statistical model of Holland B parameter according to (Vickery & Wadhera, 2008).
        Returns B in same form as input arguments.
        
        Args:
            rmw (float): Radius of maximum winds (km)
            lat (float): Latitude (degrees)
        '''
        f = _utils.calculate_coriolis( lat )
        rmw = rmw * 1000 # Conver to m
        B = 1.833 - 0.326 * np.sqrt( rmw * f )
        return B
                       
    @classmethod
    def B_vickery00(cls, pdelta, rmw ):
        ''' Statistical model of Holland B parameter according to (Vickery, 2000).
        Returns B in same form as input arguments.
        
        Args:
            pdelta (float, np.ndarray): Difference between environmental and central
                pressure (mb).
            rmw (float, np.ndarray): Radius of maximum winds (km)
        '''
        B = 1.38 - 0.00184*pdelta + 0.00309*rmw
        return B

    @classmethod 
    def B_holland80(cls, pdelta, vmax ):
        ''' Statistical model of Holland B parameter according to (Holland, 1980).
        Returns B in same form as input arguments.
        
        Args:
            pdelta (float, np.ndarray): Difference between environmental and central
                pressure (mb).
            vmax (float, np.ndarray): Maximum sustained wind at rmax (ms-1).
        '''
        # Convert units
        pdelta = pdelta * _const.mb_to_pa
        
        B = vmax**2 * np.exp(1) * _const.rho /  pdelta
        return B
    
