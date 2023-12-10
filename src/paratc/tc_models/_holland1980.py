import numpy as np
from paratc import _utils, _const, tctools
from paratc.tc_models import TCModel

class Holland1980( TCModel ):

    def __init__(self, track, grid_lon, grid_lat,
                 B_model = 'powell05'):
    
        super().check_track(track)
        data = super().make_empty_storm_dataset( track, 
                                                 grid_lon, grid_lat )
    
        # Make B if it isn't in track dataframe
        if 'B' not in track:
            if B_model == 'powell05':
                track['B'] = self.B_powell05( track.rmw, track.lat )
            elif B_model == 'wr04':
                track['B'] = self.B_wr04( track.rmw, track.vmax, track.lat )
            elif B_model == 'vickery00':
                track['B'] == self.B_vickery00( track.pdelta, track.rmw )
            else:
                raise Exception(f'B_model unknown: {B_model}')

        track['B'] = np.clip( track['B'], 1, 2.5)

        # Generate pressure and gradient wind speeds
        n_time = len(track)
        pressure = np.zeros( (n_time, *grid_lon.shape) )
        wind_g = np.zeros( (n_time, *grid_lon.shape) )
    
        for tii in range( n_time ):
            tr_ii = track.iloc[tii]
    
            # Make pressure and gradient wind
            dist_ii = data.dist_cent[tii].values
            pressure[tii] = self.pressure_equation( dist_ii, tr_ii.rmw, 
                                                    tr_ii.B, tr_ii.penv,
                                                    tr_ii.pcen, tr_ii.lat )
            wind_g[tii] = self.gradient_wind_equation( dist_ii, tr_ii.pdelta, 
                                                       tr_ii.B, tr_ii.rmw, tr_ii.lat ) 

        data['pressure'] = (['time','y','x'], pressure)
        data['windspeed'] = (['time','y','x'], wind_g)

        self.data = data
        self.track = track
            
    @classmethod
    def pressure_equation(cls, dist_cent, rmw, B,
                       penv, pcen, lat ):
        """The Holland 1980 Pressure profile"""

        # Convert units
        penv = _const.mb_to_pa * penv
        pcen = _const.mb_to_pa * pcen
        dist_cent = _const.km_to_m * dist_cent
        rmw = _const.km_to_m * rmw
        
        too_close = dist_cent < 1
        
        rmw_norm = (rmw / dist_cent)**B
        pressure = pcen + (penv - pcen)*np.exp( -rmw_norm )
        pressure[too_close] = pcen
        return pressure

    @classmethod
    def gradient_wind_equation(cls, dist_cent, pdelta, B, 
                            rmw, lat ):

        # Convert units
        pdelta = _const.mb_to_pa * pdelta
        dist_cent = _const.km_to_m * dist_cent
        rmw = _const.km_to_m * rmw
        
        # Calculate the various terms one by one
        f = _utils.calculate_coriolis( lat )
        rf = dist_cent * f / 2
        rf2 = rf**2
        rmax_norm = (rmw / dist_cent)**B
    
        inside_sqrt = rmax_norm * (B / _const.rho) * pdelta * np.exp( -rmax_norm) + rf2
        Vg = np.sqrt( inside_sqrt ) - rf
        Vg[ dist_cent < 0.1 ] = 0
        return Vg

    @classmethod
    def B_powell05(cls, rmw, lat ):
        return 1.881 - 0.00557*rmw - 0.01295*lat

    @classmethod
    def B_wr04(cls, rmw, vmax, lat ):
        return 1.0036 + 0.0173*vmax + 0.0313*np.log(rmw) + 0.0087*lat

    @classmethod
    def B_vickery00(cls, pdelta, rmw ):
        return 1.38 - 0.00184*pdelta + 0.00309*rmw

    @classmethod 
    def B_holland80(cls, pdelta, ):
        # the factor 100 is from conversion between mbar and pascal
        hol_b = gradient_winds**2 * np.exp(1) * _const.rho / np.fmax(np.spacing(1), pdelta)
        return np.clip(hol_b, 1, 2.5)
    
