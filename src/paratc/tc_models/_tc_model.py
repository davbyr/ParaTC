import numpy as np
from paratc import _utils, _const, tctools, inflow_models, bg_models
import xarray as xr

class TCModel():

    def check_track(self, track ):
        
        # Make pdelta, just in case
        if 'pdelta' not in track:
            track['pdelta'] = track['penv'] - track['pcen']

        # Get hour and timestep
        if 'timestep' not in track:
            track['timestep'] = self.get_timestep_from_time( track.time )

        # Get translation velocity
        if 'utrans' not in track or 'vtrans' not in track:
            trans_speed, utrans, vtrans = tctools.get_translation_vector( track.lon.values, 
                                                             track.lat.values, 
                                                             track.timestep.values )
            track['utrans'], track['vtrans'] = utrans, vtrans
            track['trans_speed'] = trans_speed

        self.n_time = len(track)

    def make_empty_storm_dataset(self, track, grid_lon, grid_lat):
        
        # Make output grid dataset
        n_time = len(track)

        # Calculate distances from storm center
        dist_cent = np.zeros( (n_time, *grid_lon.shape) )
        for ii in range( n_time ):
            dist_cent[ii] = _utils.haversine( grid_lon, grid_lat, 
                                          track.iloc[ii].lon, 
                                          track.iloc[ii].lat, 
                                          radians=False )
        data = xr.Dataset()
        if 'time' in track:
            data['time'] = (['time'], track.time)
        else:
            data['time'] = (['time'], np.arange(n_time))

        data['lon'] = (['y','x'], grid_lon)
        data['lat'] = (['y','x'], grid_lat)
        data['dist_cent'] = (['time','y','x'], dist_cent)

        return data

    def make_wind_vectors(self, inflow_model = 'wang20', inflow_angle = 0):

        # Expand wind speed into vectors using inflow angle
        n_time = len(self.track)
        wind_u = np.zeros( (n_time, *self.data.lon.shape) )
        wind_v = np.zeros( (n_time, *self.data.lon.shape) )

        for tii in range( n_time ):

            ws_ii = self.data.windspeed[tii].values
            dist_ii = self.data.dist_cent[tii].values
            tr_ii = self.track.iloc[tii]
        
            # Make inflow angle
            if inflow_model == 'wang20':
                inflow_angle = inflow_models.wang20( dist_ii, tr_ii.rmw )
            elif inflow_model == 'constant': 
                # In this case we will just use inflow_angle
                pass
            else:
                raise Exception(f' Inflow angle model not found: {inflow_model}' )

            u_ii, v_ii = tctools.windspeed_to_vector( ws_ii, 
                                                      inflow_angle, 
                                                      self.data.lon, 
                                                      self.data.lat,
                                                      tr_ii.lon, tr_ii.lat )
            wind_u[tii] = u_ii
            wind_v[tii] = v_ii

        self.data['wind_u'] = (['time','y','x'], wind_u)
        self.data['wind_v'] = (['time','y','x'], wind_v)

    def add_background_winds(self, bg_model='constant', 
                             bg_alpha=.55, bg_beta=20):

        wind_u = self.data.wind_u
        wind_v = self.data.wind_v

        for tii in range(self.n_time):

            tr_ii = self.track.iloc[tii]
            
            # Make background flow
            hemisphere = _utils.get_hemisphere( self.track.lat[0] )
            if bg_model == 'constant':
                u_bg, v_bg = bg_models.uniform_flow( tr_ii.utrans, tr_ii.vtrans,
                                                     bg_alpha, bg_beta, hemisphere)
            elif bg_model == 'miyazaki61':
                u_bg, v_bg = bg_models.miyazaki61( dist_cent, tr_ii.utrans, 
                                                   tr_ii.vtrans, tr_ii.rmw )
                u_bg, v_bg = _utils.rotate_vectors( u_bg, v_bg, bg_beta, radians=False)
                u_bg, v_bg = bg_alpha * u_bg, bg_alpha * v_bg
            else:
                raise Exception(f' Background flow model not found: {bg_model}' )
    
            wind_u[tii] += u_bg
            wind_v[tii] += v_bg

        self.make_windspeed_from_uv()

    def make_windspeed_from_uv(self):
        windspeed = np.sqrt( self.data.wind_u**2 + self.data.wind_v**2 )
        self.data['windspeed'] = windspeed
    
    @classmethod
    def get_timestep_from_time(cls, time ):
        hours = [ (timeii - time[0]).total_seconds() / (60**2) for timeii in time]
        hours = np.array( hours )
        timestep = np.zeros_like( hours )
        timestep[1:] = hours[1:] - hours[:-1]
        return timestep