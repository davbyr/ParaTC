import numpy as np
from paratc import tctools, inflow_models, bg_models, stress_models
from paratc import _utils, _const, _output_formats
import xarray as xr
import xesmf as xe
import os
import matplotlib.pyplot as plt

class TCModel():

    ''' General tropical cyclone model. This is inherited by tropical cyclone model classes such as
    Holland1980(). This class contains more general functions that can be applied to any storm model
    dataset. This will calculate some variables if they are not already in track dataset:

        pdelta: Difference between central pressure and environmental pressure
        timestep: Timestep in hours between subsequent track steps
        utrans, vtrans: Storm translation vectors in m/s
        rmw: Radius of max winds, generated using a statistical model where not present.

    Additionally, rmw will be checked for missing values (0 or NaN) and have them filled with
    statistical estimates.

    Args:
        track (pd.DataFrame): Pandas Dataframe with columns representing tropical cyclone track.
        grid_lon (np.ndarray): 2D grid longitudes
        grid_lat (np.ndarray): 2D grid latitudes
    '''
    def __init__(self, track, grid_lon, grid_lat ):
        
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

        # Where (if) central pressure > env pressure then set to env pressure
        c_gt_env = track['pcen'] > track['penv']
        track['pcen'] = track['pcen'].where( ~c_gt_env, track['penv'])

        n_time = len(track)

        # Calculate distances from storm center
        dist_cent = np.zeros( (n_time, *grid_lon.shape) )
        for ii in range( n_time ):
            dist_cent[ii] = _utils.haversine( grid_lon, grid_lat, 
                                          track.iloc[ii].lon, 
                                          track.iloc[ii].lat, 
                                          radians=False )

        # Make output grid dataset and save to object
        data = xr.Dataset()
        if 'time' in track:
            data['time'] = (['time'], track.time)
        else:
            data['time'] = (['time'], np.arange(n_time))

        data['lon'] = (['y','x'], grid_lon)
        data['lat'] = (['y','x'], grid_lat)
        data['dist_cent'] = (['time','y','x'], dist_cent)
        data = data.set_coords(['lon','lat','time'])

        # Assign variables to this instance
        self.data = data
        self.n_time = n_time
        self.track = track

    def scale_winds( self, alpha ):
        '''
        Scales windspeeds and wind vectors as alpha * windspeed, alpha * wind_u, alpha * wind_v.
        This is done in place.

        You may want to scale winds to e.g:
            1. Convert from gradient level to surface level winds
            2. Convert from 1-min to 10-min sustained winds.

        Args:
            alpha (float): Scaling parameter
        '''
        data = self.data
        data['windspeed'] = data['windspeed'] * alpha
        

    def make_wind_vectors(self, inflow_model = 'constant', inflow_angle = 0):
        '''
        Make wind U and V components using the combination of windspeed and an
        inflow angle model. These components will be saved to the object's dataset using
        names wind_u and wind_v.

        Args:
            inflow_model (str): Inflow angle model to use. Default is 'constant', which will
                apply a constant rotate to all vectors. Other options:
                    wang20: Apply piecewise function of distance.
            inflow_angle (str): If inflow_angle is 'constant', this is the angle to apply.
        '''

        # Expand wind speed into vectors using inflow angle
        n_time = len(self.track)
        data = self.data
        wind_u = np.zeros( (n_time, *data.lon.shape) )
        wind_v = np.zeros( (n_time, *data.lon.shape) )

        for tii in range( n_time ):

            # Get relevant variables for this iteration
            ws_ii = data.windspeed[tii].values
            dist_ii = data.dist_cent[tii].values
            tr_ii = self.track.iloc[tii]
        
            # Make inflow angle
            if inflow_model == 'wang20':
                inflow_angle = inflow_models.wang20( dist_ii, tr_ii.rmw )
            elif inflow_model == 'constant': 
                # In this case we will just use inflow_angle
                pass
            else:
                raise Exception(f' Inflow angle model not found: {inflow_model}' )

            # Apply inflow angle and save to output array
            u_ii, v_ii = tctools.windspeed_to_vector( ws_ii, inflow_angle, 
                                                      data.lon, data.lat,
                                                      tr_ii.lon, tr_ii.lat )
            wind_u[tii] = u_ii
            wind_v[tii] = v_ii

        # Save data to object's dataset
        data['wind_u'] = (['time','y','x'], wind_u)
        data['wind_v'] = (['time','y','x'], wind_v)
        data.attrs['inflow_model'] = inflow_model
        data.attrs['inflow_angle'] = inflow_angle

    def add_background_winds(self, bg_model='constant', 
                             bg_alpha=.55, bg_beta=20):
        ''' 
        Add background wind field to the cyclone using a specified model, rotation and scaling.
        This step boils down to adding some form of the environmental flow to the model,
        which helps create the necessary asymmetries in the cyclone model. Using this function
        modifies the model in place, making changes to wind_u and wind_v. Windspeed will also
        be recalculated.

        Args
        '''

        # Get data and loop over track timesteps
        data = self.data
        wind_u = data.wind_u
        wind_v = data.wind_v

        for tii in range(self.n_time):

            tr_ii = self.track.iloc[tii]
            
            # Make background flow
            hemisphere = _utils.get_hemisphere( self.track.lat[0] )
            if bg_model == 'constant':
                u_bg, v_bg = tr_ii.utrans, tr_ii.vtrans
            elif bg_model == 'miyazaki61':
                u_bg, v_bg = bg_models.miyazaki61( data.dist_cent[tii].values, 
                                                   tr_ii.utrans, 
                                                   tr_ii.vtrans, tr_ii.rmw )
            elif bg_model == 'MN05':
                u_bg, v_bg = bg_models.MN05( data.dist_cent[tii].values, 
                                             tr_ii.utrans, 
                                             tr_ii.vtrans, tr_ii.rmw )
            else:
                raise Exception(f' Background flow model not found: {bg_model}' )

            if bg_beta != 0:
                u_bg, v_bg = _utils.rotate_vectors( u_bg, v_bg, bg_beta, radians=False)
            wind_u[tii] += bg_alpha * u_bg
            wind_v[tii] += bg_alpha * v_bg

        self.make_windspeed_from_uv()

        # Save attributes
        data.attrs['bg_model'] = bg_model
        data.attrs['bg_alpha'] = bg_alpha
        data.attrs['bg_beta'] = bg_beta

    def make_wind_stress( self, cd_model='large_pond', cd_min = 0, cd_max = 3.5e-3 ):
        '''
        Calculates wind stress as a quadratic function of windspeed, applying a 
        statistical model of the drag coefficient C_d. 

        Windstress vectors will be saved to the TCModel.data dataset as 
        stress_u and stress_v.
    
        Args:
            cd_model (str): The drag coefficient model to use. 
                See cd_* functions in stress_models module.
            cd_min (float): Minimum value for drag coefficient CD
            cd_max (float): Maximum value for drag coefficient CD
        '''

        data = self.data
        
        # Calculate CD according to model and clip to min, max bounds
        if cd_model == 'andreas':
            cd = stress_models.cd_andreas( data.windspeed )
        elif cd_model == 'large_pond':
            cd = stress_models.cd_large_pond( data.windspeed )
        elif cd_model == 'garratt':
            cd = stress_models.cd_garratt( data.windspeed )
        elif cd_model == 'peng':
            cd = stress_models.cd_peng( data.windspeed )
    
        cd = np.clip(cd, cd_min, cd_max)

        # Calculate wind stress vectors
        tau, tau_u, tau_v = stress_models.quadratic_stress_equation( data.wind_u, 
                                                                     data.wind_v,
                                                                     cd = cd)

        # Save to dataset
        data['stress_u'] = (['time','y','x'], tau_u.values)
        data['stress_v'] = (['time','y','x'], tau_v.values)
        data.attrs['cd_model'] = cd_model
        data.attrs['cd_min'] = cd_min
        data.attrs['cd_max'] = cd_max
        

    def make_windspeed_from_uv(self):
        ''' Uses pythagoras to recalculate windspeed from U and V vectors within this object '''
        windspeed = np.sqrt( self.data.wind_u**2 + self.data.wind_v**2 )
        self.data['windspeed'] = windspeed
    
    @classmethod
    def get_timestep_from_time(cls, time ):
        hours = [ (timeii - time[0]).total_seconds() / (60**2) for timeii in time]
        hours = np.array( hours )
        timestep = np.zeros_like( hours )
        timestep[1:] = hours[1:] - hours[:-1]
        return timestep

    def interpolate_to_arakawa(self, lon_u, lat_u, lon_v, lat_v):
        ''' Interpolate the current tropical cyclone onto an Arakawa rho, U, V grid.
        This function uses XESMF to interpolate the dataset in space.
        '''

        data = self.data

        # Find lists of variables present in dataset
        u_vars = [ vname for vname in ['stress_u','wind_u'] if vname in data ]
        v_vars = [ vname for vname in ['stress_v','wind_v'] if vname in data ]
    
        # Regrid stresses onto U using XESMF
        ds_u = xr.Dataset( coords = dict( lon = (['y','x'], lon_u),
                                          lat = (['y','x'], lat_u) ) )
        regridder = xe.Regridder(data, ds_u, "bilinear")
        ds_u = regridder( data[u_vars] )
        ds_u = ds_u.rename({'lon':'lon_u', 'lat':'lat_u'})
        ds_u = ds_u.swap_dims({'x':'x_u', 'y':'y_u'})

        # Regrid stresses onto V using XESMF
        ds_v = xr.Dataset( coords = dict( lon = (['y','x'], lon_v),
                                          lat = (['y','x'], lat_v) ) )
        regridder = xe.Regridder(data, ds_v, "bilinear")
        ds_v = regridder( data[v_vars] )
        ds_v = ds_v.rename({'lon':'lon_v', 'lat':'lat_v'})
        ds_v = ds_v.swap_dims({'x':'x_v', 'y':'y_v'})

        # Place interpolated variables back into self.data
        for vname in u_vars:
            data[vname] = ds_u[vname]
        for vname in v_vars:
            data[vname] = ds_v[vname]
            
    def to_ROMS(self, ds_grd, rotate_to_grid = True ):
        '''
        Convert storm dataset to a format appropriate for ROMS forcing format.
        This will be done in place, changing self.data.

        Dataset will be interpolated onto U and V coordinates using interpolate_to_arakawa(),
        which uses XESMF. Variables will be renamed and relevant attributes will be added.

        Args:
            ds_grd (xr.Dataset): ROMS grid in xarray Dataset form, opened from netcdf.
                Must contain rho, u and v coordinates.
            rotate_to_grid (bool): If true, rotate vector variables to be relative to
                ROMS grid, rather than North/East. This is done by applying an anti-clockwise
                rotation using ds_grd.angle.
        '''

        data = self.data

        # Rotate grid using rotate_vectors()
        if rotate_to_grid:
            rot_angle = ds_grd.angle.rename({'eta_rho':'y', 'xi_rho':'x'})
            U_rot, V_rot = _utils.rotate_vectors( self.data['stress_u'],
                                                  self.data['stress_v'],
                                                  rot_angle )
            data['stress_u'] = U_rot
            data['stress_v'] = V_rot
        
        # Interpolate to Arakawa grid first
        self.interpolate_to_arakawa( ds_grd.lon_u.values, ds_grd.lat_u.values,
                                     ds_grd.lon_v.values, ds_grd.lat_v.values )
        
        # Rename and assign attributes
        if 'stress_u' in data:
            self.data = _output_formats.make_ROMS_dataset( ds_grd, time = self.track.time, 
                                                   stress_u = data.stress_u.values, 
                                                   stress_v = data.stress_v.values, 
                                                   pressure = data.pressure.values )
        else:
            self.data = _output_formats.make_ROMS_dataset( ds_grd, time = self.track.time, 
                                                   pressure = data.pressure.values )

        return

    def to_netcdf( fp_out, **kwargs ):
        ''' Write this objects dataset to netCDF file using xarray.Dataset.to_netcdf().
        Kwargs are passed to the xarray function. Will remove file if it already exists.

        Args:
            fp_out (str): Output file string.
        '''

        if os.path.exists(fp_out):
            os.remove(fp_out)
        self.data.to_netcdf( fp_out, **kwargs )

    def plot( self, timestep = None, field = 'windspeed' ):
        ''' Quick plotting routined to view the generated storm. 

        Args:
            timestep (int, None): If None (default), will plot the envelope of field.
                If int, will plot the storm for that timestep
            field (str): Field to plot. Either 'pressure' or 'windspeed'
        '''
        
        data = self.data
        f = plt.figure( figsize = (6,5))

        # Set cmap and title
        if field == 'windspeed':
            cmap = plt.get_cmap('Reds',8)
            title = 'Parametric TC wind speed (m/s)'
        elif field == 'pressure':
            cmap = plt.get_cmap('Greens_r',8)
            title = 'Parametric TC pressure (m/s)'

        if timestep is not None:
            plt.pcolormesh( data.lon, data.lat, data[field].isel(time=timestep), cmap = cmap)
            plt.colorbar()
            plt.quiver( data.lon[::3,::3], data.lat[::3,::3], 
                        data.wind_u[timestep][::3,::3], 
                        data.wind_v[timestep][::3,::3] )
        else:
            if field == 'windspeed':
                data_opt = data.windspeed.max(dim='time')
            elif field == 'pressure':
                data_opt = data.pressure.min(dim='time')
            plt.pcolormesh( data.lon, data.lat, data_opt, cmap = cmap ) 

        plt.plot( self.track.lon, self.track.lat, c='b', linewidth=1 )
        plt.scatter( self.track.lon, self.track.lat, c='b', s=10)

        plt.xlim( data.lon.min(), data.lon.max())
        plt.ylim( data.lat.min(), data.lat.max())
        plt.title(title)

        