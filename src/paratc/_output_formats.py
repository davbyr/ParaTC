def make_ROMS_dataset( ds_grd, time, 
                       stress_u = None, stress_v=None, 
                       pressure = None, 
                       track_lon = None, track_lat = None):
    ''' Create a forcing dataset in the correct format for ROMS. You can provide
    each oof pressure, stress_u and stress_v optionally. Only those provided will
    be saved to the dataset. Track_lon and track_lat do not change the dataset, but
    can be optionally saved to the dataset for later reference.

    Args:
        ds_grd (xr.Dataset): ROMS grid file, read into an xarray Dataset
        time (array ): Time variable for dataset.
        stress_u (np.ndarray): U component of windstress with dimension (time, y, x)
        stress_v (np.ndarray): V component of windstress with dimension (time, y, x)
        pressure (np.ndarray): Surface pressure with dimension (time, y, x)
        track_lon (np.ndarray): Tropical cyclone track longitudes
        track_lat (np.ndarray): Tropical cyclone track latitudes.

    Returns:
        Xarray Dataset object in ROMs format, ready for writing.
        
    '''

    ds_tmp = ds_grd[['lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v']]
    ds_tmp['sms_time'] = (['sms_time'], time)
    ds_tmp['pair_time'] = (['pair_time'], time)

    if stress_u is not None:
        ds_tmp['sustr'] = (['sms_time','eta_u', 'xi_u'], stress_u)
        ds_tmp['sustr'].attrs = {'long_name':"surface u-momentum stress",
                                 'units':"m/s",
                                 'field':"surface u-momentum stress",
                                 'time':"sms_time",
                                 'coordinates':'lon_u lat_u' }
    if stress_v is not None:
        ds_tmp['svstr'] = (['sms_time','eta_v', 'xi_v'], stress_v)
        ds_tmp['svstr'].attrs = {'long_name':"surface v-momentum stress",
                                 'units':"Newton meter-2",
                                 'field':"surface v-momentum stress",
                                 'time':"sms_time",
                                 'coordinates':'lon_v lat_v' }

    if pressure is not None:
        ds_tmp['Pair'] = (['pair_time','eta_rho', 'xi_rho'], pressure)
        ds_tmp['Pair'].attrs = {'long_name':"surface air pressure",
                                 'units':"millibar",
                                 'field':"surface air pressure",
                                 'time':"pair_time",
                                 'coordinates':'lon_rho lat_rho' }
        
    ds_tmp.sms_time.encoding['units'] = 'days since 1900-01-01'
    ds_tmp.pair_time.encoding['units'] = 'days since 1900-01-01'

    if track_lon is not None:
        ds_tmp['track_lon'] = (['sms_time'], track_lon)
        ds_tmp['track_lat'] = (['sms_time'], track_lat)

    return ds_tmp
