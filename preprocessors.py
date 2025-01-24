import numpy as np
from thermodynamic_functions import qs_calc
import scipy.io as sio
from copy import deepcopy
import xarray as xr
from thermodynamic_functions import calc_pres_from_height

def preprocess_data(dir):

    """
    Preprocess data before feeding into plume model.
    The data is converted into a two-dimensional numpy array (time, height)
    of temperature and moisture.
    """

    print('READING AND PREPARING FILES')

    ####### LOAD temp. & sp.hum DATA ########

    file_temp = 'tdry_profiles.mat'
    file_hum = 'q_profiles.mat'
    file_pres = 'pressure_dim_5hPa_interp.mat'
    file_cwv = 'cwv_timeseries_5minavgs_Nauru_01Jan1998_31Dec2008.mat'

    ### Read temp in K ###
    fil = sio.loadmat(dir+file_temp)
    temp = fil['tdrycholloworig']

    ### Read sp.hum in g/kg ###
    fil = sio.loadmat(dir+file_hum)
    sphum = fil['qnmcholloworig'] 
    sphum *= 1e-3 ### Convert sp. humidity from g/kg -> kg/kg 

    ### Read pressure levels ###
    fil = sio.loadmat(dir+file_pres)
    lev = np.int_(np.squeeze(fil['pressure'])) ## convert from short to int

    ### Remove some levels with NaNs
    ind_fin = np.where(np.isfinite(temp[0,:]))[0]

    temp_fin = temp[:,ind_fin]
    sphum_fin = sphum[:,ind_fin]
    lev_fin = lev[ind_fin]

    return temp_fin, sphum_fin, lev_fin, None


def intersect_times(t1, t2):
    return t1.sel(time = t1.time.isin(t2), drop = True)

def preprocess_ARMBE(fils):

    """
    Assumes ARMBE data. Uses the high-resolution height coordinates to calculate pressure.
    Computes specific humidity from dewpoint temperature and pressure.
    """

    ds = xr.open_mfdataset(fils)
    vars = ['temperature_h', 'dewpoint_h', 'pressure_sfc', 'alt', 'precip_rate_sfc']
    ds_subset = ds[vars]

    # get times with no NaNs in the lowest 5 km
    valid_times = ds_subset['temperature_h'].sel(height = slice(0, 5e3)).dropna('time').time  # get times where the lowest 5 km is NaN free.
    valid_times_dewpoint = ds_subset['dewpoint_h'].sel(height = slice(0, 5e3)).dropna('time').time  # get times where the lowest 5 km is NaN free.
    valid_times_pres_sfc = ds_subset['pressure_sfc'].dropna('time').time  # get times where the lowest 5 km is NaN free.

    # get the intersection of times with valid data
    for t in [valid_times_dewpoint, valid_times_pres_sfc]:
        valid_times = intersect_times(valid_times, t)
        
    ds_subset= ds_subset.sel(time = valid_times)

    ds_subset = ds_subset.assign(pressure_h = calc_pres_from_height(ds.height, ds.temperature_h, 
                                                 ds.pressure_sfc))

    ds_subset = ds_subset.assign(sphum_h = qs_calc(ds_subset.pressure_h, ds_subset.dewpoint_h))

    return ds_subset.temperature_h.values, ds_subset.sphum_h.values, ds_subset.pressure_h.values, ds_subset.time


def preprocess_Nauru(fils):

    """
    Assumes ARMBE data. Uses the high-resolution height coordinates to calculate pressure.
    Computes specific humidity from dewpoint temperature and pressure.
    """

    ds = xr.open_mfdataset(fils)
    vars = ['temperature', 'humidity']
    ds_subset = ds[vars]
    lev = ds.levels.broadcast_like(ds['temperature'])
    return ds_subset.temperature.values, ds_subset.humidity.values * 1e-3, lev.values, ds_subset.time



