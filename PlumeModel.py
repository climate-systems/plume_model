
'''
PURPOSE: To run the plume model on moisture (specific humidity)
         and temperature inputs and to output plume virtual temp.
                  
AUTHOR: Fiaz Ahmed

DATE: 05/28/24
'''

import numpy as np
from thermodynamic_functions import theta_e_calc, qs_calc, temp_v_calc
from thermo_functions import plume_lifting
from scipy.interpolate import interp1d
from copy import deepcopy
import xarray as xr
from pathlib import Path
from thermodynamic_functions import calc_geopotential_height
import warnings

class PlumeModel:

    """
    This is a plume model. It takes in thermodynamic profiles
    """

    def __init__(self, fils, 
                 preprocess, output_dir, 
                 output_file_name, mix_opt = 'DIB',
                 interpolate = True,
                 launch_opt = 'surface', # launch from surface or specify pressure level 
                 launch_level = 1000,
                 DIB_mix_upper = 450, 
                 rain_out = 1e-3, 
                 microphysics = 1, 
                 C0 = 0.1) -> None:
        
        if launch_opt not in ['surface', 'specified']:
            raise ValueError('launch_opt must be either surface or specified')
        if mix_opt not in ['DIB', 'NOMIX']:
            raise ValueError('mix_opt must be either DIB or NOMIX')

        self.fils = fils
        self.preprocess = preprocess
        self.launch_lev = launch_level  # launch level in hPa
        self.launch_opt = launch_opt
        self.DIB_mix_upper = DIB_mix_upper
        self.micro = microphysics
        self.C0 = C0

        self.temp_v_plume = None
        self.temp_plume = None
        self.mix_opt = mix_opt
        self.rain_out = rain_out  # rain out hydrometeors above this value
        self.interpolate = interpolate
        self.output_file_name = Path(output_dir) / output_file_name

    @staticmethod
    def __interpolate_data(T, q, lev, launch_lev):

        lev_highres = np.arange(1050, 150, -5)  # highres levels at 5 hPa
        lev_highres = np.int_(lev_highres)

        T_interp = np.zeros((T.shape[0], lev_highres.size))
        q_interp = np.zeros_like(T_interp)

        ## Interpolate the temperature and sp.humidity information ###
        for i in range(T.shape[0]):
            f = interp1d(lev[i,:], T[i,:], bounds_error = False)
            T_interp[i,:] = f(lev_highres)

            f = interp1d(lev[i,:], q[i,:], bounds_error = False)
            q_interp[i,:] = f(lev_highres)

        return T_interp, q_interp, lev_highres


    def preprocess_data(self):

        """
        Take thermodynamic profiles and preprocess them
        to be fed into the plume model. Interpolate the data
        to a higher vertical resolution before running the model.
        The output is expected to be a (time, pressure) 2-D array. 
        """

        # preprocess data
        T, q, lev, self.time = self.preprocess(self.fils)
        self.T, self.q, self.lev = T, q, lev

        # interpolate to higher resolution
        if self.interpolate:
            T, q, self.lev = self.__interpolate_data(T, q, lev, self.launch_lev)

        Tmasked = np.ma.masked_invalid(T)
        qmasked = np.ma.masked_invalid(q)
        z = calc_geopotential_height(Tmasked, qmasked, self.lev, vertical_axis = -1)

        # Change data type 
        self.T = np.float_(T)
        self.q = np.float_(q)
        self.z = np.float_(z)

        self.time_dim = np.arange(self.T.shape[0])
        self.ind_launch = np.zeros((self.time_dim.size), dtype='int')
        self.ind_surface = np.zeros((self.time_dim.size), dtype='int')  # index of the nearest non-NaN level to the surface
        
        # empty arrays for mixing coefficients
        self.c_mix_DIB = np.zeros_like((self.T)) 
        self.c_mix_NOMIX = np.zeros_like((self.T))

        ### Obtain indices closest to surface level 

        isurf = np.isnan(self.T).argmin(axis = 1)# + 1  # find the first NaN in the temperature profile. This is the launch level
        assert (self.lev[isurf]> 800).all(), 'Surface level is below 800 hPa'
        self.ind_surface[:] = isurf
    
        if self.launch_opt == 'surface':
            self.ind_launch[:] = isurf 
        elif self.launch_opt == 'specified':
            ilaunch = np.argmin(abs(self.lev -self.launch_lev))
            self.ind_launch[:] = ilaunch 

        ## Launch plume from specified launch level ##

        Tlaunch = self.T[[range(self.ind_launch.size)],[self.ind_launch]]
        qlaunch = self.q[[range(self.ind_launch.size)],[self.ind_launch]]

        assert (np.isfinite(Tlaunch).all()), 'Launch level temperature is NaN'
        assert (np.isfinite(qlaunch).all()), 'Launch level temperature is NaN'

 
    def mixing_coefficients(self):

        """
        Prescribing DIB mixing coefficients
        """
        iz_upper = np.argmin(abs(self.lev - self.DIB_mix_upper))

        assert(all(self.ind_surface >= 0))
        self.c_mix_DIB[:] = np.nan
        w_mean = np.zeros_like(self.c_mix_DIB)

        with warnings.catch_warnings(record = True) as w:  # catch divide by zero warning
            for i in range(self.T.shape[0]):

                ## Compute Deep Inflow Mass-Flux 
                ## The mass-flux (= vertical velocity) is a sine wave from near surface
                ## to iz_upper. Designed to be 1 at the level of launch
                ## and 0 above max. w i.e., no mixing in the upper trop.
                arg = np.pi * 0.5 * (self.lev[self.ind_surface[i] - 1] - self.lev) / (self.lev[self.ind_surface[i] - 1] - self.lev[iz_upper])
                w_mean[i, :] = np.sin(arg)    
                self.c_mix_DIB[i, 1:-1]= (w_mean[i, 2:] - w_mean[i, :-2])/(w_mean[i, 2:] + w_mean[i, :-2])

        self.c_mix_DIB[:, iz_upper:] = 0.0
        self.c_mix_DIB[self.c_mix_DIB < 0] = 0.0
        self.c_mix_DIB[np.isinf(self.c_mix_DIB)] = 0.0

        assert (np.ma.masked_invalid(self.c_mix_DIB)>=0).all() & (np.ma.masked_invalid(self.c_mix_DIB)<=1).all(), 'Mixing coeffs. must be between 0 and 1'

    def run_plume(self, mix):

        # Set output variables 
        self.temp_plume, self.temp_v_plume = np.zeros_like(self.T), np.zeros_like(self.T)
        self.q_plume, self.ql_plume, self.qi_plume = np.zeros_like(self.T), np.zeros_like(self.T), np.zeros_like(self.T)
        self.q_rain = np.zeros_like(self.T)  # raining hydrometeors

        if mix == 'DIB':
            mixing_coefs = self.c_mix_DIB
        elif mix == 'NOMIX':
            mixing_coefs = self.c_mix_NOMIX

        # Run entraining plume model
        print(f'RUNNING {mix} PLUME COMPUTATION')
        plume_lifting(self.T, self.q, self.temp_v_plume, self.temp_plume, 
                      self.q_plume, self.ql_plume, self.qi_plume, self.q_rain,
                      mixing_coefs, self.lev, self.ind_launch, self.rain_out,
                      self.micro, self.C0)
        
        self.mix_opt = mix

    def postprocess_save(self):

        """
        Save the plume properties to a netCDF file.
        """

        qsat_env = qs_calc(self.lev, self.T)
        Tv_env = temp_v_calc(self.T, self.q, 0.0)
        T_env = deepcopy(self.T)
        q_env = deepcopy(self.q)

        # fill plume properties with NaNs
        nan_idx = np.where(np.isnan(self.T))
        self.temp_plume[nan_idx] = np.nan
        self.temp_v_plume[nan_idx] = np.nan

        zero_idx = np.where(self.temp_plume == 0)  # levels where plume is not active      
        T_env[zero_idx] = np.nan
        Tv_env[zero_idx] = np.nan
        q_env[zero_idx] = np.nan

        self.temp_v_plume[zero_idx] = np.nan
        self.temp_plume[zero_idx] = np.nan

        nan_idx = np.where(np.isnan(self.temp_plume))
        self.q_plume[nan_idx] = np.nan
        self.ql_plume[nan_idx] = np.nan
        self.qi_plume[nan_idx] = np.nan
        self.q_rain[nan_idx] = np.nan

        thetae_env = theta_e_calc(self.lev, self.T, self.q, 0.0)
        thetae_sat_env = theta_e_calc(self.lev, self.T, qsat_env, 0.0)
        thetae_plume = theta_e_calc(self.lev, self.temp_plume, self.q_plume, self.ql_plume)

        print('SAVING FILE')

        data_vars = dict(T_plume = (("time", "lev"), self.temp_plume),
                         Tv_plume = (("time", "lev"), self.temp_v_plume),
                         T_env = (("time", "lev"), T_env),
                         q_env = (("time", "lev"), q_env),
                         Tv_env = (("time", "lev"), Tv_env),
                         thetae_env = (("time", "lev"), thetae_env),
                         thetae_sat_env = (("time", "lev"), thetae_sat_env),
                         thetae_plume = (("time", "lev"), thetae_plume),
                         q_plume = (("time", "lev"), self.q_plume),
                         ql_plume = (("time", "lev"), self.ql_plume),
                         qi_plume = (("time", "lev"), self.qi_plume),
                         q_rain = (("time", "lev"), self.q_rain)    
                         )
        coords = dict(time = self.time, lev = self.lev)

        attrs = {"mixing formulation" : f"{self.mix_opt}"}

        # manually clobber file, as sometimes we can get permission errors
        file_out = self.output_file_name.with_suffix('.nc')
        if Path.exists(file_out):
            Path.unlink(file_out)

        file_out = str(file_out)
        ds = xr.Dataset(data_vars, coords, attrs)
        ds.to_netcdf(file_out)
        print(f'File saved as {file_out}')

    def main(self):

        self.preprocess_data()
        self.mixing_coefficients()
        self.run_plume(mix = self.mix_opt)
        self.postprocess_save()