
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
                 conserved = 1,
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
        self.conserved = conserved

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
                      mixing_coefs, self.lev, self.ind_launch, self.conserved, 
                      self.rain_out, self.micro, self.C0)
        
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

class JPPlume(PlumeModel):
    
    """
    Run John Peters' plume model using the unsaturated
    and saturated lapse rate formulas from (Peters et al. 2022)
    https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1 
    """

    def __init__(self, fils, preprocess, output_dir, output_file_name, mix_opt='DIB', 
                 interpolate=True, launch_opt='surface', launch_level=1000, DIB_mix_upper=450, rain_out=0.001, 
                 conserved=1, microphysics=1, C0=0.1, fracent = 0., prate = 0., T1 = 273.15, T2 = 233.15):
        
        super().__init__(fils, preprocess, output_dir, output_file_name, mix_opt, interpolate, launch_opt, 
                         launch_level, DIB_mix_upper, rain_out, conserved, microphysics, C0)
        self.T1 = T1
        self.T2 = T2
        self.fracent = fracent
        self.prate = prate
        
    
    def declare_constants(self):
        """
        Declare CONSTANTS
        Rd: dry gas constant
        Rv: water vapor gas constant
        epsilon = Rd/Rv   
        g: gravitational acceleration
        cp: heat capacity of dry air at constant pressure
        xlv: reference latent heat of vaporization at the triple point temperature
        xls: reference latent heat of sublimation at the triple point temperature
        cpv: specific heat of water vapor at constant pressure
        cpl: specific heat of liquid water
        cpi: specific heat of ice
        ttrip: triple point temperature
        eref: reference pressure at the triple point temperature
        """
        self.const = dict(Rd = 287.04, Rv = 461.5, g = 9.81, cp = 1005, xlv = 2501000)
        self.const['epsilon'] = self.const['Rd']/self.const['Rv']
        self.const  = self.const | dict(xls = 2834000, cpv = 1870, cpl = 4190, cpi = 2106, ttrip = 273.15, eref = 611.2)
        

    def prepare_jp_plume(self):

        # preprocess data and get mixing coefficients
        super().preprocess_data()
        super().mixing_coefficients()
        self.declare_constants()

        const = self.const
        #ESTIMATE THE MOIST STATIC ENERGY (MSE)
        self.__compute_geopotential_height()
        self.MSE = const['cp'] * self.T + const['xlv'] * self.q + self.phi
        self.z = self.phi / const['g'] # geopotential height
        self.mn_hgt = np.nanargmin( self.MSE, axis = 1 ) #FIND THE INDEX OF THE HEIGHT OF MINIMUM MSE
        
        #descriminator function between liquid and ice (i.e., omega defined in the
        #beginning of section 2e in Peters et al. 2022)

        # Set output variables 
        self.temp_plume, self.temp_v_plume = np.full_like(self.T, np.nan), np.full_like(self.T, np.nan)
        self.q_plume, self.ql_plume, self.qi_plume = np.full_like(self.T, np.nan), np.full_like(self.T, np.nan), np.full_like(self.T, np.nan)
        self.qt_plume, self.q_rain = np.full_like(self.T, np.nan), np.full_like(self.T, np.nan)  # raining hydrometeors

        # initialize plume variables
        if self.ind_launch[0] > 0:
            indx = slice(0, self.ind_launch[0] + 1)
        elif self.ind_launch == 0:
            indx = 0

        self.temp_plume[:, indx] = self.T[:, indx] #set initial values to that of the environment
        self.q_plume[:, indx] = self.q[:, indx] #set initial values to that of the environment
        self.qt_plume[:, indx] = self.q_plume[:, indx] #set initial values to that of the environment

        self.B_plume = None  # buoyancy of the plume
    

    def __compute_geopotential_height(self):
        Tv = temp_v_calc(self.T, self.q, 0.0)  # possibly redundant calc for Tv
        integrand = self.const['Rd'] * Tv / self.lev
        dp = abs( np.gradient(self.lev) )
        self.phi = np.nancumsum(integrand * dp, axis = 1, out = integrand)

    @staticmethod
    def omega(T,T1,T2):
        """
        descriminator function between liquid and ice (i.e., omega defined in the
        beginning of section 2e in Peters et al. 2022)
        """
        return ((T - T1)/(T2-T1))*np.heaviside((T - T1)/(T2-T1),1)*np.heaviside((1 - (T - T1)/(T2-T1)),1) + np.heaviside(-(1 - (T - T1)/(T2-T1)),1);
    @staticmethod
    def domega(T,T1,T2):
        return (np.heaviside(T1-T,1) - np.heaviside(T2-T,1))/(T2-T1)


    def __compute_rsat(self, T, p, iceflag):
        #FUNCTION THAT CALCULATES THE SATURATION MIXING RATIO    
        #THIS FUNCTION COMPUTES THE SATURATION MIXING RATIO, USING THE INTEGRATED
        #CLAUSIUS CLAPEYRON EQUATION (eq. 7-12 in Peters et al. 2022).
        #https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1 

        #input arguments
        #T temperature (in K)
        #p pressure (in hPa)
        #iceflag (give mixing ratio with respect to liquid (0), combo liquid and
        #ice (2), or ice (3)
        #T1 warmest mixed-phase temperature
        #T2 coldest mixed-phase temperature
        
        #NOTE: most of my scripts and functions that use this function need
        #saturation mass fraction qs, not saturation mixing ratio rs.  To get
        #qs from rs, use the formula qs = (1 - qt)*rs, where qt is the total
        #water mass fraction

        #CONSTANTS
        const = self.const
        p = p * 100 #convert pressure to Pa

        T1, T2 = self.T1, self.T2
        omeg = self.omega(T, T1, T2)
        Rv = const['Rv']
        cpv, cpl, cpi = const['cpv'], const['cpl'], const['cpi']
        xlv, xls = const['xlv'], const['xls']
        ttrip = const['ttrip']
        eref = const['eref']
        epsilon = const['epsilon']

        if iceflag == 0:

            term1 = ( cpv - cpl )/Rv
            term2 = ( xlv - ttrip * (cpv - cpl) )/Rv
            esl = np.exp( (T - ttrip) * term2 / (T * ttrip)) * eref * pow(T/ttrip, term1)
            qsat = epsilon * esl / (p - esl)

        elif iceflag == 1: #give linear combination of mixing ratio with respect to liquid and ice (eq. 20 in Peters et al. 2022)

            term1 = (cpv - cpl)/Rv
            term2 = ( xlv - ttrip*(cpv-cpl) ) / Rv
            esl_l = np.exp( ( T - ttrip ) * term2/( T * ttrip )) * eref * pow( T/ttrip, term1)
            qsat_l = epsilon * esl_l/(p - esl_l)
            term1 = (cpv - cpi) / Rv
            term2 = ( xls - ttrip * (cpv - cpi)) / Rv
            esl_i = np.exp( (T - ttrip) * term2/(T * ttrip)) * eref * pow(T/ttrip, term1)
            qsat_i = epsilon * esl_i / ( p - esl_i )
            qsat = ( 1 - omeg ) * qsat_l + omeg * qsat_i

        elif iceflag == 2: #only give mixing ratio with respect to ice
            term1 = ( cpv - cpi ) / Rv
            term2=( xls - ttrip * ( cpv - cpi )) / Rv
            esl = np.exp( (T - ttrip ) * term2/(T * ttrip) )* eref * pow(T/ttrip, term1)
            esl = min( esl , p * 0.5 )
            qsat = epsilon * esl/(p-esl)

        return qsat

    def drylift(self, T, qv, T0, qv0, fracent):

        #CONSTANTS
        Rd, Rv = self.const['Rd'], self.const['Rv'] #dry air & water vapor gas constants
        cp, cpv = self.const['cp'], self.const['cpv'] #specific heat of dry air & water vapor at constant pressure
        g = self.const['g'] #gravitational acceleration
        
        cpmv = (1 - qv) * cp + qv * cpv
        B = g *( (T-T0)/T0 + (Rv/Rd - 1) * (qv - qv0) )
        eps = -fracent*(T - T0)
        gamma_d = - (g + B)/cpmv + eps
        return gamma_d
    
    #LAPSE RATE FOR A SATURATED PARCEL
    def moistlift(self, T, qv, qvv, qvi, qt, T0, q0, fracent, prate):
        
        #CONSTANTS
        const = self.const
        T1, T2 = self.T1, self.T2
        cp, cpv, cpl, cpi = const['cp'], const['cpv'], const['cpl'], const['cpi']
        xlv, xls = const['xlv'], const['xls']
        g, ttrip = const['g'], const['ttrip']
        Rd, Rv, epsilon = const['Rd'], const['Rv'], const['epsilon']
    
        qt = max(qt, 0.0)
        qv = max(qv, 0.0)
        
        OMEGA = self.omega(T, T1, T2)
        dOMEGA = self.domega(T, T1, T2)
        
        cpm = (1 - qt) * cp + qv * cpv + (1 - OMEGA) * ( qt - qv ) * cpl + OMEGA * ( qt - qv ) * cpi
        Lv = xlv + ( T - ttrip ) * (cpv - cpl)
        Li = ( xls - xlv ) + ( T - ttrip ) * ( cpl - cpi )
        Rm0 = (1 - q0) * Rd + q0 * Rv
        
        T_rho = T * (1 - qt + qv/epsilon)
        T_rho0 = T0 * ( 1 - q0 + q0/epsilon )
        B = g * (T_rho - T_rho0) / T_rho0
        
        Qvsl = qvv / ( epsilon - epsilon * qt + qv)
        Qvsi = qvi / ( epsilon - epsilon * qt + qv)
        Q_M = (1 - OMEGA) * qvv / (1 - Qvsl) + OMEGA * qvi / (1 - Qvsi)
        L_M = Lv * (1 - OMEGA) * qvv / (1 - Qvsl) + (Lv + Li) * OMEGA * qvi / (1 - Qvsi)

        eps_T =  -fracent * (T - T0)
        eps_qv = -fracent * (qv - q0)
        eps_qt = -fracent * (qt - q0) - prate * (qt-qv)
        term1 = -B
        
        term2 = - Q_M * (Lv + Li * OMEGA ) * g / ( Rm0 * T0 )
        
        term3 = -g
        term4 = (cpm - Li * (qt - qv) * dOMEGA) * eps_T
        term5 = ( Lv + Li * OMEGA ) * (eps_qv + qv/(1-qt) * eps_qt )

        term6 = cpm
        term7 = -Li * (qt - qv) * dOMEGA
        term8 = ( Lv + Li * OMEGA ) * ( -dOMEGA * (qvv - qvi) + L_M/(Rv * pow(T,2)) )
        gamma_m =( term1 + term2 + term3 + term4 + term5) / (term6 + term7 + term8)
        return gamma_m

    def compute_moist_lapse_rate(self, T_plume, q_plume, qt_plume, T_env, Q0_env, p, fracent, prate):
        """
        Wrapper to compute moist lapse rate
        """
        # moistlift(T, qv, qvv, qvi, T0, q0, qt, fracent, prate):
        qvv = (1 - qt_plume) * self.__compute_rsat(T_plume, p, 0)
        qvi = (1 - qt_plume) * self.__compute_rsat(T_plume, p, 2)
        return self.moistlift(T_plume, q_plume, qvv, qvi, qt_plume, T_env, Q0_env, fracent, prate)

    def lift_parcel_adiabatic(self, fracent, prate):
        
        iend = self.lev.size - 1
        itime = self.T.shape[0]
        T1, T2 = self.T1, self.T2
        iceflag = 1
        Rd, Rv, g = self.const['Rd'], self.const['Rv'], self.const['g']
        
        #for iz in np.arange(start_loc+1,z0.shape[0]):
        #
        #
        #I REVISED THIS A BIT.  TO MAKE THE CODE FASTER, I HAVE THE CALCULATION CUT OUT WHEN THE INTEGRATED NEGATIVE BUOYANCY ("BRUN") 
        #BECOMES MORE NEGATIVE THAN THAN THE TOTAL INTEGRATED POSITIVE BUOYANCY.  I RESTRICT THIS TO ONLY HAPPEN AFTER WE HAVE PASSED 
        #THE HEIGHT OF MINIMUM MSE.  UNCOMMENT THE FOR LOOP ABOVE AND COMMENT OUT THE WHILE LOOP IF YOU JUST WANT TO INTEGRATE TO THE TOP OF THE SOUNDING.
        #THE +25 PART IN THE WHILE STATEMENT IS A PAD ON B_RUN (THE NEGATIVE CAPE HAS TO BE 25 J/KG LESS THAN THE POSITIVE CAPE TO KILL THE LOOP)
        #while iz<(z0.shape[0])-1 and (z0[iz]<z0[mn_hgt] or (B_run+25)>0):
        
        for it in range(itime):

            q_sat_prev = 0
            B_run = 0
            iz = self.ind_launch[it]

            while (iz < iend) and ( (iz < self.mn_hgt[it])  or  (B_run + 250  > 0) ):

                iz = iz + 1
                rsat = self.__compute_rsat(self.temp_plume[it, iz-1], self.lev[iz-1], iceflag)

                q_sat = ( 1 - self.qt_plume[it, iz-1] ) * rsat
                delta_z = self.z[it, iz] - self.z[it, iz-1]


                if self.q_plume[it, iz-1] < q_sat: #if we are unsaturated, go up at the unsaturated adiabatic lapse rate (eq. 19 in Peters et al. 2022)


                    dry_lapse_rate = self.drylift(self.temp_plume[it, iz-1], self.q_plume[it, iz-1], 
                                                  self.T[it, iz-1], self.q[it, iz-1], fracent)

                    # if it == 0:
                    #     print(q_sat, rsat, self.temp_plume[it, iz-1] )
                    #     print(dry_lapse_rate)
                    #     print('--'*20)

                    self.temp_plume[it, iz] = self.temp_plume[it, iz-1] + delta_z * dry_lapse_rate
                    self.q_plume[it, iz] = self.q_plume[it, iz-1] - delta_z * fracent * ( self.q_plume[it, iz-1] - self.q[it, iz-1] )
                    self.qt_plume[it, iz] = self.q_plume[it, iz]
                    q_sat = (1 - self.qt_plume[it, iz]) * self.__compute_rsat(self.temp_plume[it, iz], self.lev[iz], iceflag)

                    if self.q_plume[it, iz] >= q_sat: #if we hit saturation, split the vertical step into two stages.  The first stage advances at the saturated lapse rate to the saturation point, and the second stage completes the grid step at the moist lapse rate


                        satrat = (self.q_plume[it, iz] - q_sat_prev) / ( q_sat - q_sat_prev )
                        dz_dry = satrat * delta_z
                        dz_wet=(1 - satrat) * delta_z

                        T_halfstep = self.temp_plume[it, iz-1] + dz_dry * dry_lapse_rate
                        Qv_halfstep = self.q_plume[it, iz-1] - dz_dry * fracent * ( self.q_plume[it, iz-1] - self.q[it, iz-1] )
                        Qt_halfstep = self.q_plume[it, iz] 

                        p_halfstep =  self.lev[iz-1] * satrat + self.lev[iz] * ( 1 - satrat)
                        T0_halfstep = self.T[it, iz-1] * satrat + self.T[it, iz] * ( 1 - satrat )
                        Q0_halfstep = self.q[it, iz-1] * satrat + self.q[it, iz] * (1-satrat)

                        # moist lapse rate
                        moist_lr = self.compute_moist_lapse_rate(T_halfstep, Qv_halfstep, Qt_halfstep, T0_halfstep, 
                                                                 Q0_halfstep, p_halfstep, fracent, prate)


                        self.temp_plume[it, iz] = T_halfstep + dz_wet * moist_lr
                
                        self.qt_plume[it, iz] = self.qt_plume[it, iz-1] -  delta_z * fracent * ( Qt_halfstep - Q0_halfstep )
                        self.q_plume[it, iz] = ( 1 - self.qt_plume[it,iz] ) * self.__compute_rsat(self.temp_plume[it, iz], self.lev[iz], 1)

                        if self.qt_plume[it, iz] < self.q_plume[it, iz]:
                            self.q_plume[it, iz] = self.qt_plume[it, iz]


                    q_sat_prev = q_sat
                    
                else: #if we are already at saturation, just advance upward using the saturated lapse rate (eq. 24 in Peters et al. 2022)
                    moist_lr = self.compute_moist_lapse_rate(self.temp_plume[it, iz-1], self.q_plume[it, iz-1], self.qt_plume[it, iz-1],
                                                            self.T[it, iz-1], self.q[it, iz-1], self.lev[iz-1], fracent, prate)
                    
                    self.temp_plume[it, iz] = self.temp_plume[it, iz-1] + delta_z * moist_lr
                             
                    self.qt_plume[it, iz] = self.qt_plume[it, iz-1] - delta_z * (fracent * ( self.qt_plume[it, iz-1] - self.q[it, iz-1] )  + prate * ( self.qt_plume[it, iz-1] - self.q_plume[it, iz-1]) )
                    self.q_plume[it, iz] = ( 1 - self.qt_plume[it, iz] ) * self.__compute_rsat(self.temp_plume[it, iz], self.lev[iz], 1)

                    if self.qt_plume[it, iz] < self.q_plume[it, iz]:
                        self.q_plume[it, iz] = self.qt_plume[it, iz]


                num = 1 + (Rv/Rd) * self.q_plume[it, iz] - self.qt_plume[it, iz]
                den = 1 + self.q[it, iz] * (Rv - Rd) / Rd
                B_run = B_run + g *  ( ( self.temp_plume[it, iz] / self.T[it, iz] ) * (num / den) - 1 ) * delta_z


        T_rho_lif = self.temp_plume[it, :] * (1 + (Rv/Rd) * self.q_plume[it, :] - self.qt_plume[it, :])
        T_0_lif = self.T[it, :] * (1 + (Rv/Rd - 1) * self.q[it, :])
        self.B_plume = g * (T_rho_lif - T_0_lif) / T_0_lif
        self.temp_v_plume = temp_v_calc(self.temp_plume, self.q_plume, self.qt_plume)
        
        def main(self):
            self.prepare_jp_plume()
            self.lift_parcel_adiabatic(self.fracent, self.prate)
            self.postprocess_save()
    