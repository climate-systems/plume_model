"""
Name: thermo.pyx

Author: Fiaz Ahmed (exhaustive rewrite of original IDL code by Chris Holloway and Kathleen Schiro)

Date: 2024-06-05

Description: This file contains several utility functions for thermodynamic calculation and 
             a plume lifting function.
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp,log,pow, sqrt
import cython
# from sys import exit

cdef extern from "math.h":
    bint isfinite(double x)


DTYPE = np.float64
DTYPE1 = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE1_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)

# Thermodyanmic constants 

cdef temp_v_calc(double temp, double q, double qt):

    """
    Compute virtual temperature following 
    Emanuel (1994), pg. 113, Eq. (4.3.6).

    Input Arguments
    ---------------
    temperature (K), specific humidity (kg/kg) and specific total water (vapor + water + ice)

    Returns
    -------
    Density temperature (which reduces to virtual temperature if qt = q)
    """

    cdef double r, rl, rc
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS
    cdef double temp_v
    
    EPS=RD/RV

    r = q/(1 - qt)
    rT = qt/(1 - qt)

    temp_v = temp * (1 + (r/EPS)) 
    temp_v = temp_v/(1 + rT)    
    
    return temp_v

cdef es_calc_bolton(double temp):
    # in hPa

    cdef double tmelt  = 273.15
    cdef double tempc, es  
    tempc = temp - tmelt 
    es = 6.112*exp(17.67*tempc/(243.5+tempc))
    return es

cdef es_calc(double temp):

    cdef double ttrip  = 273.15
    cdef double cpv = 1870
    cdef double cpl = 4190
    cdef double Rv = 461.5
    cdef double xlv = 2501000.0
    cdef double eref = 611.2

    cdef double term1, term2, esl

    term1 = ( cpv - cpl )/Rv
    term2 = ( xlv - ttrip * (cpv - cpl) )/Rv
    esl = exp( (temp - ttrip) * term2 / (temp * ttrip) ) * eref * pow( temp / ttrip, term1)
    
    return esl

cdef esi_calc(double temp):

    cdef double ttrip  = 273.15
    cdef double cpv = 1870
    cdef double cpl = 4190
    cdef double Rv = 461.5
    cdef double xlv = 2501000.0
    cdef double eref = 611.2
    cdef double xls = 2834000.0
    cdef double cpi = 2106.0

    cdef double term1, term2, esi

    term1 = (cpv - cpi) / Rv
    term2 = ( xls - ttrip * (cpv - cpi)) / Rv

    esi = exp( (temp - ttrip) * term2/(temp * ttrip)) * eref * pow(temp/ttrip, term1)

    return esi


cdef qs_calc(double press_hPa, double temp):

    cdef double tmelt  = 273.15
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS, press, tempc, es, qs

    EPS = RD/RV

    press = press_hPa * 100. 
    tempc = temp - tmelt 

    es = es_calc(temp) 
    qs = (EPS * es) / (press + ((EPS-1.)*es))
    return qs

cdef qsi_calc(double press_hPa, double temp):

    cdef double tmelt  = 273.15
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS
    cdef double press,esi,qsi
        
    EPS=RD/RV
    press = press_hPa * 100
    esi = esi_calc(temp) 
    qsi = (EPS * esi) / (press + ((EPS-1.)*esi))
    return qsi


# from ChatGPT
cdef omega(double T, double T1, double T2):
    """
    Discriminator function between liquid and ice 
    (i.e., omega defined in the beginning of section 2e in Peters et al. 2022)
    """
    cdef double x = (T - T1) / (T2 - T1)
    cdef double h1 = 1.0 if x > 0 else 0.0
    cdef double h2 = 1.0 if (1 - x) > 0 else 0.0
    return x * h1 * h2 + (1.0 if (1 - x) <= 0 else 0.0)

cdef compute_rsat(double press_hPa, double temp, double T1, double T2, int iceflag):

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

    cdef double press

    press = press_hPa * 100 #convert pressure to Pa
    omeg = omega(temp, T1, T2)

    cdef double RV=461.5
    cdef double RD=287.04
    cdef double cpv = 1870.
    cdef double cpl = 4190.
    cdef double cpi = 2106.
    cdef double xls = 2834000.
    cdef double xlv = 2501000.
    cdef double ttrip = 273.15
    cdef double eref = 611.2
    cdef double EPS

    cdef double rsat, esl, esi, qsat_l, qsat_i, term1, term2
    cdef double esl_l, esl_i

    EPS = RD/RV

    if iceflag == 0:

        term1 = ( cpv - cpl )/RV
        term2 = ( xlv - ttrip * (cpv - cpl) )/RV
        esl = exp( (temp - ttrip) * term2 / (temp * ttrip)) * eref * pow(temp / ttrip, term1)
        rsat = EPS * esl / (press - esl)

    elif iceflag == 1: #give linear combination of mixing ratio with respect to liquid and ice (eq. 20 in Peters et al. 2022)

        term1 = (cpv - cpl)/RV
        term2 = ( xlv - ttrip * ( cpv - cpl ) ) / RV
        esl_l = exp( ( temp - ttrip ) * term2/( temp * ttrip )) * eref * pow( temp / ttrip, term1)
        qsat_l = EPS * esl_l/(press - esl_l)

        term1 = (cpv - cpi) / RV
        term2 = ( xls - ttrip * (cpv - cpi)) / RV
        esl_i = exp( (temp - ttrip) * term2/(temp * ttrip)) * eref * pow(temp / ttrip, term1)
        qsat_i = EPS * esl_i / ( press - esl_i )
        
        rsat = ( 1 - omeg ) * qsat_l + omeg * qsat_i

    elif iceflag == 2: #only give mixing ratio with respect to ice
        term1 = ( cpv - cpi ) / RV
        term2 = ( xls - ttrip * ( cpv - cpi )) / RV
        esi = exp( (temp - ttrip ) * term2/(temp * ttrip) )* eref * pow(temp / ttrip, term1)
        esi = min( esi , press * 0.5 )
        rsat = EPS * esi / ( press - esi )

    return rsat


cdef theta_v_calc(double press_hPa, double temp, double q, double qt):

    """
    Compute virtual potential temperature following 
    Emanuel (1994), pg. 112, Eq. (4.3.2).

    Input Arguments
    ---------------
    pressure (hPa), temperature (K), specific humidity (kg/kg) and specific total water (vapor + water + ice)

    Returns
    -------
    Density potential temperature (which reduces to virtual potential temperature if qt = q)

    Reference
    ---------
        Emanuel, K. A. (1994). Atmospheric convection. Oxford University Press, USA.

    """

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double  RD=287.04
    cdef double pres, temp_v,theta_v

    press = press_hPa * 100.
    temp_v = temp_v_calc(temp, q, qt)
    theta_v = temp_v * pow((pref/press), (RD/CPD) )
    return theta_v

cdef theta_il_calc(double press_hPa, double temp, double q, double ql, double qi, int qsat_opt):

    """
    Compute ice-liquid potential temperature (theta_il) following eq.(25) from Bryan and Fritsch 2004, MWR.
    The theta_il is approximately conserved under internal phase change (i.e., between vapor to ice to liquid).
    In the absence of any condensate, the theta_il reduces to the potential temperature.

    Input Arguments
    ---------------
    pressure (hPa), temperature (K), specific humidity q (kg/kg), specific liquid water ql (kg/kg) 
    specific ice water qi (kg/kg), qsat_opt: 1-use empirical formulations for rh, 2-use formulas from Peters et al. 2022

    Returns
    -------
    Ice - liquid potential temperature theta_il (K)

    References
    ---------
        - Bryan, G. H., & Fritsch, J. M. (2004). A reevaluation of ice–liquid water potential temperature. Monthly weather review, 132(10), 2421-2431.
        - Tripoli, G. J., & Cotton, W. R. (1981). The use of ice-liquid water potential temperature as a thermodynamic variable in deep atmospheric models. 
          Monthly Weather Review, 109(5), 1094-1102.
        - Hauf, T., & Höller, H. (1987). Entropy and potential temperature. Journal of Atmospheric Sciences, 44(20), 2887-2901.

    """

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CI=2106.0
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double LS0=2.834E6
    cdef double T1 = 273.15
    cdef double T2 = 233.15

    cdef double press,tempc
    cdef double r, rl, ri, rt, qsl, qsi
    cdef double ALV, LS, chi, gam
    cdef double theta_il, Hl, Hi, e
    cdef double exponent, CPDV, ci

    press = press_hPa * 100. 
    tempc = temp - tmelt 

    r = q / (1. - q - ql - qi)
    rl =  ql / (1. - q - ql - qi)
    ri =  qi / (1. - q - ql - qi)
    rt = r + rl + ri

    ALV = ALV0 - (CPVMCL * tempc)    
    LS = LS0 + (CPV - CI) * tempc

    CPDV = CPD + rt * CPV

    chi = (RD + rt * RV) / CPDV
    gam = (rt * RV) / CPDV

    # saturation vapor pressures
    e = r * press /(EPS + r)
    # rsat = EPS * esl / (press - esl)

    if qsat_opt == 1:

        Hl = e/es_calc(temp)
        Hi = e/esi_calc(temp)

    elif qsat_opt == 2:

        qsl = compute_rsat(press_hPa, temp, T1, T2, 0) * (1 - q - ql - qi)
        qsi = compute_rsat(press_hPa, temp, T1, T2, 2) * (1 - q - ql - qi)
        Hl = q/qsl
        Hi = q/qsi


    exponent = (-ALV*rl - LS*ri)/(CPDV * temp) + (RV/CPDV) * (rl * log(Hl) +  ri * log(Hi))

    # eq. (25) of Bryan and Fritsch, 2004.
    theta_il = temp * pow(pref / press, chi) * pow(1. - (rl + ri)/(EPS + rt), chi) * pow(1. - (rl + ri)/rt, -gam) * exp(exponent)
    return theta_il

cdef theta_e_calc (double press_hPa, double temp, double q, double ql, double qi):

    """
    Function from Todd Emmenegger

    Compute equivalent potential temperature (thetae) following  Eq. (2.67) in Stevens & Siebesma (2020)
    which, with ice correction, is equivalent to Eq. (4.5.11) in Emanuel (1994).
 
    Input Arguments
    ---------------
    pressure (hPa), temperature (K), specific humidity q (kg/kg) and specific liquid water ql (kg/kg) 

    Returns
    -------
    Equivalent potential temperature thetae (K)

    Reference
    ---------
        Emanuel, K. A. (1994). Atmospheric convection. Oxford University Press, USA.

    """

    cdef double pref = 100000.
    cdef double tmelt  = 273.1636783445389; # Triple point = freezing point
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CPVMCL=2320.0
    cdef double CL = 4184.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double LF = 0.3337E6 # latent heat of fusion at 0-deg-C (lv0-ls)
    cdef double press, tempc,theta_e
    cdef double r, ev_hPa, TL, chi_e, H, ALV
    cdef double qt, es, ci
    
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C

    ci = 2106. + 7.3 * tempc
    ALV = ALV0 - CPVMCL * tempc

    qt = q + ql + qi
    r = q / (1. - q - ql - qi)

    # vapor pressures
    e = r * press /(EPS + r)

    if temp >= tmelt:
        es = es_calc(temp)
    elif temp < tmelt:
        es = esi_calc(temp)

    RE = ( 1 - qt ) * RD
    R = RE + q * RV

    CPL = CPD + qt * ( CL - CPD)

    chi = RE / CPL
    gamma = RV * q / CPL
    gammi = ( CL - ci ) * qi / CPL

    exponent = ( q * ALV / temp - LF * qi / tmelt ) / CPL
    theta_e = temp * pow( pref * R / (press * RE), chi) * pow(es/e, gamma) * pow(tmelt / temp, gammi) * exp( exponent )
    return theta_e


cdef theta_l_calc(double press_hPa, double temp,double q,double ql):

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double press,tempc,theta_l
    cdef double r,rl,rt
    cdef double ALV,chi, gamma
    
    press = press_hPa * 100.
    tempc = temp - tmelt 
    r = q / (1. - q - ql)
    rl =  ql / (1. - q - ql)
    rt = r + rl

    ALV = ALV0 - (CPVMCL * tempc)

    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))
    gam = (rt * RV) / (CPD + (rt * CPV))

    theta_l = temp * pow(pref / press, chi) * pow(1. - rl/(EPS + rt), chi) * pow((1. - rl/rt), -gam) * exp( -ALV * rl /((CPD + rt * CPV) * temp))
    return theta_l

cdef phi_temp(double press_hPa, double temp_plume, double qt, 
              double rain_out_thresh, int freeze, int conserved, int micro, double C0, 
              int qsat_opt):

    """
    Wrapper to calculate ice-liquid-vapor partitioning and conserved variable phi for a given temperature and total water content,
        based on the microphysics scheme.
    
    Input Arguments
    ---------------
    pressure (hPa), 
    temp: temperature (K), 
    qt: total water content (vapor + liquid + ice) (kg/kg) 
    rain_out_thresh: threshold for rainout (kg/kg) 
    freeze: flag for freezing (0 or 1) 
    conserved: flag for conserved variable (1 - theta_il, 2 - theta_e)
    micro: microphysics scheme (1 or 2)
    C0 : autoconversion coefficient (unitless). The fraction of condensate above the rain_out threshold
        that is converted to rain.
    qsat_opt: option to compute qsat in different ways (1 or 2)
    1: use empirical formulas, 2: use formulas from Peters et al. 2022

    Returns
    -------
    theta_il: ice-liquid-vapor partitioning and total water content (kg/kg)
    q: vapor mixing ratio (kg/kg)
    ql: liquid water mixing ratio (kg/kg)
    qi: ice mixing ratio (kg/kg)
    qt: total water content (kg/kg)

    """

    cdef double phi, q, ql, qi, qt_out

    # use microphysics to compute vapor-water-ice partitioning
    if micro == 1:
        q, ql, qi, qt_out = microphysics1(press_hPa, temp_plume, qt, freeze)

    elif micro == 2:
        q, ql, qi, qt_out = microphysics2(press_hPa, temp_plume, qt)

    # compute theta_il
    if conserved == 1:
        phi = theta_il_calc(press_hPa, temp_plume, q, ql, qi, qsat_opt)
    elif conserved == 2:
        phi = theta_e_calc(press_hPa, temp_plume, q, ql, qi)

    return phi, q, ql, qi, qt_out

cdef invert_phi(double press_hPa, double phi_target, double qt, 
                double rain_out_thresh, int freeze, int conserved, int micro, double C0, 
                double T0, int qsat_opt):

    """
    Invert conserved variable phi to obtain temperature using the secant method.

    Our given function has the form f[x, y(x), ...]. Note that y is a function of x.
    For a target value f0, and initial guesses x0 and x1, the secant method gives an updated guess x2:
    x2 = x1 - (f[x1, y(x1), ...] - f0) * (x1 - x0) / (f[x1, y(x1), ...] - f[x0, y(x0), ...]).  
    The updates stop when the new guess is within a specified tolerance of the old guess (convergence is assumed).
    This method is liable to break down if the function is not smooth or if the initial guesses are poor.

    For our case, f[x, y(x)] = theta_il[p, T, q, ql(T), qi(T)], where the condensate amounts ql and qi are functions of temperature T.
    The microphysics parameterization will determine ql and qi for a given T. The target value f0 is the theta_il that must be inverted. 

    Input Arguments
    ---------------
    pressure (hPa), theta_il (K), 
    qt: total water content (vapor + liquid + ice) (kg/kg),
    rain_out_thresh: threshold for rainout (kg/kg),
    freeze: flag for freezing (0 or 1),
    conserved: flag for conserved variable (1 - theta_il, 2 - theta_e),
    micro: microphysics scheme (1 or 2),
    C0 : autoconversion coefficient (unitless) 
    T0: first guess for temperature (K)

    Returns
    -------
    temp: inverted temperature(K)
    plume water species consistent with temp: 
        q: vapor mixing ratio (kg/kg)
        ql: liquid water mixing ratio (kg/kg)
        qi: ice mixing ratio (kg/kg)

    Note
    ----
    The total water qt must be unaffected by the inversion. 

    """

    cdef double pref = 100000.
    cdef double CPD = 1005.7
    cdef double CPV = 1870.0
    cdef double RD = 287.04
    cdef double RV = 461.5
    cdef double ALV0=2.501E6
    cdef double press, Ttemp, T1
    cdef double deltaT = 999.0
    cdef double temp, phi1, phi0
    cdef double fx, dfx, dfxx
    cdef double q, ql, qi, q_rain, qt_new
    cdef double q0, ql0, qi0, q_rain0, qt0
    
    cdef double rt, chi, gam

    press = press_hPa * 100. 
    rt = qt/(1-qt)
    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))

    T1 = T0 + 2.0  # add 2 K for the second guess

    while abs(deltaT) >= 1e-4:  # convergence criterion
        
        phi1, q, ql, qi, qt_new = phi_temp(press_hPa, T1, qt, rain_out_thresh, freeze, conserved, micro, C0, qsat_opt) # f[x1, y(x1), ...]
        phi0, q0, ql0, qi0, qt0 = phi_temp(press_hPa, T0, qt, rain_out_thresh, freeze, conserved, micro, C0, qsat_opt) # f[x0, y(x0), ...]

        fx = phi1 - phi_target  # f[x, y(x), ...] - f0 # minimize this
        dfx = (phi1 - phi0)/(T1 - T0) # f'[x, y(x), ...]
        deltaT = fx / dfx
        T0 = T1
        T1 = T1 - deltaT  # update x1

    phi_new, q, ql, qi, qt_new = phi_temp(press_hPa, T1, qt, rain_out_thresh, freeze, conserved, micro, C0, qsat_opt)

    return T1, q, ql, qi


cdef microphysics1(double press_hPa, double temp_plume, double qt, int freeze):

    """
    Microphysics scheme 1: no supersaturation allowed. All condensed water is in the form of liquid above 0C and ice below 0C.
    The freeze flag is used to check if the plume is below/above freezing.

    Input Arguments
    ---------------
    pressure (hPa), temperature (K), total water content (vapor + liquid + ice) (kg/kg), freeze flag (0 or 1)

    Returns
    -------
    q: vapor mixing ratio (kg/kg)
    ql: liquid water mixing ratio (kg/kg)
    qi: ice mixing ratio (kg/kg)
    qt: total water content (kg/kg)
    """

    cdef double tmelt = 273.15
    cdef double qsw, qsi
    cdef double q_plume, ql_plume, qi_plume, qt_plume

    if freeze == 0:
        qi_plume = 0.0
        qsw = qs_calc(press_hPa, temp_plume) # get saturation sp. humidity wrt water

        if qt < qsw:
            q_plume = qt
            ql_plume = 0.0
        else:
            q_plume = qsw                               
            ql_plume = qt - q_plume  # plume liq. water

    elif freeze == 1:
        ql_plume = 0.0
        qsi = qsi_calc(press_hPa, temp_plume)

        if qt < qsi:
            q_plume = qt
            qi_plume = 0.0
        else:
            q_plume = qsi
            qi_plume = qt - q_plume

    qt_plume = q_plume + ql_plume + qi_plume
    return q_plume, ql_plume, qi_plume, qt_plume

cdef microphysics2(double press_hPa, double temp_plume, double qt):

    """
    Microphysics scheme 2: supersaturation wrt water allowed between 0C to -40C. All condensed water is in the form of liquid above 0C and ice below -40C.
    In the mixed-phase region (-40C to 0C), the fraction of ice and water is specified by a temperature-dependent function Fi.
    Following Bryan and Fritsch 2004, MWR. 

    Input Arguments
    ---------------
    pressure (hPa), temperature (K), total water content (vapor + liquid + ice) (kg/kg), freeze flag (0 or 1)

    Returns
    -------
    q: vapor mixing ratio (kg/kg)
    ql: liquid water mixing ratio (kg/kg)
    qi: ice mixing ratio (kg/kg)
    qt: total water content (kg/kg)

    Reference
    ---------
        Bryan, G. H., & Fritsch, J. M. (2004). A reevaluation of ice–liquid water potential temperature. Monthly weather review, 132(10), 2421-2431.
    """

    
    cdef double tmelt = 273.15
    cdef double qsw, qsi
    cdef double q_plume, ql_plume, qi_plume, q_rain, qt_plume
    cdef double Fi
    cdef double T1 = 273.15, T2 = 233.15

    q_rain = 0.0

    if temp_plume > 273.15:  # above freezing

        qi_plume = 0.0
        qsw = qs_calc(press_hPa, temp_plume) # get saturation sp. humidity wrt water
        if qt < qsw:
            q_plume = qt
            ql_plume = 0.0
        else:
            q_plume = qsw                               
            ql_plume = qt - q_plume  # plume liq. water


    elif temp_plume <= 273.15 and temp_plume >= 233.15:  # mixed-phase region

        qsw = qs_calc(press_hPa, temp_plume) # get saturation sp. humidity wrt water
        qsi = qsi_calc(press_hPa, temp_plume)

        Fi = (273.15 - temp_plume)/(273.15 - 233.15)  # fraction of ice in the mixed-phase region

        if qt < qsi:   # unsaturated wrt ice and water
            q_plume = qt
            qi_plume = 0.0
            ql_plume = 0.0
        
        elif qt > qsi and qt < qsw: # saturated wrt ice, unsaturated wrt water
            q_plume = qsi
            qi_plume = qt - q_plume
            ql_plume = 0.0

        elif qt > qsw:  # saturated wrt both ice and water
            q_plume = Fi * qsi + (1 - Fi) * qsw
            qi_plume = Fi * (qt - q_plume)
            ql_plume = qt - q_plume - qi_plume

    elif temp_plume < 233.15:  # below -40C

        ql_plume = 0.0
        qsi = qsi_calc(press_hPa, temp_plume)

        if qt < qsi:
            q_plume = qt
            qi_plume = 0.0
        else:
            q_plume = qsi
            qi_plume = qt - q_plume

    qt_plume = q_plume + ql_plume + qi_plume
    return q_plume, ql_plume, qi_plume, qt_plume

cdef microphysics3(double press_hPa, double temp_plume, double qt, int freeze):

    """
    Microphysics scheme 3:  All condensed water is in the form of liquid above 0C. 
    The liquid water is converted to ice to maintain the plume temperature at freezing.
    The freeze flag is used to check if the plume is below/above freezing.

    Input Arguments
    ---------------
    pressure (hPa), temperature (K), total water content (vapor + liquid + ice) (kg/kg), freeze flag (0 or 1)

    Returns
    -------
    q: vapor mixing ratio (kg/kg)
    ql: liquid water mixing ratio (kg/kg)
    qi: ice mixing ratio (kg/kg)
    qt: total water content (kg/kg)
    """

    cdef double tmelt = 273.15
    cdef double qsw, qsi
    cdef double q_plume, ql_plume, qi_plume, qt_plume

    if freeze == 0:
        qi_plume = 0.0
        qsw = qs_calc(press_hPa, temp_plume) # get saturation sp. humidity wrt water

        if qt < qsw:
            q_plume = qt
            ql_plume = 0.0
        else:
            q_plume = qsw                               
            ql_plume = qt - q_plume  # plume liq. water

    elif freeze == 1:
        ql_plume = 0.0
        qsi = qsi_calc(press_hPa, temp_plume)

        if qt < qsi:
            q_plume = qt
            qi_plume = 0.0
        else:
            q_plume = qsi
            qi_plume = qt - q_plume

    qt_plume = q_plume + ql_plume + qi_plume
    return q_plume, ql_plume, qi_plume, qt_plume



cdef rainout_hydrometeors(double temp, double press_hPa, double qt,
                          double ql, double qi, double rain_out_thresh, 
                          double C0):

    """
    Rainout hydrometeors above a specified threshold with efficiency C0. The conversion to rain is irreversible.
    
    Returns
    -------
    ql_new: liquid water mixing ratio (kg/kg)
    qi_new: ice water mixing ratio (kg/kg)
    qt_new: total water content (kg/kg)
    q_rain: rainout (kg/kg)
    """
    
    cdef double q_rain, ql_new, qi_new, qiql_ratio

    q_rain = 0.0
    ql_new = ql
    qi_new = qi

    qsw = qs_calc(press_hPa, temp) # get saturation sp. humidity wrt water
    qsi = qsi_calc(press_hPa, temp)

    if temp > 273.15:  # Above freezing
        if ql > rain_out_thresh:
            q_rain = C0 * (ql - rain_out_thresh)                              
            ql_new = ql - q_rain            

    if temp < 273.15:  # Below freezing
        if qi > 0 or ql > 0:
            qiql_ratio = qi/(ql + qi)

            # Convert condensate to rain above threshold 
            # maintain the same ratio between qi and ql as before
            if (qi + ql) > rain_out_thresh:                  
                q_rain = C0 * (qi + ql - rain_out_thresh)
                qi_new = qiql_ratio * (qi + ql - q_rain)
                ql_new = (1 - qiql_ratio) * (qi + ql - q_rain)
        
    qt_new = qt - (ql + qi) + (ql_new + qi_new)

    # check water conservation, turned off for speed, but turn on if debugging
    # if abs(qt_new + q_rain - qt) > 1e-7: 
    #     print(qt_new + q_rain, qt)
    #     raise AssertionError("Total water not conserved")

    return ql_new, qi_new, qt_new, q_rain

cdef freeze_all_liquid(double press_hPa, double temp, double qt):
    
    """
    Convert all liquid water to ice in one (irreversible) step following Emanuel (1994), p. 139.
    The enthalpy of vapor-liquid equilibrium at T1 is assumed to be equal to the enthalpy of vapor-ice equilibrium at T2
    This relationship is used to obtain an expression for T2 - T1. 
    
    Input Arguments
    ---------------
    pressure (hPa), temperature (K), total water content (vapor + liquid + ice) (kg/kg)

    Returns
    -------
    temp_new: temperature (K) = temp + (T2 - T1)
    q_new: vapor mixing ratio (kg/kg)
    qi_new: liquid water mixing ratio (kg/kg) is excess above ice saturation

    Reference
    --------
    Emanuel, K. A. (1994). Atmospheric convection. Oxford University Press, USA.
    """

    # convert liquid water to ice in one (irreversible) step :
    cdef double LF0 = 0.3337E6
    cdef double LV0 = 2.501E6
    cdef double RV = 461.5
    cdef double CPD = 1005.7
    cdef double CL = 4190.0
    cdef double alpha, bet, b1, b2
    cdef double qs_fr, q_fr, ql_fr , q2_fr, qi_fr, qsi_fr
    cdef double rl_fr, rs_fr, rt_fr
    cdef double a_fr, b_fr, c_fr, deltaTplus
    cdef double qsw

    qsw = qs_calc(press_hPa, temp)  # get saturation sp. humidity wrt water 
    
    if qt < qsw:  # if plume is unsaturated wrt liquid water
        q_fr = qt  
    else:
        q_fr = qsw
    
    ql_fr = qt - q_fr   # get total water that must be frozen

    # compute mixing ratios
    rl_fr = ql_fr/(1. - q_fr - ql_fr) 
    rs_fr = qsw/(1. - qsw)
    rt_fr = rl_fr + rs_fr

    alpha = 0.009705  ## linearized e#/e* around 0C (to -1C)
    b1 = esi_calc(temp)  # saturation vapor pressure over ice
    b2 = es_calc(temp)   # saturation vapor pressure over water
    bet = b1 / b2


    a_fr = (LV0 + LF0) * alpha * LV0 * rs_fr / (RV * pow(temp, 2))
    b_fr = CPD + (CL * rt_fr) + (alpha * (LV0 + LF0) * rs_fr) +(bet * (LV0 + LF0) * LV0 * rs_fr /(RV * pow(temp, 2)))
    c_fr = ((-1.) * LV0 * rs_fr) - (LF0 * rt_fr) + (bet * (LV0 + LF0) * rs_fr)
    deltaTplus = (-b_fr + sqrt(pow(b_fr,2) - (4 * a_fr * c_fr))) / (2 * a_fr)

    return temp + deltaTplus, q_fr, ql_fr  # return temperature, vapor and ice


def plume_lifting(np.ndarray[DTYPE_t, ndim=2] temp_env,
                    np.ndarray[DTYPE_t, ndim=2] q_env,
                    np.ndarray[DTYPE_t, ndim=2] temp_v_plume,
                    np.ndarray[DTYPE_t, ndim=2] temp_plume,
                    np.ndarray[DTYPE_t, ndim=2] q_plume,
                    np.ndarray[DTYPE_t, ndim=2] ql_plume,
                    np.ndarray[DTYPE_t, ndim=2] qi_plume,
                    np.ndarray[DTYPE_t, ndim=2] q_rain,
                    np.ndarray[DTYPE_t, ndim=2] phi_plume,
                    np.ndarray[DTYPE_t, ndim=2] c_mix,
                    np.ndarray[DTYPE1_t, ndim=1] pres, 
                    np.ndarray[DTYPE1_t, ndim=1] ind_init,
                    int conserved, double rain_out_thresh, 
                    int micro, double C0, int qsat_opt):  

    '''
    Description
    -----------
    Function to lift a plume with given environmental sounding, specified launch level and microphysics.
    The plume is assumed to conserve ice-liquid potential temperature (theta_il) and total water (qt)
    except upon mixing and hydrometeor rainout.

    The steps involved in plume lifting are:

        i. The plume is initialized with the environmental temperature and specific humidity at the launch level.
    
        ii. Mixing is specified using an array of coefficients c_mix. For Deep Inflow B (DIB), c_mix = 1 at the launch level, 
            and monotonically decreases above the launch level. For non-entraining (NOMIX), c_mix = 0
            everywhere at and above the launch level.
    
        iii. a) The plume is lifted. The plume theta_il and qt variables are mixed with environmental values. 
             b) The temperature is inferred from the new theta_il and qt, using the specified microphysics scheme.
             c) Hydrometeors are rained out according to specified functions. This changes the total water, and the liquid and ice water contents.
                A new theta_il is accordingly computed with the new qt, ql and qi. This procedure is discussed in Tripoli and 
                Cotton 1981, (pg. 1097, col. 2) and is termed the 'saturation adjustment' method in Bryan and Fritsch 2004.
             d) If the microphysics1 scheme is chosen, all the liquid water is converted to ice when the plume temperature falls below freezing.
             e) Steps a-d are repeated for the next vertical level.

    Input arguments
    ---------------
    temp_env: 2D array
        Environmental temperature profile (K)
    q_env: 2D array
        Environmental specific humidity profile (kg/kg)
    c_mix: 2D array
        Mixing coefficients. 
    ind_init: 1D array
        Index of the initial launch level
    conserved: int
        Conserved variable (1 or 2)
        1 - Conserve ice-liquid potential temperature 
        2 - Conserve equivalent potential temperature 
    rain_out_thresh: float
        Threshold for rainout (kg/kg)
    micro: int
        Microphysics scheme (1 or 2)
        1 - no supersaturation allowed. All condensed water is in the form of liquid above 0C and ice below 0C
        2 - supersaturation allowed between 0-40C. All condensed water is in the form of liquid above 0C and ice below -40C (Following Bryan and Fritsch 2002, MWR)
    C0: float
        Autoconversion coefficient (unitless). The fraction of condensate above the rain_out threshold
        that is converted to rain.
    qsat_opt: int
        option to use different formulas to compute qsat (1 or 2)
        1: uses empirical formulas, 2: uses formulas from Peters et al. 2022
    
    Modified arguments
    ------------------
    temp_plume: 2D array
        Plume temperature profile (K)
    temp_v_plume: 2D array
        Plume virtual temperature profile (K)
    q_plume: 2D array
        Plume specific humidity profile (kg/kg)
    ql_plume: 2D array
        Plume liquid water profile (kg/kg)
    qi_plume: 2D array
        Plume ice water profile (kg/kg)
    q_rain: 2D array
        Contribution to rain (fallout) profile (kg/kg)
    phi_plume: 2D array
        Plume conserved variable profile (K)
    '''

    # Environmental T,q, lifting level, mixing coefficients and microphysics scheme are user defined

    cdef unsigned int time_size = temp_env.shape[0]        
    cdef unsigned int height_size = temp_env.shape[1]            
    cdef unsigned int freeze
    
    # phi is a generic plume-conserved variable
    cdef np.ndarray[DTYPE_t, ndim=1] phi_env= np.zeros(height_size)
    cdef np.ndarray[DTYPE_t, ndim=1] qt_plume= np.zeros(height_size)

    cdef Py_ssize_t i,j 
    
    cdef double tmelt = 273.15
    cdef double residual
        
    for j in range(time_size):
        for i in range(0, height_size):  

            if conserved == 1:
                phi_env[i] = theta_il_calc(pres[i], temp_env[j,i], q_env[j,i], 0.0, 0.0, qsat_opt)
            elif conserved == 2:
                phi_env[i] = theta_e_calc(pres[i], temp_env[j,i], q_env[j,i], 0.0, 0.0)

        # initialize plume properties
        phi_plume[j, ind_init[j]] = phi_env[ind_init[j]]
        qt_plume[ind_init[j]] = q_env[j,ind_init[j]]

        temp_plume[j,ind_init[j]] = temp_env[j,ind_init[j]]
        temp_v_plume[j,ind_init[j]] = temp_v_calc(temp_plume[j,ind_init[j]], q_env[j,ind_init[j]], q_env[j,ind_init[j]])

        q_plume[j,ind_init[j]] = q_env[j,ind_init[j]]
        ql_plume[j,ind_init[j]] = 0.0
        qi_plume[j,ind_init[j]] = 0.0

        freeze = 0

        # lift plume
        for i in range(ind_init[j] + 1, height_size):

            # Mix the ice-liquid potential temperature and total water
            if c_mix[j,i-1] > 0:
                phi_plume[j, i] = phi_plume[j, i-1] * (1. - c_mix[j,i-1]) + phi_env[i-1] * c_mix[j,i-1]
                qt_plume[i] = (qt_plume[i-1] * (1. - c_mix[j,i-1])) + (q_env[j,i-1] * c_mix[j,i-1])
            else:                
                phi_plume[j, i] = phi_plume[j, i-1]
                qt_plume[i] = qt_plume[i-1]
            
            if (isfinite(phi_plume[j, i]) & isfinite(qt_plume[i])):

                temp_plume[j,i], q_plume[j,i], ql_plume[j,i], qi_plume[j,i] = invert_phi(pres[i], phi_plume[j, i], qt_plume[i], 
                                                                                         rain_out_thresh, freeze, conserved, micro, C0, 
                                                                                         temp_plume[j, i-1], qsat_opt)

                # Rainout hydrometeors and get new qt, ql and qi 
                ql_plume[j,i], qi_plume[j,i], qt_plume[i], q_rain[j,i] = rainout_hydrometeors(temp_plume[j,i], pres[i], qt_plume[i], ql_plume[j,i], 
                                                                                              qi_plume[j,i], rain_out_thresh, C0)
                # recompute phi with new qt, ql and qi---which changes upon rainout
                if conserved == 1:
                    phi_plume[j, i] = theta_il_calc(pres[i], temp_plume[j,i], q_plume[j,i], ql_plume[j,i], qi_plume[j,i], qsat_opt)
                elif conserved == 2:
                    phi_plume[j, i] = theta_e_calc(pres[i], temp_plume[j,i], q_plume[j,i], ql_plume[j,i], qi_plume[j,i])

                # if micro = 1, freeze all water if below freezing.
                if micro == 1 and freeze == 0 and temp_plume[j,i] <= tmelt: # check if below freezing
                    temp_plume[j,i], q_plume[j,i], qi_plume[j,i] = freeze_all_liquid(pres[i], temp_plume[j,i], qt_plume[i]) 
                    ql_plume[j,i] = 0.0                                               
                    freeze = 1

                temp_v_plume[j,i] = temp_v_calc(temp_plume[j,i], q_plume[j,i], qt_plume[i])
         
                



        

