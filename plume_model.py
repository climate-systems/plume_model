
'''
PURPOSE: To run the plume model on moisture (specific humidity)
         and temperature inputs and to output plume virtual temp.
                  
AUTHOR: Fiaz Ahmed

DATE: 08/27/19
'''

import numpy as np
from netCDF4 import Dataset
from glob import glob
import datetime as dt
from dateutil.relativedelta import relativedelta
import time
import itertools
from mpi4py import MPI
from sys import exit
from numpy import dtype
from land_ocean_parameters import *
from thermodynamic_functions import *
from parameters import *
from thermo_functions import plume_lifting,calc_qsat,invert_theta_il
from scipy.interpolate import interp1d

####### MASK ########
#f=Dataset('/glade/u/home/fiaz/era_process/binning/land_sea_mask.nc','r')
f=Dataset('/glade/u/home/fiaz/analyze/plume_model_calc/land_sea_mask.nc','r')
lsm=np.asarray(f.variables['LSMASK'][:],dtype='float')
# lsm=np.asarray(f.variables['LSMASK'][:],dtype='float')

latm,lonm=f.variables['lat'][:],f.variables['lon'][:]
f.close()

mask_ocean=np.copy(lsm)
mask_ocean[mask_ocean!=1]=np.nan

mask_land=np.copy(lsm)
mask_land[mask_land!=0]=np.nan
mask_land[mask_land==0]=1.

ocean_names=['io','wp','ep','at']
land_names_jja=['ism','wafr','easm']
land_names_djf=['sasm','asm','mc','arge']
land_names=land_names_djf+land_names_jja

### Create list of file to process

# strt_date=dt.datetime(2003,1,1)
strt_date=dt.datetime(2002,1,1)
# end_date=dt.datetime(2014,12,31)
end_date=dt.datetime(2005,12,31)

dts=[]

dir='/glade/p/univ/p35681102/fiaz/erai_data/regridded/'
dirc1=dir+'era_surfp/'
dirc2=dir+'era_q/'
dirc3=dir+'era_T/'

dirc4='/glade/p/univ/p35681102/fiaz/T3B42/'
fil3='TRMM.3B42.'

list1=[]


## 
## This function will convert pressure to height for a 4-dimensional array ##
##

## subset_size


while strt_date<=end_date:

    d1=strt_date.strftime("%Y-%m-%d")
    yr=strt_date.strftime("%Y")
    dts.append(strt_date)
    fname=dirc2+'era_vertq_regridded_'+d1+'*'
    list_temp=(glob(fname))
    list_temp.sort()
    list1.append(list_temp)

    strt_date+=relativedelta(days=1)
    
chain1=itertools.chain.from_iterable(list1)
list1= (list(chain1))

### Get lat, lon ###

f=Dataset(list1[0],'r')
lat=f.variables['lat'][:]
lon=f.variables['lon'][:]
lev_era=f.variables["level"][:]
f.close()


## Create high-res vertical levels for interpolation
lev=np.arange(1000,450,-5)
#     lev=np.arange(1000,45,-5)
lev1=lev_era[lev_era<=450]    
lev=np.concatenate((lev,lev1[::-1]))
lev=np.int_(lev)


dp=np.diff(lev)*-1
i450=np.where(lev==450)[0]
i600=np.where(lev==600)[0]
i1000=np.where(lev==1000)[0]
ilev=i600[0]
ibeg=i1000[0]
####


Tmelt=Tk0 # Set freezing temperature for qsat calculations

comm=MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

if comm.rank == 0:
        jobs=list1
        jobs=split(jobs,comm.size)

else:
        jobs=None

# Scatter jobs across cores
jobs=comm.scatter(jobs,root=0)

s=time.time()

# 
# jobs=list1
for j in jobs:
    d1=j[-13:-3]
    print(j)
    date=dt.datetime.strptime(d1,"%Y-%m-%d")
    day=date.timetuple().tm_mday
    d2=date.strftime("%Y%m")

#     fname1=dirc2+'era_vertq_regridded_'+d1+'*'
#     list1=glob(fname1)
#     f=Dataset(list1[0],"r")
#     q=f.variables["Q"][:]
#     f.close()

    fname2=dirc3+'era_vertT_regridded_'+d1+'*'
    list1=glob(fname2)
    f=Dataset(list1[0],"r")
    T=f.variables["T"][:]
    f.close()

    print 'LOAD FILES'


  ## LOAD PRECIP ###
#     fname3=dirc4+fil3+d2+'*.nc'
#     list_temp=(glob(fname3))
#     f3=Dataset(list_temp[0],"r")
#     precip_trmm=f3.variables['precip_trmm'][(day-1)*4:day*4]
#     lat_trmm=f3.variables["latitude"][:]
#     lon_trmm=f3.variables["longitude"][:]
#     f3.close()   

    #### LOAD SURFACE PRESSURE ######      
    fname4=dirc1+'era_surfp_regridded_'+d1+'*.nc'
    list_temp=(glob(fname4))
    f=Dataset(list_temp[0],'r')
    pres=f.variables['pres'][:]/100 #in hPa
    f.close()
# 
    pres_ocean=pres*mask_land[None,:,:]
    #     cwv_ocean=cwv*mask_land[None,:,:]
    T_ocean=T*mask_land[None,None,:,:]
#     q_ocean=q*mask_land[None,None,:,:]

    
    pres_ocn=pres_ocean.flatten()

    T_p1=np.swapaxes(T_ocean,0,1)
    T_ocn=T_p1.reshape(lev_era.size,-1)

#     q_p1=np.swapaxes(q_ocean,0,1)
#     q_ocn=q_p1.reshape(lev_era.size,-1)

#         print 'INTERPOLATING'

    f=interp1d(lev_era,T_ocn,axis=0,bounds_error=False)
    T_ocn_interp=f(lev)
#     
#     f=interp1d(lev_era,q_ocn,axis=0,bounds_error=False)
#     q_ocn_interp=f(lev)
    
    ### Determine the pressure level closest to the surface
    ind_lev=np.argmin(abs(pres_ocn[None,:]-lev[:,None]),axis=0)

    ### Remove Nans ###

    ind_lev_fin=ind_lev[np.isfinite(pres_ocn)]
    T_fin=T_ocn_interp[:,np.isfinite(pres_ocn)]
#     q_fin=q_ocn_interp[:,np.isfinite(pres_ocn)]
    pres_fin=pres_ocn[np.isfinite(pres_ocn)]
    
    ## Launch plume two pressure levels away from near-surface (about 10 mb up) ##
    ind_launch=ind_lev_fin+2
    
    ## Flip T,q so it is compatible with plume computation functions

    T_fin=T_fin.T
#     q_fin=q_fin.T
    
    
    
    ## Compute Deep Inflow Mass-Flux ##
    ## The mass-flux (= vertical velocity) is a sine wave from near surface
    ## to 450 mb. Designed to be 1 at the level of launch

    w_mean = np.sin(np.pi*0.5*(lev[ind_launch-1][:,None]-lev[None,:])/(lev[ind_launch-1][:,None]-lev[i450]))    
    minomind = np.where(w_mean==np.nanmax(w_mean))[0][0]
    c_mix_DIB = np.zeros((pres_fin.size,lev.size)) ## 0 above max. w
    c_mix_DIB[:,1:-1]= (w_mean[:,2:] - w_mean[:,:-2])/(w_mean[:,2:]+w_mean[:,:-2])
    c_mix_DIB[c_mix_DIB<0]=0.0
    c_mix_DIB[np.isinf(c_mix_DIB)]=0.0

    ### Create influence function matrix ##
    ### I(z',z)=mixing_coeff(z')*mass_flux(z')/mass_flux(z) ###
    
#     mz=(w_mean[:,2:]+w_mean[:,:-2])    
#     mz_prime=(w_mean[:,2:]+w_mean[:,:-2])*c_mix_DIB[:,1:-1]
# 
#     inf_func=np.zeros((c_mix_DIB.shape[0],c_mix_DIB.shape[1],c_mix_DIB.shape[1]))
#     inf_func[:,1:-1,1:-1]=mz_prime[:,None,:]/mz[:,:,None]
# 
#     lower_tril=np.tril(np.ones((inf_func.shape[1],inf_func.shape[2])))
#     inf_func=inf_func*lower_tril[None,...]
# 
#     ## Add and delete rows to correspond to the backward differencing 
#     ## employed in the local mixing scheme 
#     inf_func=np.insert(inf_func,1,inf_func[:,1,:],1)
#     inf_func=np.delete(inf_func,-1,1)

    T_fin=np.float_(T_fin)
#     q_fin=np.float_(q_fin)

    
    print 'Doing plume calculation'
    

    ### Only performing plume computation upto 600 mb, since that is 
    ### the intended level of integration ##
    
    theta_il=np.zeros_like(T_fin)
    qsat=np.zeros_like(T_fin)    
    temp_v_plume_dilute=np.zeros_like(T_fin)    
    temp_plume_dilute=np.zeros_like(T_fin)    
    temp_v_plume=np.zeros_like(T_fin)    

    ## Compute qsat ##
    qsat=qs_calc(lev, T_fin)

    ### CALCULATE THERMODYNAMICS ###
    temp_v = temp_v_calc(T_fin, qsat, 0.) 
    
    
    
    ## Launch plume ###
    print('Launching protected plume')
    
    ### Protect partially? ###
    
    
#     plume_lifting_erai(T_fin[:,:ilev],q_fin[:,:ilev],temp_v_plume[:,:ilev], temp_v[:,:ilev], c_mix_DIB[:,:ilev], lev[:ilev], ind_launch)
    plume_lifting_erai(T_fin[:,:ilev],qsat[:,:ilev],temp_v_plume[:,:ilev], temp_v[:,:ilev], c_mix_DIB[:,:ilev], lev[:ilev], ind_launch)

    tvdiff=g*(temp_v_plume-temp_v)/temp_v
#             tvdiff=temp_v_plume/temp_v
    tvdiff[temp_v_plume==0]=np.nan
#             buoy_forward=tvdiff
 

#             theta_il[:,:ilev]=theta_il_calc(lev[:ilev], T[:,:ilev], q[:,:ilev], 0., 0.)
#             qsat[:,:ilev]=qs_calc(lev[:ilev], T[:,:ilev])
# 
#             ### Matrix multiply  ##
#             ##theta_il_plume (vector)= Influence function (matrix) * theta_il_env (vector)##
#             
#             theta_il_plume=np.einsum('ijk,ik->ij',inf_func,theta_il)
#             qt_il_plume=np.einsum('ijk,ik->ij',inf_func,q)
# 
#             
#             ## Inverting theta_il to get plume temperature ## 
#             print('Inverting theta_il')
#             invert_theta_il(theta_il_plume[:,:ilev],qt_il_plume[:,:ilev],temp_plume_dilute[:,:ilev],lev[:ilev])    
# 
#             ### We are assuming that condensate is carried with plume ###
#             ### Compute plume vapor and liquid water content ##
# 
#             qsat_plume=np.zeros_like(temp_plume_dilute)
#             qsat_plume[:,:ilev]=qs_calc(lev[:ilev], temp_plume_dilute[:,:ilev])
# 
#             qvap_plume=np.copy(qt_il_plume)
#             qliq_plume=qt_il_plume-qsat_plume
# 
#             qvap_plume[qt_il_plume>=qsat_plume]=qsat[qt_il_plume>=qsat_plume]
#             qliq_plume[qliq_plume<0]=0.0
#             
#             ## Computing plume virtual temperature ##
#             temp_v_plume=temp_v_calc (temp_plume_dilute, qvap_plume, qliq_plume)
#             
#             ### Compute integrated buoyancy from launch level to 600 mb ##
#                         
#             tvdiff=g*(temp_v_plume-temp_v)/temp_v
#             tvdiff[temp_v_plume==0]=np.nan
#             print(np.nanmax(tvdiff),np.nanmin(tvdiff))
# 
#             buoy_rev=tvdiff
#             
#             print('Forward:',np.nanmax(buoy_forward),np.nanmin(buoy_forward))
#             print('Reverse:',np.nanmax(buoy_rev),np.nanmin(buoy_rev))
#             exit()

    var=np.copy(tvdiff[:,ibeg:ilev])
    var1=np.copy(temp_v[:,ibeg:ilev])
    var[var1==0]=np.nan

    ind=np.where(np.isfinite(var))[1]
    lev_trunc=lev[ibeg:ilev+1]

    istrt=ind[0]#+10 ## Neglect the initial 50 mb
    buoy_integ=np.nansum(var[:,istrt:ind[-1]]*dp[None,istrt:ind[-1]],axis=1)/(lev_trunc[istrt]-lev_trunc[ind[-1]])
    
    buoy_integ_full=np.zeros_like(pres_ocn)
    buoy_integ_full[np.isfinite(pres_ocn)]=buoy_integ
    buoy_integ_full=buoy_integ_full.reshape(pres_ocean.shape[0],pres_ocean.shape[1],pres_ocean.shape[2])
    
    buoy_integ_full=buoy_integ_full*mask_land[None,:,:]

    
#                 numerical_buoy_pockets[n*4:(n+1)*4,k,:,:]=np.nansum(var[istrt:ind[-1],...]*dp[istrt:ind[-1],None,None,None],axis=0)/(lev_trunc[istrt]-lev_trunc[ind[-1]])


#             print(np.nanmax(buoy_integ),np.nanmin(buoy_integ))
#             exit()
    
    print ('Took %.2f minutes'%((time.time()-s)/60))
    
    print 'SAVING FILE'
#   fout='/glade/p/univ/p35681102/fiaz/buoyancy_profiles/ocean/'+'era_protected_buoy_oceans_hires_'+j[-13:-3]+'.nc'
#   fout='/glade/p/univ/p35681102/fiaz/buoyancy_profiles/ocean/'+'era_trop_protected_buoy_oceans_hires_'+j[-13:-3]+'.nc'
#     fout='/glade/p/univ/p35681102/fiaz/buoyancy_profiles/ocean/num_buoy_test/'+'era_test_buoy_novirt_oceans_hires_'+j[-13:-3]+'.nc'
    fout='/glade/p/univ/p35681102/fiaz/buoyancy_profiles/global/'+'era_protected_buoy_oceans_hires_'+j[-13:-3]+'.nc'

    ##### SAVE FILE ######

    try:ncfile.close()
    except:pass

    ncfile = Dataset(fout, mode='w', format='NETCDF4')

    ncfile.createDimension('time',None)
    ncfile.createDimension('lat',lat.size)
    ncfile.createDimension('lon',lon.size)

    ti = ncfile.createVariable('time',dtype('float32').char,('time'))
    lt = ncfile.createVariable('lat',dtype('float32').char,('lat'))
    ln = ncfile.createVariable('lon',dtype('float32').char,('lon'))

    bint = ncfile.createVariable('buoy_integ',dtype('float32').char,('time','lat','lon'),zlib=True)


    ti[:]=np.arange(4)
    lt[:]=lat
    ln[:]=lon

    bint[:]=buoy_integ_full
    ncfile.close()

    print 'FILE WRITTEN'
    et=time.time()
    print et-s
#   print et-s
