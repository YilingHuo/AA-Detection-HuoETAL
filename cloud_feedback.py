import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.basemap import Basemap
import copy
from scipy import stats,signal
from mpl_toolkits.basemap import addcyclic
import xarray as xr
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
sys.path.append("..")
# or sys.path.append("/Users/full/path/to/file")
from functions import (areaavg_lat,seasonget,smOLSconfidencelev,statslinregressts_slopepvalue,findvariablefile,readinvariable,moving_average,dataanomaly,
calcsatspechum,rolling_mean_along_axis,seasonal_rolling_mean_along_axis)

kernel='' ###'' for using the CAM5 kernel
stryr=1980;endyr=2022  ###use -1 if you want the whole dataset
latmin=70;glblatlim=-91
if latmin<-90:
    area='Global'
elif latmin==70:
    area='' ##Arctic
else:
    area='Lat'+str(latmin)

iens=0
##CESM
# iens=1
# variable='tas'
# # fn = '/global/cfs/cdirs/m3522/cmip6/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Amon/tas/gn/v20190320/tas_Amon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc'
# tas0 = readinvariable(variable,stryr,iens=iens)
# ds=findvariablefile(variable,iens=iens)
# filestryr=1850
##ERA5
# fn = '/global/cfs/cdirs/m1199/huoyilin/e5.sfc.t2m_ssr_ssrd_tisr_tsr_tsrc_ttr_ttrc.19402022.192x288.nc'
fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnlwrfcs_msnswrf_msnswrfcs_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402023.192x288.nc'
# fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnlwrfcs_msnswrf_msnswrfcs_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402023.73x144.nc'
# fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnswrf_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402022.90x144.nc'
filestryr=1940;filestryr1=1951
ds=nc.Dataset(fn)
tas0 = ds['t2m'][12*(stryr-filestryr):12*(endyr-filestryr+1)]

tas = dataanomaly(tas0)
lat=ds['lat'][:];lon=ds['lon'][:]
# lat=xr.DataArray.to_numpy(lat);lon=xr.DataArray.to_numpy(lon)##CESM2
tas1=tas+0
tas_areamean_12mmean=moving_average(areaavg_lat(tas,lat,latmin), 12)
## fn = '/global/cfs/cdirs/m3522/cmip6/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Amon/ta/gn/v20190320/ta_Amon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc'
variable='t'##ERA5
ta0 = readinvariable(variable,stryr,endyr,filestryr=filestryr1,iens=iens)#[:,:-toplevunused] #filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
# variable='ta'##CESM
# ta0 = readinvariable(variable,stryr,endyr,filestryr=filestryr,iens=iens)#[:,:-toplevunused] #filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
ds = findvariablefile(variable,iens=iens) ##CESM2 or ERA5
p = ds['level'][:]  ###ERA5
# p = ds['plev'][:]/100;p=xr.DataArray.to_numpy(p)#[:-toplevunused]/100  ##CESM2
del ds
verticalflip=False
if p[0]<p[-1]:
    p = np.flip(p)
    verticalflip=True
if verticalflip:    ta0 = np.flip(ta0,axis=1)
ta = dataanomaly(ta0)
ntim=len(tas);nyear=int(ntim/12);nlev=len(ta[0]);nlat=len(lat);nlon=len(lon)
KLVL='TOA' ###TOA or SFC kernel 
###CAM5 Kernel
if KLVL=='SFC':varend='S'
else:varend='T'
kernelfolder='/global/cfs/cdirs/m1199/huoyilin/cscratch/CAM5RadiativeKernels/'
ds=nc.Dataset(kernelfolder+'dp_plev.nc')
pdiff=ds['dp'][:]/100
ds=nc.Dataset(kernelfolder+'ts.kernel.nc')
ts_kernel=ds['FLN'+varend][:];ts_kernel_clearsky=ds['FLN'+varend+'C'][:]
# Read air temperature kernel
ds=nc.Dataset(kernelfolder+'t.kernel.plev.nc')
ta_kernel=ds['FLN'+varend][:];ta_kernel_clearsky=ds['FLN'+varend+'C'][:]
###CAM5 Kernel
###ERA5, ERAIKernel
# kernelfolder='/pscratch/sd/h/huoyilin/ERA5_kernels/'
# ds=nc.Dataset(kernelfolder+'thickness_normalized_ta_wv_kernel/dp_era5.nc')
# pdiff=ds['dp'][:]/100
###ERA5, ERAIKernel
###CloudSatKernel
# kernelfolder='/pscratch/sd/h/huoyilin/CloudSat_kernels/'+KLVL+'/CloudSat/'
# ds=nc.Dataset(kernelfolder+KLVL+'_CloudSat_Kerns.nc')
# ts_kernel=-ds['lw_ts'][:];ts_kernel_clearsky=-ds['lwclr_ts'][:]
# ta_kernel=-ds['lw_t'][:];ta_kernel_clearsky=-ds['lwclr_t'][:]
# alb_kernel=ds['sw_a'][:];alb_kernel_clearsky=ds['swclr_a'][:]
# q_LW_kernel=-ds['lw_q'][:];q_LW_kernel_clearsky=-ds['lwclr_q'][:]
# q_SW_kernel=ds['sw_q'][:];q_SW_kernel_clearsky=ds['swclr_q'][:]
# tmp=ds['plev'][:]/100
# pdiff=ta_kernel+np.nan
# pdiff[:,0,:,:]=ds['PS'][:]/100-tmp[0]
# pdiff[:,-1,:,:]=tmp[-1]
# for ilev in range (1,nlev-1):
#     pdiff[:,ilev,:,:]=(tmp[ilev-1]-tmp[ilev+1])/2
###CloudSatKernel
###ERA5Kernel
# ds=nc.Dataset(kernelfolder+'ERA5_kernel_ts_'+KLVL+'.nc')
# ts_kernel=-ds[KLVL+'_all'][:];ts_kernel_clearsky=-ds[KLVL+'_clr'][:]
# dpnodp='thickness_normalized';nodp=''
# if KLVL=='SFC': dpnodp='layer_specified';nodp='no';pdiff[:]=1
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_ta_'+nodp+'dp_'+KLVL+'.nc')
# ta_kernel=-ds[KLVL+'_all'][:];ta_kernel_clearsky=-ds[KLVL+'_clr'][:]
###ERA5Kernel
###ERAIKernel
# klvl1=KLVL.lower()
# ds=nc.Dataset(kernelfolder+'other_kernels/kernel_cloudsat_res2.5_level37_toa_sfc.nc') ##erai, cloudsat
# ts_kernel=-ds['ts_'+klvl1+'_all'][:];ts_kernel_clearsky=-ds['ts_'+klvl1+'_clr'][:]
# ta_kernel=-ds['ta_'+klvl1+'_all'][:];ta_kernel_clearsky=-ds['ta_'+klvl1+'_clr'][:]
# alb_kernel=ds['alb_'+klvl1+'_all'][:];alb_kernel_clearsky=ds['alb_'+klvl1+'_clr'][:]
# q_LW_kernel=-ds['wv_lw_'+klvl1+'_all'][:];q_LW_kernel_clearsky=-ds['wv_lw_'+klvl1+'_clr'][:]
# q_SW_kernel=ds['wv_sw_'+klvl1+'_all'][:];q_SW_kernel_clearsky=ds['wv_sw_'+klvl1+'_clr'][:]
###ERAIKernel
### Albedo feedback
## Collect surface shortwave radiation fields for calculating albedo change
# variable1='rsus';variable2='rsds' ### CESM
# alb=dataanomaly(readinvariable(variable1,stryr,endyr,filestryr=filestryr,iens=iens)/readinvariable(variable2,stryr,endyr,filestryr=filestryr,iens=iens))*100 ###CESM
ds=nc.Dataset(fn);variable1=ds['msdwswrf'][12*(stryr-filestryr):12*(endyr-filestryr+1)]### ERA5
alb=dataanomaly((variable1-ds['msnswrf'][12*(stryr-filestryr):12*(endyr-filestryr+1)])/variable1)*100; del variable1 ### ERA5
# Read TOA albedo kernel
ds=nc.Dataset(kernelfolder+'alb.kernel.nc')###CAM5Kernel
alb_kernel=ds['FSN'+varend][:];alb_kernel_clearsky=ds['FSN'+varend+'C'][:]###CAM5Kernel
# ds=nc.Dataset(kernelfolder+'ERA5_kernel_alb_'+KLVL+'.nc') ###ERA5Kernel
# alb_kernel=ds[KLVL+'_all'][:];alb_kernel_clearsky=ds[KLVL+'_clr'][:]###ERA5Kernel
### Water vapor feedback
# variable='hus' ###CESM2
# hus = readinvariable(variable,stryr,endyr,filestryr=filestryr,iens=iens)#[:,:-toplevunused]#filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
variable='q' ##ERA5
hus = readinvariable(variable,stryr,endyr,filestryr=filestryr1,iens=iens)#[:,:-toplevunused]#filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
if verticalflip:    hus = np.flip(hus,axis=1)
# the change in moisture
dq=dataanomaly(hus)
# Read kernels
###CAM5Kernel
ds=nc.Dataset(kernelfolder+'q.kernel.plev.nc')
q_LW_kernel=ds['FLN'+varend][:];q_SW_kernel=ds['FSN'+varend][:]
q_LW_kernel_clearsky=ds['FLN'+varend+'C'][:];q_SW_kernel_clearsky=ds['FSN'+varend+'C'][:]
###CAM5Kernel
###ERA5Kernel
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_wv_lw_'+nodp+'dp_'+KLVL+'.nc')
# q_LW_kernel=-ds[KLVL+'_all'][:];q_LW_kernel_clearsky=-ds[KLVL+'_clr'][:]
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_wv_sw_'+nodp+'dp_'+KLVL+'.nc')
# q_SW_kernel=ds[KLVL+'_all'][:];q_SW_kernel_clearsky=ds[KLVL+'_clr'][:]
###ERA5Kernel
###Change in Cloud Radiative Effect (CRE) 
## CESM
# variable1='rsdt';variable2='rsut';variable3='rsutcs'
# d_cre_sw = dataanomaly(readinvariable(variable2,stryr,iens=iens)-readinvariable(variable3,stryr,iens=iens))
# # d_cre_sw = dataanomaly(findvariablefile(variable1)[variable1][:]-findvariablefile(variable3)[variable3][:])-dataanomaly(findvariablefile(variable1)[variable1][:]-findvariablefile(variable2)[variable2][:])
# variable1='rlut';variable2='rlutcs'
# d_cre_lw = dataanomaly(readinvariable(variable2,stryr,iens=iens)-readinvariable(variable1,stryr,iens=iens))
## ERA5
ds=nc.Dataset(fn);variable1='mtnswrfcs';variable2='mtnswrf'
if KLVL=='SFC': variable1='msnswrfcs';variable2='msnswrf'
variable1=ds[variable1][12*(stryr-filestryr):12*(endyr-filestryr+1)]
variable2=ds[variable2][12*(stryr-filestryr):12*(endyr-filestryr+1)]
d_cre_sw = dataanomaly(variable1-variable2)
variable1='mtnlwrfcs';variable2='mtnlwrf'
if KLVL=='SFC': variable1='msnlwrfcs';variable2='msnlwrf'
variable1=ds[variable1][12*(stryr-filestryr):12*(endyr-filestryr+1)]
variable2=ds[variable2][12*(stryr-filestryr):12*(endyr-filestryr+1)]
d_cre_lw = dataanomaly(variable2-variable1)




dLW_ts=tas*0;dLW_ts_cs=tas*0;dLW_ta=tas*0;dLW_ta_cs=tas*0;dSW_alb=tas*0;dSW_alb_cs=tas*0
for iyear in range(nyear):
    month1=iyear*12;month2=iyear*12+12
    # Multiply monthly mean TS change by the TS kernels (function of month, lat, lon) (units W/m2)
    dLW_ts[month1:month2]=tas[month1:month2]*ts_kernel
    dLW_ts_cs[month1:month2]=ts_kernel_clearsky*tas[month1:month2]
    # Convolve air temperature kernel with air temperature change
    dLW_ta[month1:month2]=np.nansum(ta[month1:month2]*ta_kernel*pdiff,axis=1)
    dLW_ta_cs[month1:month2]=np.nansum(ta[month1:month2]*ta_kernel_clearsky*pdiff,axis=1)
    dSW_alb[month1:month2]=alb_kernel*alb[month1:month2]
    dSW_alb_cs[month1:month2]=alb_kernel_clearsky*alb[month1:month2]
### Water vapor feedback
# Calculate the change in moisture per degree warming at constant relative humidity.
q1=np.zeros([12,nlev,nlat,nlon])
t1=np.zeros([12,nlev,nlat,nlon])
for imonth in range(12):
    q1[imonth]=np.nanmean(hus[imonth::12],axis=0)
    t1[imonth]=np.nanmean(ta0[imonth::12],axis=0)
qs1 = calcsatspechum(t1,p);qs1[qs1<0]=np.nan
rh = q1/qs1;rh[rh<0]=np.nan
qs2 = calcsatspechum(ta0,p);qs1[qs1<0]=np.nan
dLW_q=tas*0;dSW_q=tas*0;dLW_q_cs=tas*0;dSW_q_cs=tas*0
for iyear in range(nyear):
    month1=iyear*12;month2=iyear*12+12
    dqdt = (qs2[month1:month2] - qs1)/ta[month1:month2]*rh
    # Normalize kernels by the change in moisture for 1 K warming at constant RH; Convolve moisture kernel with change in moisture
    dLW_q[month1:month2]=np.nansum(q_LW_kernel/dqdt*dq[month1:month2]*pdiff,axis=1)
    dSW_q[month1:month2]=np.nansum(q_SW_kernel/dqdt*dq[month1:month2]*pdiff,axis=1)
    dLW_q_cs[month1:month2]=np.nansum(q_LW_kernel_clearsky/dqdt*dq[month1:month2]*pdiff,axis=1)
    dSW_q_cs[month1:month2]=np.nansum(q_SW_kernel_clearsky/dqdt*dq[month1:month2]*pdiff,axis=1)
    
    

# #Take the area average; incorporate the part due to surface temperature change itself 
# dLW_planck_areamean_12mmean=moving_average(areaavg_lat(-dLW_planck-tas,lat,latmin), 12)
# # Take the area average 
# dLW_lapserate_areamean_12mmean=moving_average(areaavg_lat(-dLW_lapserate,lat,latmin), 12)
# dSW_alb_areamean_12mmean=moving_average(areaavg_lat(dSW_alb,lat,latmin), 12)
clev=0.05#confidence level
# seasonal
m1=0;m2=12;nm=m2-m1
if m2>12:
    tmp=seasonget(tas1,m1,m2-12,nyear)
else:
    tmp=seasonget(tas1,m1,m2,nyear)
tas_areamean_smean=moving_average(areaavg_lat(tmp,lat,latmin),nm)
tas_glbmean_smean=moving_average(areaavg_lat(tmp,lat,glblatlim),nm)
nmonth=len(tas_areamean_smean)
tas_zonalmean_smean=np.zeros([nmonth,nlat])
for ilat in range(nlat):
    tas_zonalmean_smean[:,ilat]=moving_average(np.nanmean(tmp[:,ilat,:],axis=-1),nm)
iAA=0
iwndw=1   
window_size=11+iwndw*10;mvtrndgap=int((window_size-1)/2)
####the AA index as the ratio of the 21-year moving trend centered on the year of interest of the Arctic surface temperature anomaly to the 21-year moving trend of global temperature anomaly.
#the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
refyr1=1951;refyr2=1980;
if iAA>1 and iAA<5:    
    endyraa=endyr-mvtrndgap;nyearaa=endyraa-stryr+1
    ds=nc.Dataset(fn); variable='t2m';tmp1 = dataanomaly(ds[variable][12*(stryr-mvtrndgap-filestryr):12*(endyraa+mvtrndgap-filestryr+1)])[:]###ERA5
    # variable='tas';    tmp1=dataanomaly(readinvariable(variable,stryr-mvtrndgap,endyraa+mvtrndgap,filestryr,iens))  ###CESM   
    if m2>12:
        tasaa=seasonget(tmp1,m1,m2-12,nyearaa+window_size-1)
    else:
        tasaa=seasonget(tmp1,m1,m2,nyearaa+window_size-1)
    del tmp1
elif iAA==5:
    ds=nc.Dataset(fn); variable='t2m';tmp1 = ds[variable][12*(refyr1-filestryr):12*(refyr2-filestryr+1)][:]###ERA5
    # variable='tas';    tmp1=readinvariable(variable,refyr1,refyr2,filestryr,iens)  ###CESM
    if m2>12:
        tasaa=seasonget(tmp1,m1,m2-12,refyr2-refyr1+1)
    else:
        tasaa=seasonget(tmp1,m1,m2,refyr2-refyr1+1)
    del tmp1
if iAA==1:
    tas_glbmean_smean=moving_average(areaavg_lat(tmp,lat,glblatlim),nm)
    aa_areamean_smean=tas_areamean_smean-tas_glbmean_smean #AA1
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=tas_zonalmean_smean[:,ilat]-tas_glbmean_smean 
elif iAA>1 and iAA<5:        
    t_areaavgaa=areaavg_lat(tasaa,lat,latlim=latmin)
    t_zonalavgaa=np.nanmean(tasaa,axis=-1)
    t_glbavgaa=areaavg_lat(tasaa,lat,latlim=glblatlim);del tasaa
    indexyr=np.arange(nm*window_size)
    tmpaa=np.zeros([nm*nyearaa-nm+1])
    tmpaa_zonal=np.zeros([nm*nyearaa-nm+1,nlat])
    for imonth in range(nm*nyearaa-nm+1):
        tmp=imonth+nm*window_size
        tmpglb=t_glbavgaa[imonth:tmp];tmparea=t_areaavgaa[imonth:tmp]
        tmpzonal=t_zonalavgaa[imonth:tmp]
        if iAA==2:
            glbtrnd=statslinregressts_slopepvalue(indexyr,tmpglb)
            areatrnd=statslinregressts_slopepvalue(indexyr,tmparea)
            tmpaa[imonth]=areatrnd[0]/glbtrnd[0]
            if glbtrnd[1]>clev or areatrnd[1]>clev:            tmpaa[imonth]=np.nan
            for ilat in range(nlat):
                areatrnd=statslinregressts_slopepvalue(indexyr,tmpzonal[:,ilat])
                tmpaa_zonal[imonth,ilat]=areatrnd[0]/glbtrnd[0]
                if glbtrnd[1]>clev or areatrnd[1]>clev:            tmpaa_zonal[imonth,ilat]=np.nan
        elif iAA==3:
            tmpaa[imonth]=np.std(tmparea)/np.std(tmpglb)
            for ilat in range(nlat):
                tmpaa_zonal[imonth,ilat]=np.std(tmpzonal[:,ilat])/np.std(tmpglb)
        else:
            tmpaa[imonth]=statslinregressts_slopepvalue(tmpglb,tmparea)[0]
            for ilat in range(nlat):
                tmpaa_zonal[imonth,ilat]=statslinregressts_slopepvalue(tmpglb,tmpzonal[:,ilat])[0]
        #(Trends of standardised SAT anomalies in Arctic)/(global trends of standardised SAT anomalies)
    aa_areamean_smean=moving_average(tmpaa,nm)
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=moving_average(tmpaa_zonal[:,ilat],nm)
elif iAA==5:        
    if m2>12:
        tmp=seasonget(tas0,m1,m2-12,nyear)
    else:
        tmp=seasonget(tas0,m1,m2,nyear)
    glbano=areaavg_lat(tmp,lat,glblatlim)-np.nanmean(areaavg_lat(tasaa,lat,glblatlim))
    aa_areamean_smean=moving_average((areaavg_lat(tmp,lat,latmin)-np.nanmean(areaavg_lat(tasaa,lat,latmin)))/glbano,nm)#AA5
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=moving_average((np.nanmean(tmp[:,ilat,:],axis=-1)-np.nanmean(np.nanmean(tasaa[:,ilat,:],axis=-1)))/glbano,nm)#AA5
    del tasaa
del tmp
tas_smean=seasonal_rolling_mean_along_axis(tas1,m1,m2,ax=0)
dLW_planck_smean=seasonal_rolling_mean_along_axis(-dLW_planck-tas,m1,m2,ax=0)
dLW_lapserate_smean=seasonal_rolling_mean_along_axis(-dLW_lapserate,m1,m2,ax=0)
dSW_alb_smean=seasonal_rolling_mean_along_axis(dSW_alb,m1,m2,ax=0)
dLWSW_smean=seasonal_rolling_mean_along_axis(dLWSW,m1,m2,ax=0)


### Water vapor feedback
## Add the LW and SW responses. Note the sign convention difference between LW and SW!
dLW_q_smean=seasonal_rolling_mean_along_axis(-dLW_q,m1,m2,ax=0)
dSW_q_smean=seasonal_rolling_mean_along_axis(dSW_q,m1,m2,ax=0)
dR_q_smean=dLW_q_smean+dSW_q_smean




# Cloud masking of radiative forcing
if kernel=='':
    ds=nc.Dataset(kernelfolder+'ghg.forcing.nc')
    ds1=nc.Dataset(kernelfolder+'aerosol.forcing.nc')
    cloud_masking_of_forcing_sw=ds['FSNTC'][:]-ds['FSNT'][:]+ds1['FSNTC'][:]-ds1['FSNT'][:]
    cloud_masking_of_forcing_lw=ds['FLNTC'][:]-ds['FLNT'][:]+ds1['FLNTC'][:]-ds1['FLNT'][:]
##Cloud feedback. 
### CRE + cloud masking of radiative forcing + corrections for each feedback
dLW_cloud=-d_cre_lw+(dLW_q_cs-dLW_q)+(dLW_ta_cs-dLW_ta)+(dLW_ts_cs-dLW_ts)
dSW_cloud=-d_cre_sw+(dSW_q_cs-dSW_q)+(dSW_alb_cs-dSW_alb)
if kernel=='':
    for iyear in range(nyear):
        month1=iyear*12;month2=iyear*12+12
        dLW_cloud[month1:month2]+=cloud_masking_of_forcing_lw
        dSW_cloud[month1:month2]+=cloud_masking_of_forcing_sw
        
        
        
        

clev=0.05#confidence level
m1=0;m2=12
nm=m2-m1
tmp=np.zeros([nyear*nm,nlat,nlon])
for iyear in range(nyear):
    if m2>12:
        tmp[iyear*nm:iyear*nm+m2-12]=tas1[iyear*12:iyear*12+m2-12]
        tmp[iyear*nm+m2-12:iyear*nm+nm]=tas1[iyear*12+m1:iyear*12+12]
    else:
        tmp[iyear*nm:iyear*nm+nm]=tas1[iyear*12+m1:iyear*12+m2]
tas_areamean_smean=moving_average(areaavg_lat(tmp,lat,latmin),nm)
tas_glbmean_smean=moving_average(areaavg_lat(tmp,lat,glblatlim),nm)
nmonth=len(tas_areamean_smean)
tas_zonalmean_smean=np.zeros([nmonth,nlat])
for ilat in range(nlat):
    tas_zonalmean_smean[:,ilat]=moving_average(np.nanmean(tmp[:,ilat,:],axis=-1),nm)
#the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
iAA=0
iwndw=1   
window_size=11+iwndw*10;mvtrndgap=int((window_size-1)/2)
refyr1=1951;refyr2=1980
if iAA>1 and iAA<5:
    endyraa=endyr-mvtrndgap;nyearaa=endyraa-stryr+1
    ds=nc.Dataset(fn); variable='t2m';tmp1 = dataanomaly(ds[variable][12*(stryr-mvtrndgap-filestryr):12*(endyraa+mvtrndgap-filestryr+1)])[:]###ERA5
    # variable='tas';    tmp1=dataanomaly(readinvariable(variable,stryr-mvtrndgap,endyraa+mvtrndgap,filestryr,iens))  ###CESM   
    if m2>12:
        tasaa=seasonget(tmp1,m1,m2-12,nyearaa+window_size-1)
    else:
        tasaa=seasonget(tmp1,m1,m2,nyearaa+window_size-1)
    del tmp1
elif iAA==5:
    ds=nc.Dataset(fn); variable='t2m';tmp1 = ds[variable][12*(refyr1-filestryr):12*(refyr2-filestryr+1)][:]###ERA5
    # variable='tas';    tmp1=readinvariable(variable,refyr1,refyr2,filestryr,iens)  ###CESM
    if m2>12:
        tasaa=seasonget(tmp1,m1,m2-12,refyr2-refyr1+1)
    else:
        tasaa=seasonget(tmp1,m1,m2,refyr2-refyr1+1)
    del tmp1
if iAA==1:
    tas_glbmean_smean=moving_average(areaavg_lat(tmp,lat,glblatlim),nm)
    aa_areamean_smean=tas_areamean_smean-tas_glbmean_smean #AA1
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=tas_zonalmean_smean[:,ilat]-tas_glbmean_smean 
elif iAA>1 and iAA<5:        
    t_areaavgaa=areaavg_lat(tasaa,lat,latlim=latmin)
    t_zonalavgaa=np.nanmean(tasaa,axis=-1)
    t_glbavgaa=areaavg_lat(tasaa,lat,latlim=glblatlim);del tasaa
    indexyr=np.arange(0,nm*window_size)
    tmpaa=np.zeros([nm*nyearaa-nm+1])
    tmpaa_zonal=np.zeros([nm*nyearaa-nm+1,nlat])
    for imonth in range(nm*nyearaa-nm+1):
        tmp=imonth+nm*window_size
        tmpglb=t_glbavgaa[imonth:tmp];tmparea=t_areaavgaa[imonth:tmp]
        tmpzonal=t_zonalavgaa[imonth:tmp]
        if iAA==2:
            glbtrnd=statslinregressts_slopepvalue(indexyr,tmpglb)
            areatrnd=statslinregressts_slopepvalue(indexyr,tmparea)
            tmpaa[imonth]=areatrnd[0]/glbtrnd[0]
            if glbtrnd[1]>clev or areatrnd[1]>clev:            tmpaa[imonth]=np.nan
            for ilat in range(nlat):
                areatrnd=statslinregressts_slopepvalue(indexyr,tmpzonal[:,ilat])
                tmpaa_zonal[imonth,ilat]=areatrnd[0]/glbtrnd[0]
                if glbtrnd[1]>clev or areatrnd[1]>clev:            tmpaa_zonal[imonth,ilat]=np.nan
        elif iAA==3:
            tmpaa[imonth]=np.std(tmparea)/np.std(tmpglb)
            for ilat in range(nlat):
                tmpaa_zonal[imonth,ilat]=np.std(tmpzonal[:,ilat])/np.std(tmpglb)
        else:
            tmpaa[imonth]=statslinregressts_slopepvalue(tmpglb,tmparea)[0]
            for ilat in range(nlat):
                tmpaa_zonal[imonth,ilat]=statslinregressts_slopepvalue(tmpglb,tmpzonal[:,ilat])[0]
        #(Trends of standardised SAT anomalies in Arctic)/(global trends of standardised SAT anomalies)
    aa_areamean_smean=moving_average(tmpaa,nm)
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=moving_average(tmpaa_zonal[:,ilat],nm)
elif iAA==5:        
    if m2>12:
        tmp=seasonget(tas0,m1,m2-12,nyear)
    else:
        tmp=seasonget(tas0,m1,m2,nyear)
    glbano=areaavg_lat(tmp,lat,glblatlim)-np.nanmean(areaavg_lat(tasaa,lat,glblatlim))
    aa_areamean_smean=moving_average((areaavg_lat(tmp,lat,latmin)-np.nanmean(areaavg_lat(tasaa,lat,latmin)))/glbano,nm)#AA5
    nmonth=len(aa_areamean_smean)
    aa_zonalmean_smean=np.zeros([nmonth,nlat])
    for ilat in range(nlat):
        aa_zonalmean_smean[:,ilat]=moving_average((np.nanmean(tmp[:,ilat,:],axis=-1)-np.nanmean(np.nanmean(tasaa[:,ilat,:],axis=-1)))/glbano,nm)#AA5
    del tasaa
del tmp

tas_smean=seasonal_rolling_mean_along_axis(tas1,m1,m2,ax=0)
dLW_cloud_smean=seasonal_rolling_mean_along_axis(-dLW_cloud,m1,m2,ax=0)
dSW_cloud_smean=seasonal_rolling_mean_along_axis(dSW_cloud,m1,m2,ax=0)
dR_cloud_smean=dLW_cloud_smean+dSW_cloud_smean




AvgTas='';model='ERA5'#'CESMr'+str(iens)+'i1p1f1';ERA5
Detrend=False
if Detrend: Detrendtxt='Detrend'
else: Detrendtxt=''
seasons='JFMAMJJASOND';letters='abcdefghijk'
if nm>11:
    season='Annual'
elif m2>12:
    season=seasons[m1:]+seasons[:m2-12]
else:
    season=seasons[m1:m2]
clev=.05#confidence level
res_cloudlw=np.zeros([3,nlat]);res_cloudsw=np.zeros([3,nlat]);res_cloud=np.zeros([3,nlat])
# if iAA<1:#AvgTas
#     # tmp=tas_areamean_smean  ##AvgTas
#     tmp=tas_glbmean_smean  ##AvgGlbTas
# else:
#     tmp=aa_areamean_smean
# nmonth=len(tmp)
for ilat in range(nlat):
    if iAA<1:
        tmp=np.nanmean(tas_smean[:,ilat,:],axis=1)
    else:
        tmp=aa_zonalmean_smean[:,ilat]
    nmonth=len(tmp)
    if Detrend:
        tmp=nandetrend(tmp);        
        res_cloudlw[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dLW_cloud_smean[:nmonth,ilat,:],axis=-1)))
        res_cloudsw[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dSW_cloud_smean[:nmonth,ilat,:],axis=-1)))
        res_cloud[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dR_cloud_smean[:nmonth,ilat,:],axis=-1)))
    else:
        res_cloudlw[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dLW_cloud_smean[:nmonth,ilat,:],axis=-1))
        res_cloudsw[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dSW_cloud_smean[:nmonth,ilat,:],axis=-1))
        res_cloud[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dR_cloud_smean[:nmonth,ilat,:],axis=-1))
del tmp
if iAA<1:###iAA=0 means feedback on temperature
    np.savetxt('FeedbackCloudZonalMean'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[0],res_cloudlw[0],res_cloud[0]))
    np.savetxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[1],res_cloudlw[1],res_cloud[1]))
    np.savetxt('FeedbackCloudZonalMeanRsquared'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[2],res_cloudlw[2],res_cloud[2]))
elif iAA<2:
    np.savetxt('FeedbackCloudZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[0],res_cloudlw[0],res_cloud[0]))
    np.savetxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[1],res_cloudlw[1],res_cloud[1]))
    np.savetxt('FeedbackCloudZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[2],res_cloudlw[2],res_cloud[2]))
elif iAA<5:
    np.savetxt('FeedbackCloudZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[0],res_cloudlw[0],res_cloud[0]))
    np.savetxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[1],res_cloudlw[1],res_cloud[1]))
    np.savetxt('FeedbackCloudZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[2],res_cloudlw[2],res_cloud[2]))
else:
    np.savetxt('FeedbackCloudZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[0],res_cloudlw[0],res_cloud[0]))
    np.savetxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[1],res_cloudlw[1],res_cloud[1]))
    np.savetxt('FeedbackCloudZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_cloudsw[2],res_cloudlw[2],res_cloud[2]))
