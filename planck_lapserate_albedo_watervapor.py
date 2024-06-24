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


stryr=1980;endyr=2022 ###use -1 if you want the whole dataset
latmin=70;glblatlim=-91
if latmin<-90:
    area='Global'
elif latmin==70:
    area='' ##Arctic
else:
    area='Lat'+str(latmin)
iens=0 ###just for format purpose
# ##CESM
# iens=11
# variable='tas'
# # fn = '/global/cfs/cdirs/m3522/cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/Amon/tas/gn/v20190308/tas_Amon_CESM2_historical_r1i1p1f1_gn_000101-009912.nc'
# tas0 = readinvariable(variable,stryr,iens=iens); 
# ds=findvariablefile(variable,iens=iens)
# filestryr=1850
##ERA5
# fn = 'e5.sfc.t2m_ssr_ssrd_tisr_tsr_tsrc_ttr_ttrc.19402022.192x288.nc'
fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnlwrfcs_msnswrf_msnswrfcs_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402023.192x288.nc'###CAM5 kernels
# fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnswrf_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402022.73x144.nc' ###ERA5,ERAI kernels
# fn = '/pscratch/sd/h/huoyilin/e5.sfc.t2m_msdwswrf_msnlwrf_msnswrf_mtnlwrf_mtnlwrfcs_mtnswrf_mtnswrfcs.19402022.90x144.nc' ###CloudSat kernels
filestryr=1940;filestryr1=1951
ds=nc.Dataset(fn)
tas0 = ds['t2m'][12*(stryr-filestryr):12*(endyr-filestryr+1)]

tas = dataanomaly(tas0)
lat=ds['lat'][:];lon=ds['lon'][:]
if iens>0: lat=xr.DataArray.to_numpy(lat);lon=xr.DataArray.to_numpy(lon)##CESM2
tas1=tas+0
# variable='ta'##CESM
# ta0 = readinvariable(variable,stryr,endyr,filestryr=filestryr,iens=iens)#[:,:-toplevunused] #filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
# ds = findvariablefile(variable,iens=iens)
# p = ds['plev'][:]/100;p=xr.DataArray.to_numpy(p)#[:-toplevunused]/100  ##CESM2
variable='t'##ERA5
ta0 = readinvariable(variable,stryr,endyr,filestryr=filestryr1,iens=iens)#[:,:-toplevunused] #filestryr=1951 (filestryr1) for ERA5;1850 (filestryr) for CESM2
ds = findvariablefile(variable,iens=iens)
p = ds['level'][:]  ###ERA5

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
kernel='' ###'' for using the CAM5 kernel
kernelfolder='/global/cfs/cdirs/m1199/huoyilin/cscratch/CAM5RadiativeKernels/'
if KLVL=='SFC':varend='S'
else:varend='T'
ds=nc.Dataset(kernelfolder+'dp_plev.nc')
pdiff=ds['dp'][:]/100#[:,:-toplevunused]/100
### Planck, WV & Alb 
ds=nc.Dataset(kernelfolder+'ts.kernel.nc')
ts_kernel=ds['FLN'+varend][:]
ds=nc.Dataset(kernelfolder+'t.kernel.plev.nc')
ta_kernel=ds['FLN'+varend][:]#[:,:-toplevunused]
###CAM5 Kernel
###ERA5, ERAIKernel
# kernel='ERA5kernel' ###ERAIKernel
# kernelfolder='/pscratch/sd/h/huoyilin/ERA5_kernels/'
# ds=nc.Dataset(kernelfolder+'thickness_normalized_ta_wv_kernel/dp_era5.nc')
# pdiff=ds['dp'][:]/100
###ERA5, ERAIKernel
# kernel='CloudSatKernel'
# kernelfolder='/pscratch/sd/h/huoyilin/CloudSat_kernels/'+KLVL+'/CloudSat/'
# ds=nc.Dataset(kernelfolder+KLVL+'_CloudSat_Kerns.nc')
# ts_kernel=-ds['lw_ts'][:]
# ta_kernel=-ds['lw_t'][:]
# alb_kernel=ds['sw_a'][:].data;alb_kernel[alb_kernel>9e36]=np.nan
# q_LW_kernel=-ds['lw_q'][:]
# q_SW_kernel=ds['sw_q'][:]
# tmp=ds['plev'][:]/100
# pdiff=ta_kernel+np.nan
# pdiff[:,0,:,:]=ds['PS'][:]/100-tmp[0]
# pdiff[:,-1,:,:]=tmp[-1]
# for ilev in range (1,nlev-1):
#     pdiff[:,ilev,:,:]=(tmp[ilev-1]-tmp[ilev+1])/2
# ###CloudSatKernel

###ERA5Kernel
# ds=nc.Dataset(kernelfolder+'ERA5_kernel_ts_'+KLVL+'.nc')
# ts_kernel=-ds[KLVL+'_all'][:]
# dpnodp='thickness_normalized';nodp=''
# if KLVL=='SFC': dpnodp='layer_specified';nodp='no';pdiff[:]=1
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_ta_'+nodp+'dp_'+KLVL+'.nc')
# ta_kernel=-ds[KLVL+'_all'][:]
###ERA5Kernel
###ERAIKernel
# klvl1=KLVL.lower()
# ds=nc.Dataset(kernelfolder+'other_kernels/kernel_cloudsat_res2.5_level37_toa_sfc.nc')##cloudsat,erai
# ts_kernel=-ds['ts_'+klvl1+'_all'][:]
# ta_kernel=-ds['ta_'+klvl1+'_all'][:]
# alb_kernel=ds['alb_'+klvl1+'_all'][:]
# q_LW_kernel=-ds['wv_lw_'+klvl1+'_all'][:]
# q_SW_kernel=ds['wv_sw_'+klvl1+'_all'][:]
###ERAIKernel
### Albedo feedback
## Collect surface shortwave radiation fields for calculating albedo change
# variable1='rsus';variable2='rsds' ### CESM
# alb=dataanomaly(readinvariable(variable1,stryr,endyr,filestryr=filestryr,iens=iens)/readinvariable(variable2,stryr,endyr,filestryr=filestryr,iens=iens))*100 ###CESM
# totaltoa=dataanomaly(readinvariable('rsdt',stryr,endyr,filestryr=filestryr,iens=iens)-readinvariable('rsut',stryr,endyr,filestryr=filestryr,iens=iens)-readinvariable('rlut',stryr,endyr,filestryr=filestryr,iens=iens))
ds=nc.Dataset(fn);variable1=ds['msdwswrf'][12*(stryr-filestryr):12*(endyr-filestryr+1)]### ERA5
alb=(variable1-ds['msnswrf'][12*(stryr-filestryr):12*(endyr-filestryr+1)])/variable1; del variable1 
alb[alb<0]=np.nan;alb[alb>1]=np.nan;alb=dataanomaly(alb)*100###ERA5
if KLVL=='SFC':totaltoa=dataanomaly((ds['msnlwrf'][:]+ds['msnswrf'][:])[12*(stryr-filestryr):12*(endyr-filestryr+1)])
else:totaltoa=dataanomaly((ds['mtnlwrf'][:]+ds['mtnswrf'][:])[12*(stryr-filestryr):12*(endyr-filestryr+1)])
# Read TOA albedo kernel
ds=nc.Dataset(kernelfolder+'alb.kernel.nc')###CAM5Kernel
alb_kernel=ds['FSN'+varend][:]###CAM5Kernel
# ds=nc.Dataset(kernelfolder+'ERA5_kernel_alb_'+KLVL+'.nc')###ERA5Kernel
# alb_kernel=ds[KLVL+'_all'][:]###ERA5Kernel

### Water vapor feedback
## Calculate the change in moisture per degree warming at constant relative humidity.
t1=np.zeros([12,nlev,nlat,nlon])
for imonth in range(12):
    t1[imonth]=np.nanmean(ta0[imonth::12],axis=0)
# variable='hus'##CESM
# hus = readinvariable(variable,stryr,endyr,filestryr=filestryr,iens=iens)#[:,:-toplevunused]#filestryr=1951 for ERA5;1850 for cesm
variable='q'##ERA5
hus = readinvariable(variable,stryr,endyr,filestryr=filestryr1,iens=iens)#[:,:-toplevunused]#filestryr=1951 for ERA5;1850 for cesm
qs1 = calcsatspechum(t1,p);del t1;qs1[qs1<0]=np.nan
qs2 = calcsatspechum(ta0,p);del ta0;qs2[qs2<0]=np.nan
if verticalflip:    hus = np.flip(hus,axis=1)
q1=np.zeros([12,nlev,nlat,nlon])
for imonth in range(12):
    q1[imonth]=np.nanmean(hus[imonth::12],axis=0)
rh = q1/qs1
# the change in moisture
dq=dataanomaly(hus);del hus
# Read kernels
ds=nc.Dataset(kernelfolder+'q.kernel.plev.nc')
q_LW_kernel=ds['FLN'+varend][:]#[:,:-toplevunused]
q_SW_kernel=ds['FSN'+varend][:]#[:,:-toplevunused]
###ERA5Kernel
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_wv_lw_'+nodp+'dp_'+KLVL+'.nc')
# q_LW_kernel=-ds[KLVL+'_all'][:]
# ds=nc.Dataset(kernelfolder+dpnodp+'_ta_wv_kernel/ERA5_kernel_wv_sw_'+nodp+'dp_'+KLVL+'.nc')
# q_SW_kernel=ds[KLVL+'_all'][:]
###ERA5Kernel




dLW_planck=tas*0
dLW_lapserate=tas*0
dSW_alb=tas*0
dLWSW=tas*0
# Project surface temperature change into height
dts3d=np.tile(tas,(nlev,1,1,1)).transpose(1,0,2,3)
if KLVL=='SFC':dts3d[:]=0
# Calculate the departure of temperature change from the surface temperature change
dt_lapserate=(ta-dts3d)#.*(p>=p_tropopause)
for iyear in range(nyear):
    month1=iyear*12;month2=iyear*12+12
    # Multiply monthly mean TS change by the TS kernels (function of month, lat, lon) (units W/m2)
    tas[month1:month2]*=ts_kernel
    # # Convolve air temperature kernel with air temperature change
    dLW_planck[month1:month2]=np.nansum(ta_kernel*dts3d[month1:month2]*pdiff,axis=1)
    #Convolve air temperature kernel with 3-d surface air temp change
    dLW_lapserate[month1:month2]=np.nansum(ta_kernel*dt_lapserate[month1:month2]*pdiff,axis=1)
    dSW_alb[month1:month2]=alb_kernel*alb[month1:month2]
    dLWSW[month1:month2]=totaltoa[month1:month2] 
del dts3d
if KLVL=='SFC':dLW_planck=tas+0
dLW_q=tas*0;dSW_q=tas*0
for iyear in range(nyear):
    month1=iyear*12;month2=iyear*12+12
    dqdt = (qs2[month1:month2] - qs1)/ta[month1:month2]*rh
    ###dlogqdt;dlogq;Uses the fractional change approximation of logarithms in the specific humidity response & normalization factor
    dqdt=dqdt/q1; dq[month1:month2]=dq[month1:month2]/q1 
    # Normalize kernels by the change in moisture for 1 K warming at constant RH; Convolve moisture kernel with change in moisture
    dLW_q[month1:month2]=np.nansum(q_LW_kernel/dqdt*dq[month1:month2]*pdiff,axis=1)
    dSW_q[month1:month2]=np.nansum(q_SW_kernel/dqdt*dq[month1:month2]*pdiff,axis=1)   
del qs1,qs2,dqdt,q1




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




AvgTas='';model='ERA5'#'CESMr'+str(iens)+'i1p1f1'
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
clev=0.05#confidence level
res_planck=np.zeros([3,nlat]);res_lapserate=np.zeros([3,nlat]);res_alb=np.zeros([3,nlat]);res_ttl=np.zeros([3,nlat])
res_qlw=np.zeros([3,nlat]);res_qsw=np.zeros([3,nlat]);res_q=np.zeros([3,nlat])
# if iAA<1:#AvgTas
#     tmp=tas_areamean_smean  ##AvgTas
#     # tmp=tas_glbmean_smean  ##AvgGlbTas
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
        tmp=nandetrend(tmp)
        res_planck[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dLW_planck_smean[:nmonth,ilat,:],axis=-1)))
        res_lapserate[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dLW_lapserate_smean[:nmonth,ilat,:],axis=-1)))
        res_alb[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dSW_alb_smean[:nmonth,ilat,:],axis=-1)))
        res_qlw[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dLW_q_smean[:nmonth,ilat,:],axis=-1)))
        res_qsw[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dSW_q_smean[:nmonth,ilat,:],axis=-1)))
        res_q[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dR_q_smean[:nmonth,ilat,:],axis=-1)))
        res_ttl[:,ilat]=statslinregressts_slopepvalue(tmp,nandetrend(np.nanmean(dLWSW_smean[:nmonth,ilat,:],axis=-1)))
    else:    
        res_planck[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dLW_planck_smean[:nmonth,ilat,:],axis=-1))
        res_lapserate[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dLW_lapserate_smean[:nmonth,ilat,:],axis=-1))
        res_alb[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dSW_alb_smean[:nmonth,ilat,:],axis=-1))
        res_qlw[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dLW_q_smean[:nmonth,ilat,:],axis=-1))
        res_qsw[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dSW_q_smean[:nmonth,ilat,:],axis=-1))
        res_q[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dR_q_smean[:nmonth,ilat,:],axis=-1))
        res_ttl[:,ilat]=statslinregressts_slopepvalue(tmp,np.nanmean(dLWSW_smean[:nmonth,ilat,:],axis=-1))
del tmp
if iAA<1:###iAA=0 means feedback on temperature
    np.savetxt('FeedbackWaterVaporZonalMean'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[0],res_qlw[0],res_q[0]))
    np.savetxt('PlanckLapseAlbTotalZonalMean'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[0],res_lapserate[0],res_alb[0],res_ttl[0]))
    np.savetxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[1],res_qlw[1],res_q[1]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[1],res_lapserate[1],res_alb[1],res_ttl[1]))
    np.savetxt('FeedbackWaterVaporZonalMeanRsquared'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[2],res_qlw[2],res_q[2]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanRsquared'+season+AvgTas+area+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[2],res_lapserate[2],res_alb[2],res_ttl[2]))
elif iAA<2:
    np.savetxt('FeedbackWaterVaporZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[0],res_qlw[0],res_q[0]))
    np.savetxt('PlanckLapseAlbTotalZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[0],res_lapserate[0],res_alb[0],res_ttl[0]))
    np.savetxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[1],res_qlw[1],res_q[1]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[1],res_lapserate[1],res_alb[1],res_ttl[1]))
    np.savetxt('FeedbackWaterVaporZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[2],res_qlw[2],res_q[2]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[2],res_lapserate[2],res_alb[2],res_ttl[2]))
elif iAA<5:
    np.savetxt('FeedbackWaterVaporZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[0],res_qlw[0],res_q[0]))
    np.savetxt('PlanckLapseAlbTotalZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[0],res_lapserate[0],res_alb[0],res_ttl[0]))
    np.savetxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[1],res_qlw[1],res_q[1]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[1],res_lapserate[1],res_alb[1],res_ttl[1]))
    np.savetxt('FeedbackWaterVaporZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[2],res_qlw[2],res_q[2]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[2],res_lapserate[2],res_alb[2],res_ttl[2]))
else:
    np.savetxt('FeedbackWaterVaporZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[0],res_qlw[0],res_q[0]))
    np.savetxt('PlanckLapseAlbTotalZonalMean'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[0],res_lapserate[0],res_alb[0],res_ttl[0]))    
    np.savetxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[1],res_qlw[1],res_q[1]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[1],res_lapserate[1],res_alb[1],res_ttl[1]))
    np.savetxt('FeedbackWaterVaporZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_qsw[2],res_qlw[2],res_q[2]))
    np.savetxt('PlanckLapseAlbTotalZonalMeanRsquared'+season+AvgTas+area+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out', (res_planck[2],res_lapserate[2],res_alb[2],res_ttl[2]))
