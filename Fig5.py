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
kernels=['ERA5kernel','CloudSatkernel','','ERAIkernel'];nkernel=len(kernels)
KLVLs=['TOA','TOA','','TOA'] ###TOA or SFC kernel;'' for using the CAM5 kernel
kernelfolders=['/pscratch/sd/h/huoyilin/ERA5_kernels/','/pscratch/sd/h/huoyilin/CloudSat_kernels/'+KLVLs[1]+'/CloudSat/','/global/cfs/cdirs/m1199/huoyilin/cscratch/CAM5RadiativeKernels/','/pscratch/sd/h/huoyilin/ERA5_kernels/other_kernels/']
kernelfiles=['ERA5_kernel_ts_'+KLVLs[0]+'.nc','TOA_CloudSat_Kerns.nc','t.kernel.nc','kernel_erai_res2.5_level37_'+KLVLs[3].lower()+'_sfc.nc']
latnames=['latitude','latitude','lat','lat']
AvgTas='';model='ERA5'
Rsquared=''
Detrend=False
if Detrend: Detrendtxt='Detrend'
else: Detrendtxt=''
clev=.05
stryr=1980;endyr=2022;refyr1=1951;refyr2=1980
# iAA=0
seasons='JFMAMJJASOND';letters='abcdefghijk'
m1=0;m2=12;nm=m2-m1
if nm>11:
    season='Annual'
elif m2>12:
    season=seasons[m1:]+seasons[:m2-12]
else:
    season=seasons[m1:m2]
latmin=70;latt1=-30;latt2=30
mglbavg=''
iwndw=1;window_size=11+iwndw*10;mvtrndgap=int((window_size-1)/2);endyraa=endyr-mvtrndgap
fntsz=15;linew=3;alpha=.15;mkedgwd=3;colors=[1,2,3,4,0,5]
nAA=5;ncol=2;fig, axes = plt.subplots(nrows=2, ncols=ncol,figsize=(8, 8), facecolor='w')
for ikernel in range(nkernel):
    kernel=kernels[ikernel];KLVL=KLVLs[ikernel]
    lat=nc.Dataset(kernelfolders[ikernel]+kernelfiles[ikernel])[latnames[ikernel]][:]
    gw=np.cos(np.deg2rad(lat)) #Gaussian weights for the CESM grid
    respavg=np.zeros([5,2])+np.nan;reslavg=np.zeros([5,2])+np.nan;resaavg=np.zeros([5,2])+np.nan;resqavg=np.zeros([5,2])+np.nan;rescavg=np.zeros([5,2])+np.nan;respglbavg=np.zeros([5])+np.nan;resravg=np.zeros([5,2])+np.nan;resclavg=np.zeros([5,2])+np.nan;rescsavg=np.zeros([5,2])+np.nan
    respstd=np.zeros([5,2])+np.nan;reslstd=np.zeros([5,2])+np.nan;resastd=np.zeros([5,2])+np.nan;resqstd=np.zeros([5,2])+np.nan;rescstd=np.zeros([5,2])+np.nan;resrstd=np.zeros([5,2])+np.nan;resclstd=np.zeros([5,2])+np.nan;rescsstd=np.zeros([5,2])+np.nan
    for iAA in range(nAA):
        res_planck=np.zeros([2,nlat]);res_lapserate=np.zeros([2,nlat]);res_alb=np.zeros([2,nlat]);res_ttl=np.zeros([2,nlat])
        res_qlw=np.zeros([2,nlat]);res_qsw=np.zeros([2,nlat]);res_q=np.zeros([2,nlat])
        if iAA<1:###iAA=0 means feedback on temperature
            res_planck,res_lapserate,res_alb,res_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMean'+Rsquared+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            res_qsw,res_qlw,res_q=np.loadtxt('FeedbackWaterVaporZonalMean'+Rsquared+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            res_cloudsw,res_cloudlw,res_cloud=np.loadtxt('FeedbackCloudZonalMean'+Rsquared+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_planck,resp_lapserate,resp_alb,resp_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_qsw,resp_qlw,resp_q=np.loadtxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_cloudsw,resp_cloudlw,resp_cloud=np.loadtxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
        elif iAA<2:
            res_planck,res_lapserate,res_alb,res_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            res_qsw,res_qlw,res_q=np.loadtxt('FeedbackWaterVaporZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            res_cloudsw,res_cloudlw,res_cloud=np.loadtxt('FeedbackCloudZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_planck,resp_lapserate,resp_alb,resp_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_qsw,resp_qlw,resp_q=np.loadtxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_cloudsw,resp_cloudlw,resp_cloud=np.loadtxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+model+Detrendtxt+KLVL+kernel+'.out')
        elif iAA<5:
                res_planck,res_lapserate,res_alb,res_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
                res_qsw,res_qlw,res_q=np.loadtxt('FeedbackWaterVaporZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
                res_cloudsw,res_cloudlw,res_cloud=np.loadtxt('FeedbackCloudZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
                resp_planck,resp_lapserate,resp_alb,resp_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
                resp_qsw,resp_qlw,resp_q=np.loadtxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
                resp_cloudsw,resp_cloudlw,resp_cloud=np.loadtxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyraa)+'window'+str(window_size)+'yr'+model+Detrendtxt+KLVL+kernel+'.out')
        else:
            res_planck,res_lapserate,res_alb,res_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
            res_qsw,res_qlw,res_q=np.loadtxt('FeedbackWaterVaporZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
            res_cloudsw,res_cloudlw,res_cloud=np.loadtxt('FeedbackCloudZonalMean'+Rsquared+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_planck,resp_lapserate,resp_alb,resp_ttl=np.loadtxt('PlanckLapseAlbTotalZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_qsw,resp_qlw,resp_q=np.loadtxt('FeedbackWaterVaporZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
            resp_cloudsw,resp_cloudlw,resp_cloud=np.loadtxt('FeedbackCloudZonalMeanPvalue'+season+AvgTas+'AA'+str(iAA)+str(stryr)+str(endyr)+'ref'+str(refyr1)+str(refyr2)+model+Detrendtxt+KLVL+kernel+'.out')
        res_rsd=res_ttl-res_planck-res_lapserate-res_alb-res_q-res_cloud;resp_rsd=resp_ttl
        ax1=axes.flat[ikernel];
        ax1.set_title('('+letters[iAA]+')',loc='left', fontsize=fntsz)
        if iAA<1:     ax1.set_title('SAT',loc='center', fontsize=fntsz)
        else:ax1.set_title('$A_%d$'%iAA,loc='center', fontsize=fntsz)
        #Area weight
        ressig=(resp_planck<clev)&(resp_lapserate<clev)&(resp_alb<clev)&(resp_q<clev)&(resp_cloud<clev)
        ressig[:] = [True for y in ressig] ###add up regardless significant or not 
        respglbavg[iAA]=np.average(res_planck, weights=gw)
        latweights0=np.where((lat>latmin)&ressig,gw,0)
        respavg[iAA,0]=np.average(res_planck, weights=latweights0)
        reslavg[iAA,0]=np.average(res_lapserate, weights=latweights0)
        resnotnan= [res_alb<100 for y in res_alb]
        latweights=np.where((lat>latmin)&ressig&resnotnan,gw,0)
        res_alb[np.abs(res_alb)>100]=np.nan;ii=~np.isnan(res_alb)
        resaavg[iAA,0]=np.average(res_alb[ii], weights=latweights0[ii])
        resqavg[iAA,0]=np.average(res_q, weights=latweights0)
        rescavg[iAA,0]=np.average(res_cloud, weights=latweights0)
        latweights1=np.where((lat>latt1)&(lat<latt2)&ressig,gw,0)
        respavg[iAA,1]=np.average(res_planck, weights=latweights1)
        reslavg[iAA,1]=np.average(res_lapserate, weights=latweights1)
        resaavg[iAA,1]=np.average(res_alb[ii], weights=latweights1[ii])
        resqavg[iAA,1]=np.average(res_q, weights=latweights1)
        rescavg[iAA,1]=np.average(res_cloud, weights=latweights1)
        res_rsd[np.abs(res_rsd)>100]=np.nan;ii=~np.isnan(res_rsd)
        resravg[iAA,0]=np.average(res_rsd[ii], weights=latweights0[ii])
        resravg[iAA,1]=np.average(res_rsd[ii], weights=latweights1[ii])
    ax1=axes.flat[ikernel];
    ax1.set_title('('+letters[ikernel]+')',loc='left', fontsize=fntsz)
    if kernel=='':ax1.set_title('CAM5', fontsize=fntsz)###CAM5 kernel
    else:    ax1.set_title(kernel[:-6], fontsize=fntsz)
    markers='o^v<s';msz=10;cpsz=3;alpha=.35
    handles_markers = [];markers_labels = []
    for iAA in range(nAA):
        if iAA<1: lbl='SAT'
        else:lbl='$A_%d$'%iAA
        if model[0:4]=='CESM':pts = ax1.errorbar([0], [0], xerr=[0],yerr=[0],marker=markers[iAA], c='k',mfc='none',ms=msz,label=lbl)
        else:pts = ax1.scatter([0], [0],marker=markers[iAA], edgecolors='k', facecolors='none',s=msz*10,label=lbl)
        handles_markers.append(pts);markers_labels.append(lbl);pts.remove()
        marker, caps, bars = ax1.errorbar(respavg[iAA,1]-respglbavg[iAA],respavg[iAA,0]-respglbavg[iAA],yerr=respstd[iAA,0],xerr=respstd[iAA,1],color='C%d'%(colors[0]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
        marker, caps, bars =ax1.errorbar(reslavg[iAA,1],reslavg[iAA,0],xerr=reslstd[iAA,1],yerr=reslstd[iAA,0],color='C%d'%(colors[1]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
        marker, caps, bars =ax1.errorbar(resqavg[iAA,1],resqavg[iAA,0],xerr=resqstd[iAA,1],yerr=resqstd[iAA,0],color='C%d'%(colors[3]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
        marker, caps, bars =ax1.errorbar(rescavg[iAA,1],rescavg[iAA,0],xerr=rescstd[iAA,1],yerr=rescstd[iAA,0],color='C%d'%(colors[4]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
        marker, caps, bars =ax1.errorbar(resaavg[iAA,1],resaavg[iAA,0],xerr=resastd[iAA,1],yerr=resastd[iAA,0],color='C%d'%(colors[2]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
        marker, caps, bars =ax1.errorbar(resravg[iAA,1],resravg[iAA,0],xerr=resrstd[iAA,1],yerr=resrstd[iAA,0],color='C%d'%(colors[5]),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
        [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    ax1.set_xlim([-2,3]);ax1.set_ylim([-2,3])
    minimum = np.min((ax1.get_xlim(),ax1.get_ylim()));maximum = np.max((ax1.get_xlim(),ax1.get_ylim()));ax1.axline([minimum, minimum], [maximum, maximum], color = 'k', linestyle = '--',lw=linew/3)
    ax1.axhline(y = 0, color = 'k', linestyle = '--',lw=linew/3);ax1.axvline(x = 0, color = 'k', linestyle = '--',lw=linew/3)
    if ikernel%ncol<1:ax1.set_ylabel('Arctic feedback', fontsize=fntsz)
    if ikernel>ncol-1:ax1.set_xlabel('Tropical feedback', fontsize=fntsz)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='both', which='minor',length=5);
fig.tight_layout()
plt.show()
