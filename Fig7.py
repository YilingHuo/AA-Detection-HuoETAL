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
KLVL=''###TOA or SFC kernel;'' for using the CAM5 TOA kernel
if KLVL=='SFC': labelp='Tsfc';labellr='Tatm'
else:labelp='P';labellr='LR'
AvgTas='';model='CESM' #CESM;E3SM;ERA5
Rsquared=''
Detrend=False
if Detrend: Detrendtxt='Detrend'
else: Detrendtxt=''
clev=.05
stryr=1980;endyr=2014;refyr1=1951;refyr2=1980
seasons='JFMAMJJASOND';letters='abcdefghijk'
m1=0;m2=12;nm=m2-m1
if nm>11:
    season='Annual'
elif m2>12:
    season=seasons[m1:]+seasons[:m2-12]
else:
    season=seasons[m1:m2]
kernelfolder='/global/cfs/cdirs/m1199/huoyilin/cscratch/CAM5RadiativeKernels/';ds=nc.Dataset(kernelfolder+'t.kernel.nc')
lat=ds['lat'][:];nlat=len(lat);gw=ds['gw'][:] #Gaussian weights for the CESM grid
latmin=70;latt1=-30;latt2=30
mglbavg=''
iwndw=1;window_size=11+iwndw*10;mvtrndgap=int((window_size-1)/2);endyraa=endyr-mvtrndgap
fntsz=15;linew=3;alpha=.15;mkedgwd=3;colors=[1,2,3,4,0,5]
nAA=5;ncol=3;fig, axes = plt.subplots(nrows=2, ncols=ncol,figsize=(16, 11), facecolor='w')
respavg=np.zeros([5,2])+np.nan;reslavg=np.zeros([5,2])+np.nan;resaavg=np.zeros([5,2])+np.nan;resqavg=np.zeros([5,2])+np.nan;rescavg=np.zeros([5,2])+np.nan;respglbavg=np.zeros([5])+np.nan;resravg=np.zeros([5,2])+np.nan;resclavg=np.zeros([5,2])+np.nan;rescsavg=np.zeros([5,2])+np.nan
respstd=np.zeros([5,2])+np.nan;reslstd=np.zeros([5,2])+np.nan;resastd=np.zeros([5,2])+np.nan;resqstd=np.zeros([5,2])+np.nan;rescstd=np.zeros([5,2])+np.nan;resrstd=np.zeros([5,2])+np.nan;resclstd=np.zeros([5,2])+np.nan;rescsstd=np.zeros([5,2])+np.nan
###CESM ebsemble
def resensstd(res_ens,lat):
    nens=np.sum(~np.isnan(res_ens[:,0]))
    tmp=np.zeros([nens,2])+np.nan
    for iens in range(nens):
        tmp[iens,0]=np.average(res_ens[iens], weights=np.where(lat>latmin,gw,0))
        tmp[iens,1]=np.average(res_ens[iens], weights=np.where((lat>latt1)&(lat<latt2),gw,0))
    return np.nanstd(tmp,axis=0)
###CESM ebsemble
for iAA in range(nAA):
    ##CESM ensemble 
    nens=11 
    if iAA==2: nens=10
    res_planck_ens=np.zeros([nens,nlat]);res_planckmglbavg_ens=np.zeros([nens,nlat]);res_lapserate_ens=np.zeros([nens,nlat]);res_alb_ens=np.zeros([nens,nlat]);res_q_ens=np.zeros([nens,nlat]);res_cloud_ens=np.zeros([nens,nlat]);res_rsd_ens=np.zeros([nens,nlat])
    resp_planck_ens=np.zeros([nens,nlat]);resp_lapserate_ens=np.zeros([nens,nlat]);resp_alb_ens=np.zeros([nens,nlat]);resp_q_ens=np.zeros([nens,nlat]);resp_cloud_ens=np.zeros([nens,nlat]);resp_rsd_ens=np.zeros([nens,nlat])
    for iens in range(nens):
        model='CESMr'+str(iens+1)+'i1p1f1'
    ###CESM ensemble 

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
    ####CESM ensemble
            if iAA==2 and iens>1 and iens<5:
                res_planck=res_planck+np.nan;res_lapserate=res_lapserate+np.nan;res_alb=res_alb+np.nan;res_q=res_q+np.nan;res_cloud=res_cloud+np.nan
                resp_planck=resp_planck+np.nan;resp_lapserate=resp_lapserate+np.nan;resp_alb=resp_alb+np.nan;res_q=resp_q+np.nan;resp_cloud=resp_cloud+np.nan
            else:
    #####CESM ensemble
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
# #         #####CESM ensemble
        res_planck_ens[iens]=res_planck;res_lapserate_ens[iens]=res_lapserate;res_alb_ens[iens]=res_alb;res_q_ens[iens]=res_q;res_cloud_ens[iens]=res_cloud
        resp_planck_ens[iens]=resp_planck;resp_lapserate_ens[iens]=resp_lapserate;resp_alb_ens[iens]=resp_alb;resp_q_ens[iens]=resp_q;resp_cloud_ens[iens]=resp_cloud
        res_planckmglbavg_ens[iens]=res_planck-np.average(res_planck, weights=gw)
        res_rsd_ens[iens]=res_rsd;resp_rsd_ens[iens]=resp_ttl
    respstd[iAA]=resensstd(res_planck_ens,lat);reslstd[iAA]=resensstd(res_lapserate_ens,lat);resastd[iAA]=resensstd(res_alb_ens,lat);resqstd[iAA]=resensstd(res_q_ens,lat);rescstd[iAA]=resensstd(res_cloud_ens,lat);rescstd[iAA]=resensstd(res_cloud_ens,lat);resrstd[iAA]=resensstd(res_rsd_ens,lat)
    res_planck=np.nanmean(res_planck_ens,axis=0);res_lapserate=np.nanmean(res_lapserate_ens,axis=0);res_alb=np.nanmean(res_alb_ens,axis=0);res_q=np.nanmean(res_q_ens,axis=0);res_cloud=np.nanmean(res_cloud_ens,axis=0);res_rsd=np.nanmean(res_rsd_ens,axis=0)

    ax1=axes.flat[iAA];
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
    resclavg[iAA,0]=np.average(res_cloudlw, weights=latweights0)    
    resclavg[iAA,1]=np.average(res_cloudlw, weights=latweights1)    
    rescsavg[iAA,0]=np.average(res_cloudsw, weights=latweights0)    
    rescsavg[iAA,1]=np.average(res_cloudsw, weights=latweights1)    
    if model[0:4]=='CESM':
        linep, =ax1.plot(lat,res_planck,lw=linew,color='C%d'%(colors[0]),label=labelp+' '+"%.2f" % (respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_planck_ens, axis=0);ax1.fill_between(lat,res_planck-stddv, res_planck+stddv,color=linep.get_color(),alpha=alpha)
        linel, =ax1.plot(lat,res_lapserate,lw=linew,color='C%d'%(colors[1]),label=labellr+' '+"%.2f" %(reslavg[iAA,0])+" %.0f%%" % (100 * reslavg[iAA,0]/respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_lapserate_ens, axis=0);ax1.fill_between(lat,res_lapserate-stddv, res_lapserate+stddv,color=linel.get_color(),alpha=alpha)
        linea, =ax1.plot(lat,res_alb,lw=linew,color='C%d'%(colors[2]),label='A '+"%.2f" %(resaavg[iAA,0])+" %.0f%%" % (100 * resaavg[iAA,0]/respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_alb_ens, axis=0);ax1.fill_between(lat,res_alb-stddv, res_alb+stddv,color=linea.get_color(),alpha=alpha)
        lineq, =ax1.plot(lat,res_q,lw=linew,color='C%d'%(colors[3]),label='WV '+"%.2f" %(resqavg[iAA,0])+" %.0f%%" % (100 * resqavg[iAA,0]/respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_q_ens, axis=0);ax1.fill_between(lat,res_q-stddv, res_q+stddv,color=lineq.get_color(),alpha=alpha)
        linec, =ax1.plot(lat,res_cloud,lw=linew,color='C%d'%(colors[4]),label='C '+"%.2f" %(rescavg[iAA,0])+" %.0f%%" % (100 * rescavg[iAA,0]/respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_cloud_ens, axis=0);ax1.fill_between(lat,res_cloud-stddv, res_cloud+stddv,color=linec.get_color(),alpha=alpha)
        liner, =ax1.plot(lat,res_rsd,lw=linew,color='C%d'%(colors[5]),label='R '+"%.2f" %(resravg[iAA,0])+" %.0f%%" % (100 * resravg[iAA,0]/respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        stddv=np.nanstd(res_rsd_ens, axis=0);ax1.fill_between(lat,res_rsd-stddv, res_rsd+stddv,color=linec.get_color(),alpha=alpha)
    else:
        ii=(resp_planck<clev);tmp=res_planck+np.nan;tmp[ii]=res_planck[ii]
        linep, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[0]),label=labelp+' '+"%.2f" % (respavg[iAA,0]))#+" %.1f%%" % (100 * respavg/resavg))
        ax1.plot(lat,res_planck,lw=linew,color=linep.get_color(),alpha=alpha*2)#+" %.1f%%" % (100 * respavg/resavg))
        ii=(resp_lapserate<clev);tmp=res_lapserate+np.nan;tmp[ii]=res_lapserate[ii]
        linel, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[1]),label=labellr+' '+"%.2f" %(reslavg[iAA,0])+" %.0f%%" % (100 * reslavg[iAA,0]/respavg[iAA,0]))
        ax1.plot(lat,res_lapserate,lw=linew,color=linel.get_color(),alpha=alpha*2)
        ii=(resp_alb<clev);tmp=res_alb+np.nan;tmp[ii]=res_alb[ii]
        linea, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[2]),label='A '+"%.2f" %(resaavg[iAA,0])+" %.0f%%" % (100 * resaavg[iAA,0]/respavg[iAA,0]))
        ax1.plot(lat,res_alb,lw=linew,color=linea.get_color(),alpha=alpha*2)
        ii=(resp_q<clev);tmp=res_q+np.nan;tmp[ii]=res_q[ii]
        lineq, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[3]),label='WV '+"%.2f" %(resqavg[iAA,0])+" %.0f%%" % (100 * resqavg[iAA,0]/respavg[iAA,0]))
        ax1.plot(lat,res_q,lw=linew,color=lineq.get_color(),alpha=alpha*2)
        ii=(resp_cloud<clev);tmp=res_cloud+np.nan;tmp[ii]=res_cloud[ii]
        linec, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[4]),label='C '+"%.2f" %(rescavg[iAA,0])+" %.0f%%" % (100 * rescavg[iAA,0]/respavg[iAA,0]))
        ax1.plot(lat,res_cloud,lw=linew,color=linec.get_color(),alpha=alpha*2)
        ii=(resp_ttl<clev);tmp=res_rsd+np.nan;tmp[ii]=res_rsd[ii]
        liner, =ax1.plot(lat,tmp,lw=linew,color='C%d'%(colors[5]),label='R '+"%.2f" %(resravg[iAA,0])+" %.0f%%" % (100 * resravg[iAA,0]/respavg[iAA,0]))
        ax1.plot(lat,res_rsd,lw=linew,color=liner.get_color(),alpha=alpha*2)
    ax1.axhline(y = 0, color = 'k', linestyle = '--',lw=linew)
    ax1.set_xlim([30,90]);ax1.set_ylim([-6,6])
    ax1.set_xticks(np.arange(30, 91, step=20), [u'30\N{DEGREE SIGN}N', u'50\N{DEGREE SIGN}N',u'70\N{DEGREE SIGN}N',u'90\N{DEGREE SIGN}N'], fontsize=fntsz) 
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='both', which='minor',length=5);
    if iAA<2: 
        ax1.set_ylabel('Feedback (W m$^{-2}$ K$^{-1}$)', fontsize=fntsz)
        ax1.xaxis.set_tick_params(labelbottom=False)
    else: 
        ax1.set_xlabel('Latitude', fontsize=fntsz);ax1.set_ylabel('Feedback (W m$^{-2}$)', fontsize=fntsz)
    if Rsquared=='Rsquared':
        ax1.set_ylim([0,.13])
        if iAA%ncol<1: ax1.set_ylabel('R$^{2}$', fontsize=fntsz)
    if iAA%ncol>0: ax1.yaxis.set_tick_params(labelleft=False)
    ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz,handletextpad=0.0, handlelength=0)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
    ax1.grid()
ax1=axes.flat[-1];
ax1.set_title('('+letters[nAA]+')',loc='left', fontsize=fntsz)
markers='o^v<s';msz=10;cpsz=3;alpha=.35
handles_markers = [];markers_labels = []
for iAA in range(nAA):
    if iAA<1: lbl='SAT'
    else:lbl='$A_%d$'%iAA
    if model[0:4]=='CESM':pts = ax1.errorbar([0], [0], xerr=[0],yerr=[0],marker=markers[iAA], c='k',mfc='none',ms=msz,label=lbl)
    else:pts = ax1.scatter([0], [0],marker=markers[iAA], edgecolors='k', facecolors='none',s=msz*10,label=lbl)
    handles_markers.append(pts);markers_labels.append(lbl);pts.remove()
    marker, caps, bars = ax1.errorbar(respavg[iAA,1]-respglbavg[iAA],respavg[iAA,0]-respglbavg[iAA],yerr=respstd[iAA,0],xerr=respstd[iAA,1],color=linep.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    marker, caps, bars =ax1.errorbar(reslavg[iAA,1],reslavg[iAA,0],xerr=reslstd[iAA,1],yerr=reslstd[iAA,0],color=linel.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    marker, caps, bars =ax1.errorbar(resaavg[iAA,1],resaavg[iAA,0],xerr=resastd[iAA,1],yerr=resastd[iAA,0],color=linea.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    marker, caps, bars =ax1.errorbar(resqavg[iAA,1],resqavg[iAA,0],xerr=resqstd[iAA,1],yerr=resqstd[iAA,0],color=lineq.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    marker, caps, bars =ax1.errorbar(rescavg[iAA,1],rescavg[iAA,0],xerr=rescstd[iAA,1],yerr=rescstd[iAA,0],color=linec.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
    marker, caps, bars =ax1.errorbar(resravg[iAA,1],resravg[iAA,0],xerr=resrstd[iAA,1],yerr=resrstd[iAA,0],color=liner.get_color(),fmt=markers[iAA], mfc='none',ms=msz, capsize=cpsz,markeredgewidth=mkedgwd)
    [bar.set_alpha(alpha) for bar in bars];[cap.set_alpha(alpha) for cap in caps]
minimum = np.min((ax1.get_xlim(),ax1.get_ylim()));maximum = np.max((ax1.get_xlim(),ax1.get_ylim()));ax1.axline([minimum, minimum], [maximum, maximum], color = 'k', linestyle = '--',lw=linew/3)
ax1.axhline(y = 0, color = 'k', linestyle = '--',lw=linew/3);ax1.axvline(x = 0, color = 'k', linestyle = '--',lw=linew/3)
ax1.legend(handles_markers, markers_labels,frameon=False, fontsize=fntsz)
ax1.set_xlabel('Tropical feedback', fontsize=fntsz);ax1.set_ylabel('Arctic feedback', fontsize=fntsz)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5);
fig.tight_layout()
plt.show()
