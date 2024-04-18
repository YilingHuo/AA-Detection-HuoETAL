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
from functions import areaavg_lat,seasonavg,seasonget,smOLSconfidencelev,statslinregressts_slopepvalue
idataset=0 #0:HadCRUT;1:ERA5
clev=0.05#confidence level
nwndw=1;naa=7### # of windows; # of AA matrices
stryr=1901;endyr=2022-nwndw*10;nyear=endyr-stryr+1
window_size=21;mvtrndgap=int((window_size-1)/2)
stryraa=stryr-mvtrndgap;endyraa=endyr+mvtrndgap;nyearaa=nyear+2*mvtrndgap
#the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
refyr1=1951;refyr2=1980
glblatlim=-91;arealatlim=70
if glblatlim==-91:
    glb=''
elif glblatlim==0:
    glb='NH'
if arealatlim==70:
    area=''
else:
    area='lat'+str(arealatlim)
if idataset<1:
    ds=nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc')
    filestryr=1850;varname='tas_mean'
    dataset='HadCRUT'
else:
    ds=nc.Dataset('/pscratch/sd/h/huoyilin/e5.sfc.t2m.19402022.nc')
    filestryr=1940;varname='t2m'
    dataset='ERA5'
seasons='JFMAMJJASOND'
m11=[0,5,11];m21=[12,8,2];nseason=len(m11)
aa=np.zeros([naa+2,nseason,nyear])### extra ones for significant trend/trend
season = ["" for x in range(nseason)]
for iseason in range(nseason):
    m1=m11[iseason];m2=m21[iseason]
    stryr1=stryr+0;refyr11=refyr1+0
    if m2-m1>11:
        season[iseason]='Annual'
        nmonth=12
    elif m2<m1:
        season[iseason]=seasons[m1:]+seasons[:m2]
        stryr1-=1 ##for DJF read in one previous year
        refyr11=refyr1-1
        nmonth=12+m2-m1
    else:
        season[iseason]=seasons[m1:m2]     
        nmonth=m2-m1
    t=ds[varname][(refyr11-filestryr)*12:(refyr2-filestryr+1)*12]
    lat=ds['latitude'][:]
    t_s=seasonavg(t,m1,m2,refyr2-refyr1+1)
    t_areaavgref=np.mean(areaavg_lat(t_s,lat,latlim=arealatlim))
    t_glbavgref=np.mean(areaavg_lat(t_s,lat,latlim=glblatlim));del t,t_s
    ####the AA index as the ratio of the 21-year moving trend centered on the year of interest of the Arctic surface temperature anomaly to the 21-year moving trend of global temperature anomaly.
    stryraa1=stryr1-mvtrndgap
    t=ds[varname][(stryraa1-filestryr)*12:(endyraa-filestryr+1)*12]
    t_s=seasonget(t,m1,m2,nyearaa)
    t_areaavgaa_m=areaavg_lat(t_s,lat,latlim=arealatlim)-t_areaavgref
    t_glbavgaa_m=areaavg_lat(t_s,lat,latlim=glblatlim)-t_glbavgref
    t_s=seasonavg(t,m1,m2,nyearaa)
    t_areaavgaa=areaavg_lat(t_s,lat,latlim=arealatlim)-t_areaavgref
    t_glbavgaa=areaavg_lat(t_s,lat,latlim=glblatlim)-t_glbavgref;del t,t_s
    #{SAT anomaly in Arctic} – {SAT anomaly in NH}
    # Convert array of integers to pandas series
    numbers_series = pd.Series(t_areaavgaa-t_glbavgaa)
    # Get the window of series of observations of specified window size
    # Create a series of moving averages of each window
    # Convert pandas series back to list
    moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
    # Remove null entries from the list
    aa[0,iseason] = moving_averages_list[window_size - 1:]  
    #the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
    numbers_series = pd.Series(t_areaavgaa/t_glbavgaa)
    moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
    aa[4,iseason]=moving_averages_list[window_size - 1:]
    # (SAT anomaly in HA/SD in Arctic)/(global SAT anomaly/global SD)
    numbers_series = pd.Series((t_areaavgaa/np.std(t_areaavgaa))/(t_glbavgaa/np.std(t_glbavgaa)))
    moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
    aa[5,iseason]=moving_averages_list[window_size - 1:]
    ##|SAT 21-year linear trend in Arctic|/|SAT 21-year linear trend in NH|   A2 
    ##{Inter-annual SAT variability in Arctic}/{inter-annual SAT variability in NH}
    ##the ratio of the inter-annual temperature variability in a 21-year running-window between the Arctic and the NH (A3)
# Coefficient of linear regression between Arctic and NH SAT anomalies
#The AA defined by the ratio of the slope of the linear regression between Arctic and NH SAT anomalies in a 21-year running-window (A4).
    for iyear in range(nyear):
        tmp=iyear+window_size
        tmpglb=t_glbavgaa[iyear:tmp];tmparea=t_areaavgaa[iyear:tmp]
        aa[2,iseason,iyear]=np.std(tmparea)/np.std(tmpglb)
        indexyr=np.arange(stryraa+iyear,stryraa+tmp)
        glbtrnd=statslinregressts_slopepvalue(indexyr,tmpglb)
        areatrnd=statslinregressts_slopepvalue(indexyr,tmparea)
        aa[naa,iseason,iyear]=areatrnd[0]/glbtrnd[0]
        #(Trends of standardised SAT anomalies in Arctic)/(global trends of standardised SAT anomalies)
        if glbtrnd[1]>clev or areatrnd[1]>clev:
            aa[1,iseason,iyear]=np.nan
        else:
            aa[1,iseason,iyear]=aa[naa,iseason,iyear]+0
        glbtrnd=statslinregressts_slopepvalue(indexyr,preprocessing.scale(tmpglb))
        areatrnd=statslinregressts_slopepvalue(indexyr,preprocessing.scale(tmparea))
        aa[naa+1,iseason,iyear]=areatrnd[0]/glbtrnd[0]
        if glbtrnd[1]>clev or areatrnd[1]>clev:
            aa[6,iseason,iyear]=np.nan
        else:
            aa[6,iseason,iyear]=aa[naa+1,iseason,iyear]+0
        tmpglb=t_glbavgaa[iyear:tmp];tmparea=t_areaavgaa[iyear:tmp]
        aa[3,iseason,iyear]=statslinregressts_slopepvalue(tmpglb,tmparea)[0]
fntsz=20;letters='abcdefghijklmn';linew=3;colors='krb'
fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(17.5, 16), facecolor='w')
indexyr=np.arange(stryr,endyr+1)
ymin=[-2,0,0,-2,-15,-10,0]
ymax=[1.5,14,10,6,10,2.2,2]
ymin=[-1,0,1,-1,-1,-.2,0]
ymax=[1,12,10,5,10,2.2,2]
ymin=[-1,-4,-4,-4,-4,-2.5,-2.5]
ymax=[1,18,18,18,18,6,6]
for iaa in range(naa):    
    ax1 = axes.flat[iaa]
    ax1.set_title('('+letters[iaa]+')',fontsize=fntsz,loc='left')
    for iseason in range(nseason): 
        p=ax1.plot(indexyr,aa[iaa,iseason],lw=linew,label=season[iseason],color=colors[iseason])
        if iaa==1:
            ax1.plot(indexyr,abs(aa[naa,iseason]),lw=linew,color=p[0].get_color() ,alpha=.3)
        elif iaa==6:
            ax1.plot(indexyr,abs(aa[naa+1,iseason]),lw=linew,color=p[0].get_color() ,alpha=.3)
    ax1.set_xlim([stryr,endyr])
    if iaa>0:ax1.set_ylim([ymin[iaa],ymax[iaa]])        
    ax1.set_ylabel(r'$A_{}$'.format(iaa+1), fontsize = fntsz)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='both', which='minor',length=5);
    ax1.grid()
    if iaa==0:
        ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz,handlelength=0)
        ax1.set_ylabel(r'$A_{}$'.format(iaa+1)+' (K)', fontsize = fntsz)
    else:
        ax1.set_ylabel(r'$A_{}$'.format(iaa+1), fontsize = fntsz)
    if iaa>naa-3:
        ax1.set_xlabel('Year', fontsize = fntsz)
    else:
        ax1.xaxis.set_tick_params(labelbottom=False) 
axes[3][1].set_visible(False)
fig.suptitle('The '+str(naa)+' metrics for AA from the '+dataset+' data calculated using '+str(window_size)+'-year running-window', fontsize = fntsz)
plt.savefig('Fig2.png', bbox_inches='tight')
plt.show()
