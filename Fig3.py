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
from functions import (areaavg_lat,seasonavg,seasonget,smOLSconfidencelev,statslinregressts_slopepvalue)
datasets=['HadCRUT','ERA5','CESM2LE']
ndataset=len(datasets)
clev=0.05#confidence level
nwndw=1;naa=7### # of windows; # of AA matrices
stryr=1941+nwndw*10;endyr=2014-nwndw*10;nyear=endyr-stryr+1 
window_size=21;mvtrndgap=int((window_size-1)/2)
stryraa=stryr-mvtrndgap;endyraa=endyr+mvtrndgap;nyearaa=nyear+2*mvtrndgap
#the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
refyr1=1981;refyr2=2010
refyr1=stryr-nwndw*10;refyr2=refyr1+30
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
m11=[0];m21=[12];nseason=len(m11)
seasons='JFMAMJJASOND'
os.chdir('/global/cfs/cdirs/m1199/jianlu/CESM2_LE/monthly/TREFHT/')
files=sorted(glob.glob("b.e21.BHISTc*.nc"));ntimeslice=17
nensemble=int(len(files)/ntimeslice)
ds=nc.Dataset(files[0]);latcesm=ds['lat'][:]
aaensttl=nensemble+ndataset-1
aa=np.zeros([nseason,naa+2,aaensttl,nyear])+np.nan### extra ones for significant trend/trend
aaens=np.zeros([nseason,naa+2,nyear])+np.nan
for iseason in range(nseason):
    m1=m11[iseason];m2=m21[iseason]
    stryr1=stryr+0;refyr11=refyr1+0
    if m2-m1>11:
        nmonth=12
    elif m2<m1:
        stryr1-=1 ##for DJF read in one previous year
        refyr11=refyr1-1
        nmonth=12+m2-m1
    else:
        nmonth=m2-m1
    stryraa1=stryr1-mvtrndgap
    icount=0;enscount=0;
    for idataset in range(aaensttl):#0:HadCRUT;1:ERA5;2-:CESM2_LE
        if idataset<1:
            ds=nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc')
            filestryr=1850;varname='tas_mean'
            lat=ds['latitude'][:]
            icount+=1
        elif idataset<2:
            ds=nc.Dataset('/pscratch/sd/h/huoyilin/e5.sfc.t2m.19402022.nc')
            filestryr=1940;varname='t2m'
            lat=ds['latitude'][:]
            icount+=1
        else:
            enscount+=1
            lat=latcesm
            filestryr=1850;varname='TREFHT'
            iens=(idataset-icount)*ntimeslice
            fn=files[iens].split(".")
            scenario=fn[2][5:];iensemble=fn[4][4:]+'.'+ fn[5]
            # fileens=glob.glob('b.e21.BHIST'+scenario+'.f09_g17.LE2-'+iensemble+'.cam.h0.'+varname+'.*-*.nc')
            ds = xr.open_mfdataset(files[iens:iens+ntimeslice],combine='by_coords').load()

        t=ds[varname][(refyr11-filestryr)*12:(refyr2-filestryr+1)*12]
        t_s=seasonavg(t,m1,m2,refyr2-refyr1+1)
        t_areaavgref=np.mean(areaavg_lat(t_s,lat,latlim=arealatlim))
        t_glbavgref=np.mean(areaavg_lat(t_s,lat,latlim=glblatlim));del t,t_s
        t=ds[varname][(stryraa1-filestryr)*12:(endyraa-filestryr+1)*12]
        t_s=seasonavg(t,m1,m2,nyearaa)
        t_areaavgaa=areaavg_lat(t_s,lat,latlim=arealatlim)-t_areaavgref
        t_glbavgaa=areaavg_lat(t_s,lat,latlim=glblatlim)-t_glbavgref;del t,t_s
        if enscount<2:        t_areaavgref1=t_areaavgref+0;t_glbavgref1=t_glbavgref+0;t_areaavgaa1=t_areaavgaa+0;t_glbavgaa1=t_glbavgaa+0
        else:        t_areaavgref1+=t_areaavgref;t_glbavgref1+=t_glbavgref;t_areaavgaa1+=t_areaavgaa;t_glbavgaa1+=t_glbavgaa
        #{SAT anomaly in Arctic} – {SAT anomaly in NH}
        # Convert array of integers to pandas series
        numbers_series = pd.Series(t_areaavgaa-t_glbavgaa)
        # Get the window of series of observations of specified window size
        # Create a series of moving averages of each window
        # Convert pandas series back to list
        moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
        # Remove null entries from the list
        aa[iseason,0,idataset] = moving_averages_list[window_size - 1:]  
        #the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
        numbers_series = pd.Series(t_areaavgaa/t_glbavgaa)
        moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
        aa[iseason,4,idataset]=moving_averages_list[window_size - 1:]
        # (SAT anomaly in HA/SD in Arctic)/(global SAT anomaly/global SD)
        numbers_series = pd.Series((t_areaavgaa/np.std(t_areaavgaa))/(t_glbavgaa/np.std(t_glbavgaa)))
        moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
        aa[iseason,5,idataset]=moving_averages_list[window_size - 1:]
        ##|SAT 21-year linear trend in Arctic|/|SAT 21-year linear trend in NH|   A2 
        ##{Inter-annual SAT variability in Arctic}/{inter-annual SAT variability in NH}
        ##the ratio of the inter-annual temperature variability in a 21-year running-window between the Arctic and the NH (A3)
    # Coefficient of linear regression between Arctic and NH SAT anomalies
    #The AA defined by the ratio of the slope of the linear regression between Arctic and NH SAT anomalies in a 21-year running-window (A4).
        for iyear in range(nyear):
            tmp=iyear+window_size
            tmpglb=t_glbavgaa[iyear:tmp];tmparea=t_areaavgaa[iyear:tmp]
            # tmpglb=t_glbavgaa_m[iyear*nmonth:tmp*nmonth];tmparea=t_areaavgaa_m[iyear*nmonth:tmp*nmonth]
            aa[iseason,2,idataset,iyear]=np.std(tmparea)/np.std(tmpglb)
            indexyr=np.arange(stryraa+iyear,stryraa+tmp)
            glbtrnd=statslinregressts_slopepvalue(indexyr,tmpglb)
            areatrnd=statslinregressts_slopepvalue(indexyr,tmparea)
            aa[iseason,naa,idataset,iyear]=areatrnd[0]/glbtrnd[0]
            #(Trends of standardised SAT anomalies in Arctic)/(global trends of standardised SAT anomalies)
            if glbtrnd[1]>clev or areatrnd[1]>clev:
                aa[iseason,1,idataset,iyear]=np.nan
            else:
                aa[iseason,1,idataset,iyear]=aa[iseason,naa,idataset,iyear]+0
            glbtrnd=statslinregressts_slopepvalue(indexyr,preprocessing.scale(tmpglb))
            areatrnd=statslinregressts_slopepvalue(indexyr,preprocessing.scale(tmparea))
            aa[iseason,naa+1,idataset,iyear]=areatrnd[0]/glbtrnd[0]
            if glbtrnd[1]>clev or areatrnd[1]>clev:
                aa[iseason,6,idataset,iyear]=np.nan
            else:
                aa[iseason,6,idataset,iyear]=aa[iseason,naa+1,idataset,iyear]+0
            tmpglb=t_glbavgaa[iyear:tmp];tmparea=t_areaavgaa[iyear:tmp]
            aa[iseason,3,idataset,iyear]=statslinregressts_slopepvalue(tmpglb,tmparea)[0]
t_areaavgref1/=enscount;t_glbavgref1/=enscount;t_areaavgaa1/=enscount;t_glbavgaa1/=enscount
numbers_series = pd.Series(t_areaavgaa1-t_glbavgaa1)
# Get the window of series of observations of specified window size
# Create a series of moving averages of each window
# Convert pandas series back to list
moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
# Remove null entries from the list
aaens[iseason,0] = moving_averages_list[window_size - 1:]  
#the AA index: the ratio of the Arctic-mean to the global-mean SAT changes relative to the 1980–2009 mean
numbers_series = pd.Series(t_areaavgaa1/t_glbavgaa1)
moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
aaens[iseason,4]=moving_averages_list[window_size - 1:]
# (SAT anomaly in HA/SD in Arctic)/(global SAT anomaly/global SD)
numbers_series = pd.Series((t_areaavgaa1/np.std(t_areaavgaa1))/(t_glbavgaa1/np.std(t_glbavgaa1)))
moving_averages_list = numbers_series.rolling(window_size).mean().tolist()
aaens[iseason,5]=moving_averages_list[window_size - 1:]
fntsz=20;letters='abcdefghijklmn';linew=3;alphal=.3;alphas=.1
fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(16, 16), facecolor='w')
indexyr=np.arange(stryr,endyr+1)
ymin=[-.8,-2,-2,-2,-2,-.5,-.5]
ymax=[2.5,12,12,12,12,4,4]
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
for iaa in range(naa):    
    ax1 = axes.flat[iaa]
    stryr1=stryr+0;refyr11=refyr1+0
    if m2-m1>11:
        season='Annual'
    elif m2<m1:
        season=seasons[m1:]+seasons[:m2]
    else:
        season=seasons[m1:m2]     
    ax1 = axes.flat[iseason]
    ax1.set_title('('+letters[iseason]+')',fontsize=fntsz,loc='left')
    for idataset in range(ndataset-1): ##CESM LE are ensembles
        line,=ax1.plot(indexyr,aa[iseason,iaa,idataset],lw=linew,color=colors[idataset],label=datasets[idataset])
        if iaa==1:
            ax1.plot(indexyr,abs(aa[iseason,naa,idataset]),lw=linew,color=line.get_color() ,alpha=alphal)
        elif iaa==6:
            ax1.plot(indexyr,abs(aa[iseason,naa+1,idataset]),lw=linew,color=line.get_color() ,alpha=alphal)
    n1=ndataset-1
    ensavg=aaens[iseason,iaa]
    stddv=np.nanstd(aa[iseason,iaa,n1:], axis=0)
    line,=ax1.plot(indexyr,ensavg,lw=linew,color=colors[ndataset-1],label=datasets[-1])
    if iaa==1:
        ii = np.isnan(ensavg)
        ensavg[ii]=aaens[iseason,naa,ii]
        ax1.plot(indexyr,abs(ensavg),lw=linew,color=line.get_color() ,alpha=alphal)
        stddv[ii]=np.nanstd(aa[iseason,naa,n1:,ii]);
    elif iaa==6:
        ii = np.isnan(ensavg)
        ensavg[ii]=aaens[iseason,naa+1,ii]
        ax1.plot(indexyr,abs(ensavg),lw=linew,color=line.get_color() ,alpha=alphal)
        stddv[ii]=np.nanstd(aa[iseason,naa+1,n1:,ii]);
    if iaa>0:ax1.set_ylim([ymin[iaa],ymax[iaa]])        
    ax1.fill_between(indexyr,ensavg-stddv, ensavg+stddv,color=line.get_color(),alpha=alphas)
    ax1.set_xlim([stryr,endyr])#;ax1.set_ylim([-.8,2.8])
    ax1.set_ylabel(r'$A_{}$'.format(iaa+1), fontsize = fntsz)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz);
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='both', which='minor',length=5);
    ax1.grid()
    if iaa==0:
        ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz,handlelength=0)
    if iaa==0:
        ax1.set_ylabel(r'$A_{}$'.format(iaa+1)+' (K)', fontsize = fntsz)
    else:
        ax1.set_ylabel(r'$A_{}$'.format(iaa+1), fontsize = fntsz)
    if iaa>naa-3:
        ax1.set_xlabel('Year', fontsize = fntsz)
    else:
        ax1.xaxis.set_tick_params(labelbottom=False) 
fig.suptitle('The AA metric from '+datasets[0]+', '+datasets[1]+', and '+datasets[2]+' calculated using '+str(window_size)+'-year running-window', fontsize = fntsz)
plt.savefig('Fig3.png', bbox_inches='tight')
plt.show()
