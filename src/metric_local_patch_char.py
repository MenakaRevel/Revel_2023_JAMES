#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
relationship between discharge metric and open-loop local patch characteristics
'''

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import LogNorm,Normalize,ListedColormap,BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import sys
import os
import calendar
import string
from multiprocessing import Pool
# from multiprocessing import Process
from multiprocessing import sharedctypes
from numpy import ma
import re
import math
# import my_colorbar as mbar
# import cartopy.crs as ccrs
# import cartopy
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import cartopy.feature as cfeature
# import os
import seaborn as sns
import pandas as pd
# from adjustText import adjust_text
import warnings;warnings.filterwarnings('ignore')

sys.path.append('./src')
# import params as pm
import read_grdc as grdc
import read_cgls as cgls
# import cal_stat as stat
#import plot_colors as pc
from statistics import *
from read_function import *
#====================================================================
# # def read_dis(experiment,station,syear,eyear):
# #     '''
# #     read discharge 
# #     '''
# #     fname ="./txt/"+experiment+"/outflow/"+station+".txt"
# #     df_dis=read_csv(fname,sep='\s+',header=None, names=['date','asm','opn'])
# #     df_dis.rename(columns=lambda x: x.strip(), inplace=True)
# #     df_dis.set_index(pd.to_datetime(df_dis['date']),inplace=True)
# #     syyyymmdd='%04d0101'%(syear)
# #     eyyyymmdd='%04d1231'%(eyear)
# #     return df_dis.loc[syyyymmdd:eyyyymmdd,'asm'], df_dis.loc[syyyymmdd:eyyyymmdd,'opn']

# #     # asm=[]
# #     # opn=[]
# #     # fname="./txt/"+experiment+"/outflow/"+station+".txt"
# #     # with open(fname,"r") as f:
# #     #     lines=f.readlines()
# #     # for line in lines:
# #     #     line    = re.split(" ",line)
# #     #     line    = list(filter(None, line))
# #     #     year    = int(line[0][0:4])
# #     #     # print year
# #     #     if year < syear or year > eyear:
# #     #         continue
# #     #     try:
# #     #         asmval  = float(line[1])
# #     #     except:
# #     #         asmval  = 0.0
# #     #     # asmval  = float(line[1])
# #     #     opnval  = float(line[2])
# #     #     asm.append(asmval)
# #     #     opn.append(opnval)
# #     # return np.array(asm), np.array(opn)
# # #====================================================================
# # def read_wse(experiment,station,syear,eyear):
# #     asm=[]
# #     opn=[]
# #     fname ="./txt/"+experiment+"/wse/"+station+".txt"
# #     df_wse=read_csv(fname,sep='\s+',header=None, names=['date','asm','opn'])
# #     df_wse.rename(columns=lambda x: x.strip(), inplace=True)
# #     df_wse.set_index(pd.to_datetime(df_wse['date']),inplace=True)
# #     syyyymmdd='%04d0101'%(syear)
# #     eyyyymmdd='%04d1231'%(eyear)
# #     return df_wse.loc[syyyymmdd:eyyyymmdd,'asm'], df_wse.loc[syyyymmdd:eyyyymmdd,'opn']
    
# #     # with open(fname,"r") as f:
# #     #     lines=f.readlines()
# #     # for line in lines:
# #     #     line    = re.split(" ",line)
# #     #     line    = list(filter(None, line))
# #     #     year    = int(line[0][0:4])
# #     #     # print year
# #     #     if year < syear or year > eyear:
# #     #         continue
# #     #     try:
# #     #         asmval  = float(line[1])
# #     #     except:
# #     #         asmval  = 0.0
# #     #     # asmval  = float(line[1])
# #     #     opnval  = float(line[2])
# #     #     asm.append(asmval)
# #     #     opn.append(opnval)
# #     # return np.array(asm), np.array(opn)
# # #====================================================================
def vec_par(LEVEL,ax=None):
    ax=ax or plt.gca()
    txt=re.split("-",figname)[0]+"_%02d.txt"%(LEVEL)
    os.system("./bin/print_rivvec "+tmp0+" 1 "+str(LEVEL)+" > "+txt)
    width=(float(LEVEL)**sup)*w
    #print LEVEL, width#, lon1,lat1,lon2-lon1,lat2-lat1#x1[0],y1[0],x1[1]-x1[0],y1[1]-y1[0]
    # open tmp2.txt
    with open(txt,"r") as f:
        lines = f.readlines()

    #print LEVEL, width, lines, txt
    #---
    for line in lines:
        line = filter(None, re.split(" ",line))
        lon1 = float(line[0])
        lat1 = float(line[1])
        lon2 = float(line[3])
        lat2 = float(line[4])

        # ix = int((lon1 - west)*(1/gsize))
        # iy = int((-lat1 + north)*(1/gsize))

        #- higher resolution data
        # print (north)
        ixx1 = int((lon1  - west)*60.0)
        iyy1 = int((-lat1 + north)*60.0)

        #----
        ix = catmxy[0,iyy1,ixx1] - 1
        iy = catmxy[1,iyy1,ixx1] - 1

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            continue

        if rivermap[iy-1,ix-1] == 0:
            continue

        if lon1-lon2 > 180.0:
            # print (lon1, lon2)
            lon2=180.0
        elif lon2-lon1> 180.0:
            # print (lon1,lon2)
            lon2=-180.0
        #--------
        colorVal="grey"#"k" 
        #print (lon1,lon2,lat1,lat2,width)
        plot_ax(lon1,lon2,lat1,lat2,width,colorVal,ax)
#====================================================================
def plot_ax(lon1,lon2,lat1,lat2,width,colorVal,ax=None):
    ax=ax or plt.gca()
    return ax.plot([lon1,lon2],[lat1,lat2],color=colorVal,linewidth=width,zorder=105,alpha=alpha)
#====================================================================
def plot_hydrograph(num,grdcid,station,experiments,syear=2016,eyear=2019,ax=None):
    """
    Plot all the hydrograph of the all experiments
    """
    start_dt=datetime.date(syear,1,1)
    end_dt=datetime.date(eyear,12,31)
    start=0
    last=(end_dt-start_dt).days + 1
    # colorbar
    cmap=plt.cm.get_cmap("tab20c")
    norm=BoundaryNorm(np.arange(0,20+0.1,1),cmap.N) #len(experiments)
    ax=ax or plt.gca()
    # read grdc
    grdcid=int(grdcid)
    org=grdc.grdc_dis(str(grdcid),syear,eyear)
    org=np.array(org)
    ax.plot(np.arange(start,last),ma.masked_less(org,0.0),label="GRDC",color="#34495e",linewidth=1.0,zorder=101) #,marker = "o",markevery=swt[point])
    for i,experiment in enumerate(experiments):
        # read discharge
        asm,opn=read_dis(experiment,str(grdcid),syear,eyear)
        # plot
        linewidth=0.3 #1.0/float(len(labels))
        ax.plot(np.arange(start,last),asm,label=exp,color=cmap(norm(i)),linewidth=linewidth,alpha=1,zorder=105)
    ax.plot(np.arange(start,last),opn,label="open-loop",color="grey",linewidth=linewidth,linestyle="--",alpha=1,zorder=104)
    # print (last,len(opn),len(org),syear,eyear)
    # Make the y-axis label, ticks and tick labels match the line color.
    if num in [0,11,10,9]:
        ax.set_ylabel('discharge ($m^3s^{-1}$)', color='k',fontsize=6)
        # ax.yaxis.label.set_size(10)
    ax.set_xlim(xmin=0,xmax=last+1)
    ax.tick_params('y', colors='k')
    # scentific notaion
    ax.ticklabel_format(style="sci",axis="y",scilimits=(0,0),useOffset=1,useLocale=False,useMathText=True)
    # ax.yaxis.major.formatter._useMathText=True
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.offsetText.set_fontsize(6)
    ax.yaxis.get_offset_text().set_x(-0.15)
    # ax.yaxis.get_offset_text().set_y(0.10)
    if eyear-syear > 1:
        dtt=1
        dt=int(math.ceil(((eyear-syear)+2)/dtt))
    else:
        dtt=1
        dt=(eyear-syear)+2
    xxlist=np.linspace(0,last,dt,endpoint=True)
    #xxlab=[calendar.month_name[i][:3] for i in range(1,13)]
    xxlab=np.arange(syear,eyear+2,dtt)
    ax.set_xticks(xxlist)
    if num in [6,7,8,9]:
        ax.set_xticklabels(xxlab,fontsize=6)
    else:
        ax.set_xticklabels([])
    # ax.set_xticklabels(xxlab,fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.text(0.05,1.10,"%s) %s"%(string.ascii_lowercase[num],station)
        ,ha="left",va="center",transform=ax.transAxes,fontsize=6)
#====================================================================
def plot_best_hydrograph(num,grdcid,station,expnum,experiment,syear=2016,eyear=2019,ax=None):
    """
    Plot all the hydrographs of the all experimets
    """
    start_dt=datetime.date(syear,1,1)
    end_dt=datetime.date(eyear,12,31)
    start=0
    last=(end_dt-start_dt).days + 1
    # colorbar
    cmap=plt.cm.get_cmap("tab20c")
    norm=BoundaryNorm(np.arange(0,20+0.1,1),cmap.N) #len(experiments)
    ax=ax or plt.gca()
    # read grdc
    grdcid=int(grdcid)
    org=grdc.grdc_dis(str(grdcid),syear,eyear)
    org=np.array(org)
    ax.plot(np.arange(start,last),ma.masked_less(org,0.0),label="GRDC",color="#34495e",linewidth=1.0,zorder=101) #,marker = "o",markevery=swt[point])
    asm,opn=read_dis(experiment,str(grdcid),syear,eyear)
    # plot
    linewidth=0.3 #1.0/float(len(labels))
    ax.plot(np.arange(start,last),asm,label=exp,color=cmap(norm(expnum)),linewidth=linewidth,alpha=1,zorder=105)
    ax.plot(np.arange(start,last),opn,label="open-loop",color="grey",linewidth=linewidth,linestyle="--",alpha=1,zorder=104)
    # print (last,len(opn),len(org),syear,eyear)
    # Make the y-axis label, ticks and tick labels match the line color.
    if num in [0,11,10,9]:
        ax.set_ylabel('discharge ($m^3s^{-1}$)', color='k',fontsize=6)
        # ax.yaxis.label.set_size(10)
    ax.set_xlim(xmin=0,xmax=last+1)
    ax.tick_params('y', colors='k')
    # scentific notaion
    ax.ticklabel_format(style="sci",axis="y",scilimits=(0,0),useOffset=1,useLocale=False,useMathText=True)
    # ax.yaxis.major.formatter._useMathText=True
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.offsetText.set_fontsize(6)
    ax.yaxis.get_offset_text().set_x(-0.15)
    # ax.yaxis.get_offset_text().set_y(0.10)
    if eyear-syear > 1:
        dtt=1
        dt=int(math.ceil(((eyear-syear)+2)/dtt))
    else:
        dtt=1
        dt=(eyear-syear)+2
    xxlist=np.linspace(0,last,dt,endpoint=True)
    #xxlab=[calendar.month_name[i][:3] for i in range(1,13)]
    xxlab=np.arange(syear,eyear+2,dtt)
    ax.set_xticks(xxlist)
    if num in [6,7,8,9]:
        ax.set_xticklabels(xxlab,fontsize=6)
    else:
        ax.set_xticklabels([])
    # ax.set_xticklabels(xxlab,fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.text(0.05,1.10,"%s) %s"%(string.ascii_lowercase[num],station)
        ,ha="left",va="center",transform=ax.transAxes,fontsize=6)
#====================================================================
def metric_local_patch(vslist,syear,eyear,egm08,egm96):
    '''
    calculate wse metrics for open-loop
    '''
    metric=[]
    for vsidx in range(len(vslist)):
        # read vs observation
        org=cgls.cgls_continous_WSE(vslist[vsidx],syear=syear,smon=1,sday=1,
            eyear=eyear,emon=12,eday=31,egm08=egm08[vsidx],egm96=egm96[vsidx])
        # read simulations
        asm,opn=read_wse('NOM_WSE_ERA5_CGLS_062',vslist[vsidx],syear,eyear)
        print (org)
        print (opn)
        # RMSE
        rmse=RMSE(opn,org)
        # bias
        bias=BIAS(opn,org)
        # NSE
        nse=NSE(opn,org)
        # KGE, CC, BR, RV
        kge,cc,br,rv=KGE_components(opn,org)
        # 
        metric.append([rmse,bias,nse,kge,cc,br,rv])
    return np.array(metric)
#====================================================================
def calc_metric(vslist,syear,eyear,egm08,egm96):
    '''
    calculate wse metrics for list of VS list
    '''
    metric=[]
    for vsidx in range(len(vslist)):
        # print (vslist[vsidx])
        # read vs observation
        org=cgls.cgls_continous_WSE(vslist[vsidx],syear=syear,smon=1,sday=1,
            eyear=eyear,emon=12,eday=31,egm08=egm08[vsidx],egm96=egm96[vsidx])
        # read simulations
        asm,opn=read_wse('NOM_WSE_ERA5_CGLS_062',vslist[vsidx],syear,eyear)
        #
        # print (len(org), len(opn))
        # RMSE
        rmse=RMSE(opn,org)
        # bias
        bias=BIAS(opn,org)
        # NSE
        nse=NSE(opn,org)
        # KGE, CC, BR, RV
        kge,cc,br,rv=KGE_components(opn,org)
        # KGED
        kged=KGED(opn,org)
        # 
        metric.append([rmse,bias,nse,kge,cc,br,rv,kged])
    return np.array(metric)
#====================================================================
argv=sys.argv
syear=int(argv[1])
eyear=int(argv[2])
CaMa_dir=argv[3]
mapname=argv[4]
exlist=argv[5]
stlist=argv[6]
figname=argv[7]
ncpus=int(sys.argv[8])
ens_mem=49
seaborn_map=True
# seaborn_map=False
#==================================
#=== read experiment names ===
with open(exlist,"r") as exf:
    linesexf=exf.readlines()
experiments=[]
labels=[]
for lineexf in linesexf:
    lineexf = re.split(":",lineexf)
    lineexf = list(filter(None, lineexf))
    labels.append(lineexf[0])
    experiments.append(lineexf[1].strip())
#=============================
lexp=len(experiments)
#==================================
#=== read GRDC station list ===
with open(stlist,"r") as stf:
    linessta=stf.readlines()
grdcids=[]
stations=[]
for linesta in linessta:
    linesta = re.split("#",linesta)
    linesta = list(filter(None, linesta))
    grdcids.append(int(linesta[0]))
    stations.append(linesta[1].strip())
#===================
fname=CaMa_dir+"/map/"+mapname+"/params.txt"
with open(fname,"r") as f:
    lines=f.readlines()
#-------
nx     = int(list(filter(None, re.split(" ",lines[0])))[0])
ny     = int(list(filter(None, re.split(" ",lines[1])))[0])
gsize  = float(list(filter(None, re.split(" ",lines[3])))[0])
lon0   = float(list(filter(None, re.split(" ",lines[4])))[0])
lat0   = float(list(filter(None, re.split(" ",lines[7])))[0])
west   = float(list(filter(None, re.split(" ",lines[4])))[0])
east   = float(list(filter(None, re.split(" ",lines[5])))[0])
south  = float(list(filter(None, re.split(" ",lines[6])))[0])
north  = float(list(filter(None, re.split(" ",lines[7])))[0])
#----
nextxy = CaMa_dir+"/map/"+mapname+"/nextxy.bin"
rivwth = CaMa_dir+"/map/"+mapname+"/rivwth_gwdlr.bin"
rivhgt = CaMa_dir+"/map/"+mapname+"/rivhgt.bin"
rivlen = CaMa_dir+"/map/"+mapname+"/rivlen.bin"
elevtn = CaMa_dir+"/map/"+mapname+"/elevtn.bin"
lonlat = CaMa_dir+"/map/"+mapname+"/lonlat.bin"
uparea = CaMa_dir+"/map/"+mapname+"/uparea.bin"
nextxy = np.fromfile(nextxy,np.int32).reshape(2,ny,nx)
rivwth = np.fromfile(rivwth,np.float32).reshape(ny,nx)
rivhgt = np.fromfile(rivhgt,np.float32).reshape(ny,nx)
rivlen = np.fromfile(rivlen,np.float32).reshape(ny,nx)
elevtn = np.fromfile(elevtn,np.float32).reshape(ny,nx)
lonlat = np.fromfile(lonlat,np.float32).reshape(2,ny,nx)
uparea = np.fromfile(uparea,np.float32).reshape(ny,nx)
#===================
rivnum="/cluster/data6/menaka/HydroDA/dat/rivnum_"+mapname+".bin"
rivnum=np.fromfile(rivnum,np.int32).reshape(ny,nx)
rivermap=((nextxy[0]>0)*1.0)*((rivnum==1)*1.0+(rivnum==7)*1.0)
#===================
satcov="./dat/satellite_coverage.bin"
# satcov=np.fromfile(satcov,np.float32).reshape(ny,nx)
#====================================================================
#higher resolution data
fname=CaMa_dir+"/map/"+mapname+"/1min/location.txt"
with open(fname,"r") as f:
    lines=f.readlines()
nXX = int(list(filter(None, re.split(" ",lines[2])))[6])
nYY = int(list(filter(None, re.split(" ",lines[2])))[7])
catmxy = CaMa_dir+"/map/"+mapname+"/1min/1min.catmxy.bin"
catmxy = np.fromfile(catmxy,np.int16).reshape(2,nYY,nXX)
#===================
# River width threshold
wth_thr=50.0
upa_thr=1.0e4
num_thr=1
metric=[]
# dfrKGE = pd.DataFrame()
# dfrCC = pd.DataFrame()
# dfrBR = pd.DataFrame()
# dfrRV = pd.DataFrame()
#====================================================================
grdc    = CaMa_dir+"/map/"+mapname+"/grdc_loc.txt"
df_grdc = pd.read_csv(grdc, sep=';')
df_grdc.rename(columns=lambda x: x.strip(), inplace=True)
#====================================================================
fcgls   = "/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_conus_06min_org.txt"
df_cgls = pd.read_csv(fcgls, sep='\s+')
df_cgls.rename(columns=lambda x: x.strip(), inplace=True)
# add metrics such as RMSE, bias, CC etc.
# rmse,bias,nse,kge,cc,br,rv

metric=calc_metric(df_cgls['station'].values,syear,eyear,df_cgls['EGM08'].values,df_cgls['EGM96'].values)
print (np.shape(metric))
df_cgls['RMSE(open-loop)']=metric[:,0]
df_cgls['Bias(open-loop)']=metric[:,1]
df_cgls['NSE(open-loop)']=metric[:,2]
df_cgls['KGE(open-loop)']=metric[:,3]
df_cgls['CC(open-loop)']=metric[:,4]
df_cgls['BR(open-loop)']=metric[:,5]
df_cgls['RV(open-loop)']=metric[:,6]
df_cgls['KGED(open-loop)']=metric[:,7]
print (df_cgls.head())
print (df_cgls.columns)
print ("*********")
df_cgls.to_csv("./local_patch_char/open-loop_error_wse.csv",sep=',')

'''
#=====
def KGE_components_(s,o):
    """
	Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    print (s)
    print (o)
    BR = np.mean(s) / np.mean(o)
    RV = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    CC = np.corrcoef(o, s)[0,1]
    val=1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
    return val, CC, BR, RV

org=cgls.cgls_continous_WSE(df_cgls['station'].values[0],syear=syear,smon=1,sday=1,
    eyear=eyear,emon=12,eday=31,egm08=df_cgls['EGM08'].values[0],egm96=df_cgls['EGM96'].values[0])

# read simulations
asm,opn=read_wse('NOM_WSE_ERA5_CGLS_062',df_cgls['station'].values[0],syear,eyear)

print (opn)

print (org)

print (KGE_components_(opn,org))


#====================================================================
# get one data file for grdc list
dfname="./out/"+experiments[0]+"/datafile.csv"
df_exp= pd.read_csv(dfname, sep=';')
'''
#====================================================================
# get open-loop characterisitics
# read local patch
patchtype="conus_06min_ERA5_60"
patchdir="/cluster/data6/menaka/Empirical_LocalPatch/local_patch/conus_06min_ERA5_60"
# metric_array=np.zeros()
metric_array=[]
VScount_array=[]
for point in df_grdc.index[0::]:
    ix0=df_grdc['ix1'][point]
    iy0=df_grdc['iy1'][point]
    patchname = patchdir+"/patch%04d%04d.txt"%(ix0,iy0)
    dflp = pd.read_csv(patchname, sep='\s+',skipinitialspace = True, header=None, names=['IX', 'IY', 'weight'])
    #=====================================
    # include vs inside local patch
    # vs= np.zeros([len(dflp)],dtype=object)
    # vs[:] = 'nan'
    metricVals=[]
    VS_count=0
    for j in range(len(dflp)):
        ix = dflp['IX'][j]
        iy = dflp['IY'][j]
        #==================================
        if len(df_cgls[(df_cgls['ix']==ix) & (df_cgls['iy']==iy)].values)>0:
            # print (dfvs["station"][(dfvs['ix']==ix) & (dfvs['iy']==iy)].values)
            VS=df_cgls.loc[(df_cgls['ix']==ix) & (df_cgls['iy']==iy),"ID"].values[0]
            metricVal=df_cgls.loc[df_cgls["ID"]==VS,['RMSE(open-loop)','Bias(open-loop)','NSE(open-loop)',
            'KGE(open-loop)','CC(open-loop)','BR(open-loop)','RV(open-loop)','KGED(open-loop)']]
            metricVals.append(metricVal.values)
            # print (metricVal.values)
            VS_count=VS_count+1
    #=====================================
    # print (VS_count, df_grdc['River'][point], df_grdc['Station'][point])
    if VS_count>0:
        # calcuate min max mean
        metricVals=np.array(metricVals)
        # print (np.shape(metricVals))
        mean_metric=np.mean(metricVals,axis=0)[0]
        max_metric =np.max(metricVals,axis=0)[0]
        min_metric =np.min(metricVals,axis=0)[0]
        # print (VS_count, df_grdc['River'][point], df_grdc['Station'][point], mean_metric)
        # print (df_grdc['River'][point], 
        # df_grdc['Station'][point],np.concatenate((min_metric,mean_metric,max_metric)))
    else:
        mean_metric=np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        max_metric =np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        min_metric =np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
    #=================================
    # concatanate
    metric_combine=np.concatenate((min_metric,mean_metric,max_metric))
    print ("metric_combine:",VS_count,np.shape(metric_combine))
    metric_array.append((metric_combine))
    print ("metric_array:",np.shape(metric_array))
    VScount_array.append(VS_count)

metric_array=np.array(metric_array)
print (np.shape(metric_array))
#====
# add it into a dataframe
listOfmetrics=['minRMSE(open-loop)','minBias(open-loop)','minNSE(open-loop)',
            'minKGE(open-loop)','minCC(open-loop)','minBR(open-loop)','minRV(open-loop)','minKGED(open-loop)',
            'meanRMSE(open-loop)','meanBias(open-loop)','meanNSE(open-loop)',
            'meanKGE(open-loop)','meanCC(open-loop)','meanBR(open-loop)','meanRV(open-loop)','meanKGED(open-loop)',
            'maxRMSE(open-loop)','maxBias(open-loop)','maxNSE(open-loop)',
            'maxKGE(open-loop)','maxCC(open-loop)','maxBR(open-loop)','maxRV(open-loop)','maxKGED(open-loop)']
for idx,listOfmetric in enumerate(listOfmetrics):
    # print (idx, listOfmetric, metric_array[:,idx])
    df_grdc[listOfmetric]=metric_array[:,idx]

print (df_grdc.head())
# strip each element --> need to do
df_obj = df_grdc.select_dtypes(['object'])
df_grdc[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
df_grdc['VS_count']=np.array(VScount_array)

print (df_grdc[df_grdc['River']=='MISSISSIPPI'])
df_grdc.dropna(subset = listOfmetrics,inplace=True)
df_grdc = df_grdc.loc[:, ~df_grdc.columns.str.contains('^Unnamed')]
print (df_grdc.head())

df_grdc.to_csv("./local_patch_char/open-loop_characteristics_wse.csv", sep=";", index=False)

'''
metric_list=[]
for point in df_exp.index:
    grdcid=df_exp["GRDC_ID"][point]
    station=df_exp["STATION"][point]
    ix = df_exp['IX1'][point]#.values[0]
    iy = df_exp['IY1'][point]#.values[0]
    # read local patch characteristics
    local_patch_file='./local_patch_char/conus_06min_ERA5_60/%d.txt'%(grdcid)
    df_local_patch=pd.read_csv(local_patch_file,sep='\s+')
    df_local_patch.dropna(inplace=True)
    #
    EGM08 =  df_cgls.loc[df_cgls['station'].isin(df_local_patch['vs']),'EGM08']
    EGM96 =  df_cgls.loc[df_cgls['station'].isin(df_local_patch['vs']),'EGM96']
    # get metrics such as RMSE, bias, CC etc.
    # rmse,bias,nse,kge,cc,br,rv
    metric=metric_local_patch(df_local_patch['vs'].values,syear,eyear,egm08=EGM08,egm96=EGM96)
    mean_metric=np.mean(metric,axis=1)
    max_metric =np.max(metric,axis=1)
    min_metric =np.min(metric,axis=1)
    metric_list.append([min_metric,mean_metric,max_metric])
metric_list=np.array(metric_list)
print (np.shape(metric_list))
print ('min_metric','mean_metric','max_metric')
'''
'''
#====================================================================    
dfout = pd.DataFrame()
for exp,label in zip(experiments,labels):
    print (exp)
    dfname="./out/"+exp+"/datafile.csv"
    # print (dfname)
    df = pd.read_csv(dfname, sep=';')
    # add column with experiment label
    df['label']=[label]*len(df)
    # print (df.head())
    df = df.loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7),:]
    df.dropna(inplace=True)
    df['label']=[label]*len(df)
    df["log(UPAREA)"]=np.log10(df["UPAREA"])
    df["ELEVTN"]=[elevtn[iy,ix] for iy,ix in zip(df["IY1"],df["IX1"])]
    dfout = pd.concat([dfout,df], ignore_index = True) #, sort=False)
    #dfout[label]=df["rKGE"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
    # dfrCC[label]=df["rCC"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfrBR[label]=df["rBR"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfrRV[label]=df["rRV"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # print (df["rKGE"].values)
# dfout["max_rKGE"]=dfout.idxmax(axis=1)
# dfout["max_rKGE_int"]=dfout["max_rKGE"] #pd.Categorical(dfout["max_rKGE"]).codes
# dfout["max_rKGE_int"].replace(labels,range(len(labels)),inplace=True)
# # add lon lat
# dfout["LON"]=df["LON"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
# dfout["LAT"]=df["LAT"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
# dfout["GRDC_ID"]=df["GRDC_ID"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
# dfout["RIVER"]=df["RIVER"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
# dfout["STATION"]=df["STATION"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
dfout.dropna(axis=1,inplace=True)
print (dfout.head())
#

print (dfout["max_rKGE"].value_counts())
print (len(dfout["max_rKGE"].values))
print ((dfout["max_rKGE"].value_counts()/float(len(dfout["max_rKGE"].values)))*100.0)

dfout.rename(columns=lambda x: x.strip(), inplace=True)

print (dfout.head())
print (dfout.columns)

'''