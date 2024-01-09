#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
scatter plot of rKGE - relative KGE values
'''

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import LogNorm,Normalize,ListedColormap,BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
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
import os
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import warnings;warnings.filterwarnings('ignore')

# import params as pm
import read_grdc as grdc
# import cal_stat as stat
#import plot_colors as pc
from statistics import *
#====================================================================
def read_dis(experiment,station,syear,eyear):
    asm=[]
    opn=[]
    fname="./txt/"+experiment+"/outflow/"+station+".txt"
    with open(fname,"r") as f:
        lines=f.readlines()
    for line in lines:
        line    = re.split(" ",line)
        line    = list(filter(None, line))
        year    = int(line[0][0:4])
        # print year
        if year < syear or year > eyear:
            continue
        try:
            asmval  = float(line[1])
        except:
            asmval  = 0.0
        # asmval  = float(line[1])
        opnval  = float(line[2])
        asm.append(asmval)
        opn.append(opnval)
    return np.array(asm), np.array(opn)
#====================================================================
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
nXX = int(filter(None, re.split(" ",lines[2]))[6])
nYY = int(filter(None, re.split(" ",lines[2]))[7])
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

dfout = pd.DataFrame()
for exp,label in zip(experiments,labels):
    print (exp)
    dfname="./out/"+exp+"/datafile.csv"
    # print (dfname)
    df = pd.read_csv(dfname, sep=';')
    print (df.head())
    dfout[label]=df["rKGE"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
    # dfrCC[label]=df["rCC"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfrBR[label]=df["rBR"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfrRV[label]=df["rRV"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # print (df["rKGE"].values)
dfout["max_rKGE"]=dfout.idxmax(axis=1)
dfout["max_rKGE_int"]=dfout["max_rKGE"] #pd.Categorical(dfout["max_rKGE"]).codes
dfout["max_rKGE_int"].replace(labels,range(len(labels)),inplace=True)
# add lon lat
dfout["LON"]=df["LON"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
dfout["LAT"]=df["LAT"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
dfout["GRDC_ID"]=df["GRDC_ID"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
dfout["RIVER"]=df["RIVER"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
dfout["STATION"]=df["STATION"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
print (dfout.head())
#
print (dfout["max_rKGE"].value_counts())
print (len(dfout["max_rKGE"].values))
print ((dfout["max_rKGE"].value_counts()/float(len(dfout["max_rKGE"].values)))*100.0)

dfout.rename(columns=lambda x: x.strip(), inplace=True)

print (dfout.head())
print (dfout.columns)
################################
# river width
sup=3 #2
w=0.002 #0.02
alpha=1
width=0.5
# colorbar
cmap=plt.cm.get_cmap("tab20c")
norm=BoundaryNorm(np.arange(0,20+0.1,1),cmap.N) #len(labels)
#====================================================================
va_margin= 0.0#1.38#inch 
ho_margin= 0.0#1.18#inch
hgt=(11.69 - 2*va_margin)*(1.0/3.0)
wdt=(8.27 - 2*ho_margin)*(2.0/2.0)
# hgt=wdt
#fig=plt.figure(figsize=(8.27,11.69))
fig=plt.figure(figsize=(wdt,hgt))
#fig.suptitle("auto-correalated area")#"maximum autocorrelation length")
G = gridspec.GridSpec(ncols=1, nrows=1)
ax0=plt.subplot(G[0,0])
#---------------------------------------
for i,label in enumerate(dfout.columns.values[0:12]):
    print (i, label,dfout.columns.values[i], label==dfout.columns.values[i], dfout[dfout['GRDC_ID'].isin(grdcids)][label].values[:])
    grdcnums=dfout[dfout['GRDC_ID'].isin(grdcids)]['STATION'].values #len(grdcids) #
    rKGEval=dfout[dfout['GRDC_ID'].isin(grdcids)][label].values
    ax0.plot(grdcnums,rKGEval,color=cmap(norm(i)))
xlist=[x.strip() for x in dfout[dfout['GRDC_ID'].isin(grdcids)]['STATION'].values[:]]
ax0.set_xticklabels(xlist,rotation=90)
# ax0.set_ylim(-1,1)
ax0.set_xlabel("xlabel")
# legend 
features=[]
pnum=len(labels)
for i in np.arange(pnum):
    label=labels[i]
    features.append(mlines.Line2D([], [], color=cmap(norm(i)),label=label))
    # features.append(mlines.Line2D([], [], color=cmap(norm(i)), marker="o",
    #                       markeredgecolor="none",markersize=5, markeredgewidth=0.5,
    #                       label=label,linewidth=0.0)) #edgecolors="k",
legend=plt.legend(handles=features,bbox_to_anchor=(0.50,-0.50), loc="top center",
           bbox_transform=fig.transFigure, ncol=3,  borderaxespad=0.0, frameon=False,prop={'size': 8})#

#==============================================================================
print ("./figures/"+figname+".png")
plt.savefig("./figures/"+figname+".pdf",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".png",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".jpg",dpi=800,bbox_inches="tight", pad_inches=0.01)