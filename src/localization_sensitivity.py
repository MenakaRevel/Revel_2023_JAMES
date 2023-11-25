#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
rKGE - relative KGE values for sensitivity experiments
'''

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import BoundaryNorm, ListedColormap #LogNorm,Normalize,ListedColormap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap
# import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import sys
import os
import calendar
import string
from multiprocessing import Pool
# from multiprocessing import Process
from multiprocessing import sharedctypes
from numpy import ma
import re
# import my_colorbar as mbar
# import cartopy.crs as ccrs
# import cartopy
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import cartopy.feature as cfeature
import os
import seaborn as sns
import pandas as pd
import warnings;warnings.filterwarnings('ignore')

# import params as pm
import read_grdc as grdc
# import cal_stat as stat
#import plot_colors as pc
from statistics import *
#========================================
#====  functions for making figures  ====
#========================================
#====================================================================
def mk_dir(sdir):
  try:
    os.makedirs(sdir)
  except:
    pass
#====================================================================
def read_dis(experiment,station):
    asm=[]
    opn=[]
    fname="./txt/"+experiment+"/outflow/"+station+".txt"
    with open(fname,"r") as f:
        lines=f.readlines()
    for line in lines:
        line = re.split(" ",line)
        line = list(filter(None, line))
        asm.append(float(line[1]))
        opn.append(float(line[2]))
    # print (asm)
    return np.array(asm), np.array(opn)
#====================================================================
def read_wse(experiment,station,type0):
    asm=[]
    opn=[]
    fname="./txt/"+experiment+"/wse."+type0+"/"+station+".txt"
    with open(fname,"r") as f:
        lines=f.readlines()
    for line in lines:
        line    = re.split(" ",line)
        line    = list(filter(None, line))
        asm.append(float(line[1]))
        opn.append(float(line[2]))
    return np.array(asm), np.array(opn)
#====================================================================
def read_ul_all(experiment,station):
    ua=[]
    la=[]
    uo=[]
    lo=[]
    # fname="./txt/"+experiment+"/Q_interval/"+station+".txt"
    fname="./txt/"+experiment+"/Q_percentile/"+station+".txt"
    with open(fname,"r") as f:
        lines=f.readlines()
    for line in lines:
        line = re.split(" ",line)
        line = list(filter(None, line))
        try:
            uvala=float(line[1])
            lvala=float(line[2])
            uvalo=float(line[3])
            lvalo=float(line[4])
        except:
            uvala=0.0
            lvala=0.0
            uvalo=0.0
            lvalo=0.0
        ua.append(uvala)
        la.append(lvala)
        uo.append(uvalo)
        lo.append(lvalo)
    return np.array(ua), np.array(la), np.array(uo), np.array(lo)
#====================================================================
def get_GRDClist(fname):
    # obslist="/cluster/data6/menaka/HydroDA/dat/grdc_"+mapname+".txt"
    with open(fname,"r") as f:
        lines=f.readlines()
    stationlist=[]
    for line in lines[1::]:
        line    = re.split(";",line)
        line    = list(filter(None, line))
        # print (line)
        num     = line[0].strip()
        # basin   = line[1].strip()
        # stream  = line[2].strip()
        ix1     = int(line[3])
        iy1     = int(line[4])
        # ix2     = int(line[5])
        # iy2     = int(line[6])
        # staid   = int(num)
        #-------------------------
        if rivermap[iy1-1,ix1-1] !=1.0:
            continue
        stationlist.append(num)
    return np.array(stationlist)
#====================================================================
def print_stat(data,labels):
    # get stats
    df=pd.DataFrame(data=data, columns=labels)
    stats = boxplot_stats(df.values)
    print ("Experiment","Q1 ","Q3","Mean","Median")
    for i in np.arange(0,len(labels)):
        print ("%10s%15.5f%15.5f%15.5f%15.5f")%(labels[i],stats[i]['q1'],stats[i]['q3'],stats[i]['mean'],stats[i]['med'])
    return 0
#====================================================================
def plot_boxplot(df,num,labels,xlabel,colors="Paired",ax=None,ylim=[0,1]):
    ax=ax or plt.gca()
    # palette ="tab20c" #colors #sns.color_palette("Paired", len(labels)) 
    palette =["#e6550e","#3181bd","#6aaed6","#6aaed6","#c7daef"]
    flierprops = dict(marker='o', markerfacecolor='none', markersize=8,linestyle='none', markeredgecolor='k')
    boxprops = dict(color='grey')#facecolor='none'
    whiskerprops = dict(color='grey',linestyle="--")
    capprops = dict(color='grey')
    medianprops = dict(color='grey',linestyle="-",linewidth=1.0)
    meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='green',markersize=8)
    #
    # df=pd.DataFrame(data=data, columns=labels)#, index=stationlist)
    box=sns.boxplot(ax=ax,data=df, fliersize=0.0, palette=palette, whis=1.5\
        ,meanline=False, width=0.8, linewidth=0.3, dodge=True\
        ,meanprops=meanprops,capprops=capprops,medianprops=medianprops) #"Paired"
    #===
    sns.stripplot(ax=ax,data=df, palette=palette, size=2, jitter=0.1, edgecolor="grey", linewidth=0.0, alpha=0.5)
    # ax.axhline(0.0,color="k",linestyle="--",linewidth=0.5)
    ax.set_ylabel(xlabel, color='k',fontsize=10)
    ax.set_xlabel('Experiment', color='k',fontsize=10)
    ax.set_ylim(ymin=ylim[0],ymax=ylim[1])
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(labels,rotation = 90)
    ax.text(-0.05,1.05,"%s)"%(string.ascii_lowercase[num-1]),ha="left",va="center",transform=ax.transAxes,fontsize=8)
    # ax.plot(range(0,len(labels)),df.median(axis=0),marker="o",color='blue',linestyle='--',linewidth=1.0)
    print ("median", df.median(axis=0))
    return 0
#====================================================================
def plot_displot(data,num,labels,xlabel,colors=["#004488","#ddaa33","#ba5566"],ax=None):
    ax=ax or plt.gca()
    print (np.shape(data))
    for i,exp in enumerate(labels):
        sns.distplot(data[:,i],ax=ax, hist = False, kde = True,
                kde_kws = {'linewidth': 1,'linestyle':'-'},
                label = exp, color=colors[i],norm_hist=False)
    ax.set_xlabel(xlabel, color='k',fontsize=8) #$correlation$ $coefficient$ $(r)$
    ax.set_ylabel('Number of gauges', color='k',fontsize=8)
    ax.set_xlim(xmin=-0.5,xmax=0.5)
    ax.tick_params(labelsize=6)
    ax.text(-0.05,1.05,"%s)"%(string.ascii_lowercase[num-1]),ha="left",va="center",transform=ax.transAxes,fontsize=8)
    # plt.legend(ncol=1,loc="lower right",bbox_to_anchor=(-0.15, 0.0),borderaxespad=0.0,frameon=False)
    plt.legend(ncol=1,loc="best",borderaxespad=0.0,frameon=False)
    return 0
#====================================================================
def plot_cdfplot(data,num,labels,xlabel,colors=["#004488","#ddaa33","#ba5566"],ax=None):
    ax=ax or plt.gca()
    kwargs = {'cumulative': True}
    print (np.shape(data))
    for i,exp in enumerate(labels):
        sns.distplot(data[:,i],ax=ax, hist = False, kde = True,
                hist_kws = kwargs,
                kde_kws = {'cumulative': True, 'linewidth': 1,'linestyle':'-'},
                label = exp, color=colors[i],norm_hist=False)
    ax.set_xlabel(xlabel, color='k',fontsize=8) #$correlation$ $coefficient$ $(r)$
    ax.set_ylabel('Cumilative number of gauges', color='k',fontsize=8)
    ax.set_xlim(xmin=-1.0,xmax=0.2)
    ax.tick_params(labelsize=6)
    ax.text(-0.05,1.05,"%s)"%(string.ascii_lowercase[num-1]),ha="left",va="center",transform=ax.transAxes,fontsize=8)
    # plt.legend(ncol=1,loc="lower right",bbox_to_anchor=(-0.1, 0.0),borderaxespad=0.0,frameon=False)
    return 0
#====================================================================
argv=sys.argv
syear=int(argv[1])
eyear=int(argv[2])
CaMa_dir=argv[3]
mapname=argv[4]
exlist=argv[5]
figname=argv[6]
ncpus=int(sys.argv[7])
ens_mem=49
seaborn_map=True
# seaborn_map=False
#==================================
#=== read experiment names ===
# exlist="./experiment_list.nam"
# exlist="./Fig10-experiment_list.nam"
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
# colors=['xkcd:aqua green','xkcd:pastel blue','xkcd:soft pink','xkcd:pastel blue','xkcd:aqua green','xkcd:soft pink']

#assim_out=pm.DA_dir()+"/out/"+pm.experiment()+"/assim_out"
#assim_out=pm.DA_dir()+"/out/"+experiment+"/assim_out"
# assim_out=DA_dir+"/out/"+experiment
# print (assim_out)
# mk_dir("./figures")
# mk_dir("/figures/NSEAI")
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
rivermap=(nextxy[0]>0)*1.0 #*(rivnum==1))*1.0
#===================
satcov="./dat/satellite_coverage.bin"
# satcov=np.fromfile(satcov,np.float32).reshape(ny,nx)
#======================================================
# River width threshold
wth_thr=50.0
upa_thr=1.0e4 #2.5e4 #
num_thr=1
metric=[]
dfrKGE  = pd.DataFrame()
dfNRMSE = pd.DataFrame()
dfNSE   = pd.DataFrame()
dfKGE   = pd.DataFrame()
dfCC    = pd.DataFrame()
dfBR    = pd.DataFrame()
dfRV    = pd.DataFrame()
ii=0
var_cols=[]
for exp,label in zip(experiments,labels):
    # print (exp)
    dfname="./out/"+exp+"/datafile.csv"
    # print (dfname)
    df = pd.read_csv(dfname, sep=';')
    # print (df.head())
    #===================
    # open-loop
    if ii==0:
        dfNSE["Open-loop"]=df["NSEopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
        dfKGE["Open-loop"]=df["KGEopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
        dfCC["Open-loop"]=df["CCopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
        dfBR["Open-loop"]=df["BRopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
        dfRV["Open-loop"]=df["RVopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    #===================
    dfrKGE[label]=df["rKGE"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    dfNSE[label]=df["NSEasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    dfKGE[label]=df["KGEasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    dfCC[label]=df["CCasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    dfBR[label]=df["BRasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    dfRV[label]=df["RVasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
            
    # dfNRMSE[label+"_asm"]=df["NRMSEasm"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # var_cols.append(label+"_asm")
    # if ii==0:
    #     dfNRMSE["opn"]=df["NRMSEopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfNRMSE["opn"]]=df["NRMSEopn"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # dfrRV[label]=df["rRV"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]
    # print (df["rKGE"].values)
# dfrKGE["RIVNUM"]=rivnum[df["IY1"][df["RIV_WTH"]>=wth_thr].values-1,df["IX1"][df["RIV_WTH"]>=wth_thr].values-1]
# dfrCC["RIVNUM"]=rivnum[df["IY1"][df["RIV_WTH"]>=wth_thr].values-1,df["IX1"][df["RIV_WTH"]>=wth_thr].values-1]
# dfrBR["RIVNUM"]=rivnum[df["IY1"][df["RIV_WTH"]>=wth_thr].values-1,df["IX1"][df["RIV_WTH"]>=wth_thr].values-1]
# dfrRV["RIVNUM"]=rivnum[df["IY1"][df["RIV_WTH"]>=wth_thr].values-1,df["IX1"][df["RIV_WTH"]>=wth_thr].values-1]
# print (dfrKGE.head())
# var_cols.append("opn")
#==
# dfNRMSE["weights"]=df["UPAREA"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)]/df["UPAREA"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)].max()
wgt=(np.log10(df["UPAREA"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)])/np.log10(df["UPAREA"].loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr)].max())).values
# dfNRMSE_glb = dfNRMSE[]

# var_cols = lllevels
# print (var_cols)
# print (dfNRMSE.columns)

# var_cols=[e for e in dfNRMSE.columns if e != "opn"]
# # dfNRMSE.columns=dfNRMSE.columns.encode('ascii','ignore')
# dfNRMSE.rename(columns=lambda x: x.encode('raw_unicode_escape').decode())
# print (dfNRMSE.columns)
# print (dfNRMSE.head())
# df2 = pd.DataFrame()
# for v in var_cols:
#     df2[v]=[sum(dfNRMSE[v].values*wgt)/sum(wgt)]
# opn_weight=sum(dfNRMSE["opn"].values*wgt)/sum(wgt)
# # df2 = dfNRMSE.apply(lambda x: ([sum(x[v] * wgt) / sum(wgt) for v in var_cols]))
# df2.columns = [e for e in var_cols if e != "opn"]
# print (df2.head())

var_cols=dfrKGE.columns
# dfNRMSE.columns=dfNRMSE.columns.encode('ascii','ignore')
dfrKGE.rename(columns=lambda x: x.encode('raw_unicode_escape').decode())
# print (dfrKGE.columns)
# print (dfrKGE.head())
df2 = pd.DataFrame()
for v in var_cols:
    df2[v]=[sum(dfrKGE[v].values*wgt)/sum(wgt)]
# opn_weight=sum(dfNRMSE["opn"].values*wgt)/sum(wgt)
# df2 = dfNRMSE.apply(lambda x: ([sum(x[v] * wgt) / sum(wgt) for v in var_cols]))
df2.columns =var_cols #[e for e in var_cols if e != "opn"]
# print (df2.head())

#====================================
#---------- making fig --------------
#====================================
# data=np.transpose(metric)
# data=np.nan_to_num(data)
# print (np.shape(metric))
#====================================
# figure in A4 size
# colorbar
cmap=plt.cm.get_cmap("Blues_r")

bottom= plt.cm.get_cmap('Oranges', 128)
top= plt.cm.get_cmap('Blues_r', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue').reversed()


norm=BoundaryNorm(np.arange(0,lexp+1.1,1),newcmp.N)
# colors=['xkcd:aqua green','xkcd:pastel blue','xkcd:soft pink','xkcd:pastel blue','xkcd:aqua green','xkcd:soft pink']
# colors=["#afccdc","#3174a1","#b5d294","#3f913a","#f4adae","#bf353c"]
# colors=["#3174a1","#3f913a","#bf353c"]
# colors=["#004488","#ddaa33","#ba5566"]
# colors=["#D81B60","#FFC107","#004D40"]
colors=["#e77881","#236f79","#ab812e","#1e439a"]
va_margin= 0.0#1.38#inch 
ho_margin= 0.0#1.18#inch
hgt=(11.69 - 2*va_margin)*(2.0/3.0)
wdt=(8.27 - 2*ho_margin)*(2.0/2.0)
#fig=plt.figure(figsize=(8.27,11.69))
fig=plt.figure(figsize=(wdt,hgt))
#fig.suptitle("auto-correalated area")#"maximum autocorrelation length")
G = gridspec.GridSpec(ncols=2, nrows=2)
#--boxplot
stationlist=get_GRDClist("/cluster/data6/menaka/HydroDA/dat/grdc_"+mapname+".txt")
print ("making plot")

# rKGE boxplot
# print ("----> rKGE")
# ax1 = fig.add_subplot(G[0,0])
# plot_boxplot(dfrKGE,1,labels,"$rKGE$",colors=cmap,ax=ax1,ylim=[-1.02,1.02]) #"Paired"
# # sns.pointplot(data=dfrKGE, estimator='median' ,ax=ax1)
# # sns.pointplot(x=df.columns, y=df[df.columns].values ,ax=ax1, color="red") # ,data=df2
# # ax2 = fig.add_subplot(G[0,1])
# ax1.plot(range(len(df2.columns)),df2.loc[0,:].values.tolist(),marker="D",color="red",linestyle="--",alpha=0.5,zorder=110)
# ax1.plot(range(len(df2.columns)),dfrKGE.loc[df["UPAREA"]>=2.5e4].median(axis=0),marker="*",color="green",linestyle="--",alpha=0.5,zorder=110)

# BIAS ratio plot
ax1 = fig.add_subplot(G[0,0])
plot_boxplot(dfBR,1,dfBR.columns,"$BR$",colors=newcmp,ax=ax1,ylim=[0.0,2.02])
ax1.plot(range(len(dfBR.columns)),dfBR.median(axis=0),marker="o",color="grey",linestyle="-",alpha=0.5,zorder=110)
ax1.plot(range(len(dfBR.columns)),dfBR.loc[df["UPAREA"]>=2.5e4].median(axis=0),marker="o",color="k",linestyle="--",alpha=0.5,zorder=110)

# CC ratio plot
ax2 = fig.add_subplot(G[0,1])
plot_boxplot(dfCC,2,dfCC.columns,"$CC$",colors=newcmp,ax=ax2,ylim=[-1.02,1.02])
ax2.plot(range(len(dfCC.columns)),dfCC.median(axis=0),marker="o",color="grey",linestyle="-",alpha=0.5,zorder=110)
ax2.plot(range(len(dfCC.columns)),dfCC.loc[df["UPAREA"]>=2.5e4].median(axis=0),marker="o",color="k",linestyle="--",alpha=0.5,zorder=110)

# NSE plot
ax3 = fig.add_subplot(G[1,0])
plot_boxplot(dfNSE,3,dfNSE.columns,"$NSE$",colors=newcmp,ax=ax3,ylim=[-2.02,1.02])
ax3.plot(range(len(dfNSE.columns)),dfNSE.median(axis=0),marker="o",color="grey",linestyle="-",alpha=0.5,zorder=110)
ax3.plot(range(len(dfNSE.columns)),dfNSE.loc[df["UPAREA"]>=2.5e4].median(axis=0),marker="o",color="k",linestyle="--",alpha=0.5,zorder=110)

# KGE plot
ax4 = fig.add_subplot(G[1,1])
plot_boxplot(dfKGE,4,dfKGE.columns,"$KGE$",colors=newcmp,ax=ax4,ylim=[-2.02,1.02])
ax4.plot(range(len(dfKGE.columns)),dfKGE.median(axis=0),marker="o",color="grey",linestyle="-",alpha=0.5,zorder=110)
ax4.plot(range(len(dfKGE.columns)),dfKGE.loc[df["UPAREA"]>=2.5e4].median(axis=0),marker="o",color="k",linestyle="--",alpha=0.5,zorder=110)

def print_stat(df,df0,stat):
    print ("statistics of",stat)

    print ("median of ",stat, "UPAREA>=1.0e4")
    print (df.median(axis=0))

    # print ("percentage of positive values",stat, "UPAREA>=1.0e4")
    # print ((df.select_dtypes(include='float64').gt(0).sum(axis=0)/len(df))*100.0)

    print ("median of ",stat, "UPAREA>=2.5e4")
    print (df.loc[df0["UPAREA"]>=2.5e4].median(axis=0))

    # print ("percentage of positive values",stat, "UPAREA>=2.5e4")
    # print ((df.loc[df0["UPAREA"]>=2.5e4].select_dtypes(include='float64').gt(0).sum(axis=0)/len(df.loc[df0["UPAREA"]>=2.5e4]))*100.0)

print_stat(dfBR,df,"BR")
print_stat(dfCC,df,"CC")
print_stat(dfNSE,df,"NSE")
print_stat(dfKGE,df,"KGE")

# ax2.axhline(opn_weight,marker="o",color="k",linestyle="--")
# sns.pointplot(data=dfNRMSEopn, estimator=np.mean ,ax=ax1, color="green")
# print (dfrKGE.describe())
# print (dfrKGE.select_dtypes(include='float64').gt(0).sum(axis=0))

# rCC boxplot
# print ("----> rCC")
# ax2 = fig.add_subplot(G[2,0])
# plot_boxplot(dfrCC,2,labels,"$rCC$",colors=cmap,ax=ax2,ylim=[-1.2,1.2])
# # print (dfrCC.describe())
# # print (dfrCC.select_dtypes(include='float64').gt(0).sum(axis=0))
# # print (dfrCC.select_dtypes(include='float64').gt(0).sum(axis=0)/len(dfrCC)*100.0)
# # rBR boxplot
# print ("----> rBR")
# ax3 = fig.add_subplot(G[2,1])
# plot_boxplot(dfrBR,3,labels,"$rBR$",colors=cmap,ax=ax3,ylim=[-1.2,1.2])
# # print (dfrBR.describe())
# # print (dfrBR.select_dtypes(include='float64').gt(0).sum(axis=0))
# # print (dfrBR.select_dtypes(include='float64').gt(0).sum(axis=0)/len(dfrBR)*100.0)
# # rRV boxplot
# print ("----> rRV")
# ax4 = fig.add_subplot(G[2,2])
# plot_boxplot(dfrRV,4,labels,"$rRV$",colors=cmap,ax=ax4,ylim=[-1.2,1.2])
# # print (dfrRV.describe())
# # print (dfrRV.select_dtypes(include='float64').gt(0).sum(axis=0))
# # print (dfrRV.select_dtypes(include='float64').gt(0).sum(axis=0)/len(dfrRV)*100.0)
# #--
plt.subplots_adjust(wspace=0.05,hspace=0.1)
plt.tight_layout()
#--
print ("./figures/"+figname+".png")
plt.savefig("./figures/"+figname+".pdf",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".png",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".jpg",dpi=800,bbox_inches="tight", pad_inches=0.01)
#plt.show()