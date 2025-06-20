#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import LogNorm,Normalize,ListedColormap,BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import re
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
#====================================================================
#====================================================================
argv=sys.argv
syear=int(argv[1])
eyear=int(argv[2])
CaMa_dir=argv[3]
mapname=argv[4]
exlist=argv[5]
stlist=argv[6]
opnlist=argv[7]
figname=argv[8]
ncpus=int(sys.argv[9])
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
#====================================================================
# River width threshold
wth_thr=50.0
upa_thr=1.0e4
num_thr=1
#===================
# dfname="./out/"+expname+"/datafile.csv"
# df = pd.read_csv(dfname, sep=';')
# df = df.loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7)]
# df["log(UPAREA)"]=np.log10(df["UPAREA"])
# df["ELEVTN"]=[elevtn[iy,ix] for iy,ix in zip(df["IY1"],df["IX1"])]
#===================
dfopnl=pd.read_csv(opnlist, sep=';')
# dfopnl.set_index('ID',inplace=True)
print (dfopnl.head())
#===================
dfout = pd.DataFrame()
ii=0
for exp,label in zip(experiments,labels):
    print (ii, exp, label)
    dfname="./out/"+exp+"/datafile.csv"
    # print (dfname)
    df = pd.read_csv(dfname, sep=';')
    # add column with experiment label
    # print (df.head())
    df = df.loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7),:]
    df.rename(columns={'GRDC_ID':'ID'}, inplace=True)
    print (df.columns)
    # df.set_index('ID',inplace=True)
    # df.index.astype(dfopnl.index.dtypes.to_dict())
    # print (df.head())
    df_=pd.merge(df,dfopnl[['ID','minRMSE(open-loop)','minBias(open-loop)','minNSE(open-loop)',
            'minKGE(open-loop)','minCC(open-loop)','minBR(open-loop)','minRV(open-loop)','minKGED(open-loop)',
            'meanRMSE(open-loop)','meanBias(open-loop)','meanNSE(open-loop)',
            'meanKGE(open-loop)','meanCC(open-loop)','meanBR(open-loop)','meanRV(open-loop)','meanKGED(open-loop)',
            'maxRMSE(open-loop)','maxBias(open-loop)','maxNSE(open-loop)',
            'maxKGE(open-loop)','maxCC(open-loop)','maxBR(open-loop)','maxRV(open-loop)','maxKGED(open-loop)']],on='ID')
    df_.dropna(inplace=True)
    df_['label']=[label]*len(df_)
    # print ([label]*len(df_))
    # df_["log(UPAREA)"]=np.log10(df["UPAREA"])
    # print (df_.head())
    # df["ELEVTN"]=[elevtn[iy,ix] for iy,ix in zip(df["IY1"].values,df["IX1"].values)]
    if ii==0:
        dfout=df_.copy()
    else:
        dfout=dfout.append(df_) #, ignore_index = True)
    ii=ii+1
    print (dfout.head()) #, len(dfout))#.head())
    # dfout = pd.concat([dfout,df_], axis=0, ignore_index = True) #, sort=False)
#===================
# dfout.dropna(axis=1,inplace=True)
dfout.dropna(axis='columns',how='any',inplace=True)
print (dfout)
# print (dfout.isna().any())

dfout["BRopn[WSE]"]=1-np.abs(dfout["meanBR(open-loop)"]-1)

# dfout["DA_method"]=[mth.split("_")[0] for mth in dfout['label']]
# dfout["Obs_type"]=[mth.split("_")[1] for mth in dfout['label']]
print ("#===================")
print (dfout.head())
#===================
# colorbar
cmap=plt.cm.get_cmap("tab20c")
norm=BoundaryNorm(np.arange(0,20+0.1,1),cmap.N)
#====================================================================
va_margin= 0.0#1.38#inch 
ho_margin= 0.0#1.18#inch
hgt=(11.69 - 2*va_margin)*(1.0/3.0)
wdt=(8.27 - 2*ho_margin)*(2.0/2.0)
# fig, ax = plt.subplots(1, 1)
# dfout_plot=dfout[dfout['label']=="DIR_All_Emp"] #.pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
# ax.scatter(dfout_plot['BRopn'].values, dfout_plot['minKGED(open-loop)'].values,c=dfout_plot['rKGE'].values, cmap="viridis_r")#, markersize=5)

dfout_plot=dfout[dfout['label']=="NOM_All_Emp_050"] #.pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
fig=plt.figure(figsize=(wdt,hgt))
G = gridspec.GridSpec(ncols=3, nrows=1)
for i,met in enumerate(["minKGED(open-loop)","meanKGED(open-loop)","maxKGED(open-loop)"]):
    ax=plt.subplot(G[0,i])
    sns.scatterplot(data=dfout_plot, x="KGEopn", y=met, hue="rKGE", size="UPAREA",ax=ax, legend=None) #style="label",


# ax.set_title("Scatter plot")
# ax=sns.lmplot(data=dfout, x="meanRMSE(open-loop)",y="rKGE", hue="label",legend ="brief")

# ax=sns.lmplot(data=dfout, x="BRopn",y="rKGE", hue="label",legend ="brief")

# ax=sns.lmplot(data=dfout, x="meanCC(open-loop)",y="rKGE", hue="label",legend ="brief")
# col="Obs_type",row="DA_method",legend ="brief")

# ax=sns.lmplot(data=dfout, x="minKGED(open-loop)",y="rKGE", hue="label",legend ="brief")
# dfout_plot=dfout[dfout['label']=="DIR_All_Emp"].pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
# ax=sns.heatmap(dfout_plot, cmap="crest")

# dfout_plot=dfout[dfout['label']=="DIR_All_Emp"].pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
# x=dfout[dfout['label']=="DIR_All_Emp"]["BRopn"]
# y
# heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin='lower')
# plt.show()
# hgt=wdt
#fig=plt.figure(figsize=(8.27,11.69))
# fig=plt.figure(figsize=(wdt,hgt))
#fig.suptitle("auto-correalated area")#"maximum autocorrelation length")
# G = gridspec.GridSpec(ncols=1, nrows=1)
# ax=plt.subplot(G[0,0])
# sns.regplot(data=dfout, x="ELEVTN",y="rKGE",hue="label",palette=cmap,legend ="brief",ax=ax)
# ax=sns.lmplot(data=dfout, x="log(UPAREA)",y="rKGE",hue="label",legend ="brief") #palette=cmap,
# # ax=sns.lmplot(data=dfout, x="log(UPAREA)",y="rKGE",col="Obs_type",row="DA_method",legend ="brief")
# sns.scatterplot(data=df, x="ELEVTN",y="rKGE",hue="log(UPAREA)",style="OBS AVAILABILITY",palette="viridis_r",legend ="brief",ax=ax)
# sns.regplot(data=df, x="ELEVTN",y="rKGE",color="#8FA2B0",scatter="false",scatter_kws={'s':2},ax=ax)

# norm = plt.Normalize(df['log(UPAREA)'].min(), df['log(UPAREA)'].max())
# sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
# sm.set_array([])

# # Remove the legend and add a colorbar
# ax.get_legend().remove()
# ax.figure.colorbar(sm)


plt.savefig("./figures/"+figname+".pdf",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".png",dpi=800,bbox_inches="tight", pad_inches=0.01)
plt.savefig("./figures/"+figname+".jpg",dpi=800,bbox_inches="tight", pad_inches=0.01)
