#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
Create pdf file with all the characteristics with DA performance
- Local patch characteristics
- Physical characteristics
'''
import numpy as np
from numpy import ma
import sys
import matplotlib.pyplot as plt
import datetime
import math
from matplotlib.colors import LogNorm,Normalize,ListedColormap,BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import re
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import warnings;warnings.filterwarnings('ignore')
#====================================================================
sys.path.append('./src')
# import params as pm
import read_grdc as grdc
import read_cgls as cgls
#=========================================
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier
#=============================
def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n*multiplier - 0.5) / multiplier
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
def plot_hydrograph(grdcid,station,experiments,rKGEs,ax=None,syear=2016,eyear=2020):
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
    #============
    # plt.close()
    # # fig=plt.figure(figsize=(wdt,hgt))
    # # G = gridspec.GridSpec(ncols=1, nrows=1)
    # # ax=plt.subplot(G[0,0])
    # fig, ax = plt.subplots(figsize=(12, 4))
    # read grdc
    grdcid=int(grdcid)
    org=grdc.grdc_dis(str(grdcid),syear,eyear)
    org=np.array(org)
    ax=plt.gca() or ax
    ax.plot(np.arange(start,last),ma.masked_less(org,0.0),label="GRDC",color="#34495e",linewidth=3.0,zorder=101) #,marker = "o",markevery=swt[point])
    for i,experiment in enumerate(experiments):
        # read discharge
        asm,opn=read_dis(experiment,str(grdcid),syear,eyear)
        # plot
        linewidth=0.5 #1.0/float(len(labels))
        ax.plot(np.arange(start,last),asm,label=exp,color=cmap(norm(i)),linewidth=linewidth,alpha=1,zorder=105)
    ax.plot(np.arange(start,last),opn,label="open-loop",color="grey",linewidth=1.5,linestyle="--",alpha=1,zorder=104)
    # print (last,len(opn),len(org),syear,eyear)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax.set_ylabel('discharge ($m^3s^{-1}$)', color='k',fontsize=6)
    # ax.yaxis.label.set_size(10)
    ax.set_xlim(xmin=0,xmax=last+1)
    ax.tick_params('y', colors='k')
    # scientific notation
    ax.ticklabel_format(style="sci",axis="y",scilimits=(0,0),useOffset=1,useLocale=False,useMathText=True)
    # ax.yaxis.major.formatter._useMathText=True
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.offsetText.set_fontsize(6)
    ax.yaxis.get_offset_text().set_x(-0.02)
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
    ax.set_xticklabels(xxlab,fontsize=6)
    # ax.set_xticklabels(xxlab,fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    # ax.text(0.05,1.10,"%s"%(station)
    #     ,ha="left",va="center",transform=ax.transAxes,fontsize=6)
    ax.text(0.0,1.05,str(grdcid)+" : "+station.strip()
        ,ha='left',transform=ax.transAxes)
    # legend 
    features=[]
    pnum=len(labels)
    for i in np.arange(pnum):
        label=labels[i]+"(%03.2f)"%(rKGEs[i])
        features.append(mlines.Line2D([], [], color=cmap(norm(i)),label=label))
    legend=plt.legend(handles=features,loc="best",bbox_transform=fig.transFigure, 
        ncol=3,  borderaxespad=0.0, frameon=False,prop={'size': 6})#
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
# #===================
# dfopnl=pd.read_csv(opnlist, sep=';')
# # dfopnl.set_index('ID',inplace=True)
# print (dfopnl.head())
#===================
# dfout = pd.DataFrame()
# ii=0
# for exp,label in zip(experiments,labels):
#     print (ii, exp, label)
#     dfname="./out/"+exp+"/datafile.csv"
#     # print (dfname)
#     df = pd.read_csv(dfname, sep=';')
#     # add column with experiment label
#     # print (df.head())
#     df = df.loc[(df["SAT_COV"]==1.0) & (df["UPAREA"]>=upa_thr) & (df["RIVNUM"]<=num_thr) & (df["RIVNUM"]<=7),:]
#     df.rename(columns={'GRDC_ID':'ID'}, inplace=True)
#     print (df.columns)
#     # df.set_index('ID',inplace=True)
#     # df.index.astype(dfopnl.index.dtypes.to_dict())
#     # print (df.head())
#     df_=pd.merge(df,dfopnl[['ID','minRMSE(open-loop)','minBias(open-loop)','minNSE(open-loop)',
#             'minKGE(open-loop)','minCC(open-loop)','minBR(open-loop)','minRV(open-loop)','minKGED(open-loop)',
#             'meanRMSE(open-loop)','meanBias(open-loop)','meanNSE(open-loop)',
#             'meanKGE(open-loop)','meanCC(open-loop)','meanBR(open-loop)','meanRV(open-loop)','meanKGED(open-loop)',
#             'maxRMSE(open-loop)','maxBias(open-loop)','maxNSE(open-loop)',
#             'maxKGE(open-loop)','maxCC(open-loop)','maxBR(open-loop)','maxRV(open-loop)','maxKGED(open-loop)']],on='ID')
#     df_.dropna(inplace=True)
#     df_['label']=[label]*len(df_)
#     # print ([label]*len(df_))
#     # df_["log(UPAREA)"]=np.log10(df["UPAREA"])
#     # print (df_.head())
#     # df["ELEVTN"]=[elevtn[iy,ix] for iy,ix in zip(df["IY1"].values,df["IX1"].values)]
#     if ii==0:
#         dfout=df_.copy()
#     else:
#         dfout=dfout.append(df_) #, ignore_index = True)
#     ii=ii+1
#     print (dfout.head()) #, len(dfout))#.head())
#     # dfout = pd.concat([dfout,df_], axis=0, ignore_index = True) #, sort=False)
# #===================
# # dfout.dropna(axis=1,inplace=True)
# dfout.dropna(axis='columns',how='any',inplace=True)
# print (dfout)
# # print (dfout.isna().any())

# dfout["BRopn[WSE]"]=1-np.abs(dfout["meanBR(open-loop)"]-1)
# dfout["log(UPAREA)"]=np.log10(dfout["UPAREA"].values)

# # dfout["DA_method"]=[mth.split("_")[0] for mth in dfout['label']]
# # dfout["Obs_type"]=[mth.split("_")[1] for mth in dfout['label']]
# print ("#===================")
# print (dfout.head())

#===================
dfopnl_wse=pd.read_csv(opnlist, sep=';')
dfopnl_wse.rename(columns=lambda x: x.strip())
# dfopnl.set_index('ID',inplace=True)
dfopnl_wse.set_index('ID',inplace=True)
print ('Open-loop WSE')
print (dfopnl_wse.head())

#===================
opnlist_dis="./out/"+experiments[0]+"/datafile.csv"
dfopnl_dis=pd.read_csv(opnlist_dis, sep=';')
dfopnl_dis.rename(columns=lambda x: x.strip())
print (dfopnl_dis.head())
print (dfopnl_dis.columns)
# dfopnl.set_index('ID',inplace=True)
dfopnl_dis=dfopnl_dis.loc[:,['GRDC_ID','IX1','IY1','IX2','IY2','LON',
'LAT','OBS AVAILABILITY','KGEopn','CCopn','BRopn','RVopn','NSEasm','NSEopn',
'NRMSEopn','RIV_WTH','UPAREA','RIVNUM','SAT_COV','ELEV']]
dfopnl_dis.rename(columns={'GRDC_ID':'ID'}, inplace=True)
dfopnl_dis.set_index('ID',inplace=True)
# dfopnl_dis.index.astype(dfopnl_wse.index.dtypes.to_dict())
print ('Open-loop DIS')
print (dfopnl_dis.head())

dfout=pd.merge(dfopnl_wse,dfopnl_dis,on='ID')
dfout.dropna(inplace=True)

print (dfout.head())

for exp,label in zip(experiments,labels):
    print (exp)
    dfname="./out/"+exp+"/datafile.csv"
    # print (dfname)
    df = pd.read_csv(dfname, sep=';')
    # print (df.head())
    # print (df[df['GRDC_ID'].isin(dfopnl_wse.index)]["rKGE"])
    dfout[label+"_KGE"]=df[df['GRDC_ID'].isin(dfopnl_wse.index)]["KGEasm"].values
    dfout[label+"_rKGE"]=df[df['GRDC_ID'].isin(dfopnl_wse.index)]["rKGE"].values
    dfout[label+"_rBR"]=df[df['GRDC_ID'].isin(dfopnl_wse.index)]["rBR"].values
    dfout[label+"_rCC"]=df[df['GRDC_ID'].isin(dfopnl_wse.index)]["rCC"].values
    dfout[label+"_rRV"]=df[df['GRDC_ID'].isin(dfopnl_wse.index)]["rRV"].values
#==========================================
# dfout = dfout.loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7),:]
# dfout["max_rKGE"]=dfout[labels].idxmax(axis=1)
# dfout["max_rKGE_int"]=dfout["max_rKGE"] #pd.Categorical(dfout["max_rKGE"]).codes
# dfout["max_rKGE_int"].replace(labels,range(len(labels)),inplace=True)
# add lon lat
# dfout["LON"]=dfout["LON"].loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7)]
# dfout["LAT"]=dfout["LAT"].loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7)]
# dfout["GRDC_ID"]=dfout["GRDC_ID"].loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7)]
# dfout["RIVER"]=dfout["RIVER"].loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7)]
# dfout["STATION"]=dfout["STATION"].loc[(dfout["SAT_COV"]==1.0) & (dfout["UPAREA"]>=upa_thr) & (dfout["RIVNUM"]<=num_thr) & (dfout["RIVNUM"]<=7)]
# print (dfout[labels].head())
#
# print (dfout["max_rKGE"].value_counts())
# print (len(dfout["max_rKGE"].values))
# print ((dfout["max_rKGE"].value_counts()/float(len(dfout["max_rKGE"].values)))*100.0)

dfout.rename(columns=lambda x: x.strip(), inplace=True)

print (dfout.head())
print (dfout.columns)
################################
KGEasmlabels=[label+"_KGE" for label in labels]
KGElabels=[label+"_rKGE" for label in labels]
CClabels=[label+"_rCC" for label in labels]
BRlabels=[label+"_rBR" for label in labels]
RVlabels=[label+"_rRV" for label in labels]
#===================
# colorbar
cmap=plt.cm.get_cmap("tab20c")
norm=BoundaryNorm(np.arange(0,20+0.1,1),cmap.N)
#====================================================================
va_margin= 0.0#1.38#inch 
ho_margin= 0.0#1.18#inch
hgt=(11.69 - 2*va_margin)*(1.0/1.0)
wdt=(8.27 - 2*ho_margin)*(2.0/2.0)
# fig, ax = plt.subplots(1, 1)
# dfout_plot=dfout[dfout['label']=="DIR_All_Emp"] #.pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
# ax.scatter(dfout_plot['BRopn'].values, dfout_plot['minKGED(open-loop)'].values,c=dfout_plot['rKGE'].values, cmap="viridis_r")#, markersize=5)
# get the dimesion of the map
dec=2
val=0.20
#====================================================================
pdfname = "./pdffigure/DA_characteristics.pdf"
with PdfPages(pdfname) as pdf:
    # for label in labels:
    #     plt.close()
    #     dfout_plot=dfout[dfout['label']==label] #.pivot_table(index="BRopn", columns="minKGED(open-loop)", values="rKGE", aggfunc='mean')
    for point in dfout.index[0::]:
        plt.close()
        lon = dfout['LON'][point]
        lat = dfout['LAT'][point]
        grdcid=point #dfout["GRDC_ID"][point]
        station=dfout["Station"][point]
        river=dfout["River"][point]
        uparea=dfout["UPAREA"][point]*1e-6
        elevtn=dfout["ELEV"][point]
        obsava=dfout["OBS AVAILABILITY"][point]
        fig=plt.figure(figsize=(wdt,hgt))
        G   = gridspec.GridSpec(ncols=3, nrows=4)
        ax1 = fig.add_subplot(G[0,0])
        ax1.text(0.0,1.1,station+' - '+river,va="center",ha="left",transform=ax1.transAxes,fontsize=10)
        lllat = round_half_down(lat-val,dec)
        urlat = round_half_up(lat+val,dec)
        lllon = round_half_down(lon-val,dec)
        urlon = round_half_up(lon+val,dec)
        m = Basemap(projection='cyl',llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon, lat_ts=0,resolution='c',ax=ax1)
        try:
            # m.arcgisimage(service=maps[1], xpixels=1500, verbose=False)
            m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', xpixels=1000, ypixels=None, dpi=1200)
            print ("ArcGIS map")
        except:
            # Draw some map elements on the map
            m.drawcoastlines()
            m.drawstates()
            m.drawcountries()
            # m.drawrivers(color='blue')
            print ("Normal map")
        ax1.plot(lon ,lat ,color="r",marker="o",label="GRDC",markersize=7,linewidth=0,zorder=111) #fillstyle="none",
        # add text 
        ax2 = fig.add_subplot(G[0,1::])
        # ax2.text(0.0,1.0,'River=%s'%(river),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        # ax2.text(0.0,0.8,'Station=%s'%(station),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.0,1.0,'Uparea=%.2E$km^2$'%(uparea),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.5,1.0,'Elevation=%6.2f$m$'%(elevtn),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.0,0.9,'Satellite Covered=%d'%(obsava),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        # WSE realted
        ax2.text(0.5,0.8,'$KGED_{WSE}$(open-loop)=%3.2f,%3.2f,%3.2f'%(dfout["minKGED(open-loop)"][point],
        dfout["meanKGED(open-loop)"][point],dfout["maxKGED(open-loop)"][point])
        ,va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.5,0.6,'$BR_{WSE}$(open-loop)=%3.2f,%3.2f,%3.2f'%(dfout["minBR(open-loop)"][point],
        dfout["meanBR(open-loop)"][point],dfout["maxBR(open-loop)"][point])
        ,va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.5,0.4,'$CC_{WSE}$(open-loop)=%3.2f,%3.2f,%3.2f'%(dfout["minCC(open-loop)"][point],
        dfout["meanCC(open-loop)"][point],dfout["maxCC(open-loop)"][point])
        ,va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.5,0.2,'$RV_{WSE}$(open-loop)=%3.2f,%3.2f,%3.2f'%(dfout["minRV(open-loop)"][point],
        dfout["meanRV(open-loop)"][point],dfout["maxRV(open-loop)"][point])
        ,va="center",ha="left",transform=ax2.transAxes,fontsize=10)

        # discharge realated
        ax2.text(0.0,0.8,'$KGE_Q$(open-loop)=%3.2f'%(dfout["KGEopn"][point]),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.0,0.6,'$BR_Q$(open-loop)=%3.2f'%(dfout["BRopn"][point]),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.0,0.4,'$CC_Q$(open-loop)=%3.2f'%(dfout["CCopn"][point]),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.text(0.0,0.2,'$RV_Q$(open-loop)=%3.2f'%(dfout["RVopn"][point]),va="center",ha="left",transform=ax2.transAxes,fontsize=10)
        ax2.set_frame_on(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        # plot hydrograph
        ax3=plt.subplot(G[1,:])
        rKGEs=dfout.loc[point,KGElabels]
        plot_hydrograph(grdcid,station,experiments,rKGEs,ax3)
        # table
        dfTable=pd.DataFrame()
        dfTable['Experiment']=labels
        dfTable['KGE']=dfout.loc[point,KGEasmlabels].values
        dfTable['rKGE']=dfout.loc[point,KGElabels].values
        dfTable['rBR']=dfout.loc[point,BRlabels].values
        dfTable['rCC']=dfout.loc[point,CClabels].values
        dfTable['rRV']=dfout.loc[point,RVlabels].values
        dfTable.update(dfTable[['KGE','rKGE','rBR','rCC','rRV']].applymap('{:,.2f}'.format))
        # print (dfTable.head())
        ax3 = fig.add_subplot(G[2,:])
        ax3.table(cellText=dfTable.values, colLabels=dfTable.columns, cellLoc='center', colLoc='center', loc='center')
        ax3.axis('off')
        # ax3.axes.get_xaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
        # ax3.spines['top'].set_visible(False)
        # ax3.spines['right'].set_visible(False)
        # ax3.spines['bottom'].set_visible(False)
        # ax3.spines['left'].set_visible(False)
        # for i,met in enumerate(["minKGED(open-loop)","meanKGED(open-loop)","maxKGED(open-loop)"]):
        #     ax=plt.subplot(G[0,i])
        #     sns.scatterplot(data=dfout_plot, x="KGEopn", y=met, hue="rKGE", size="log(UPAREA)",ax=ax, palette='RdBu', legend=None) #style="label",
        # for i,met in enumerate(["minCC(open-loop)","meanCC(open-loop)","maxCC(open-loop)"]):
        #     ax=plt.subplot(G[1,i])
        #     sns.scatterplot(data=dfout_plot, x="CCopn", y=met, hue="rKGE", size="log(UPAREA)",ax=ax, palette='RdBu', legend=None) #style="label",
        # for i,met in enumerate(["minBR(open-loop)","meanBR(open-loop)","maxBR(open-loop)"]):
        #     ax=plt.subplot(G[2,i])
        #     sns.scatterplot(data=dfout_plot, x="BRopn", y=met, hue="rKGE", size="log(UPAREA)",ax=ax, palette='RdBu', legend=None) #style="label",
        # for i,met in enumerate(["minRV(open-loop)","meanRV(open-loop)","maxRV(open-loop)"]):
        #     ax=plt.subplot(G[3,i])
        #     sns.scatterplot(data=dfout_plot, x="RVopn", y=met, hue="rKGE", size="log(UPAREA)",ax=ax, palette='RdBu', legend=None) #style="label",
        #===============================
        # fig.suptitle(label, fontsize=10)
        # axtext=
        # plt.text(0.5,1.05,label,ha='center',transform=ax.transAxes)
        #=====================================================================
        # norm = plt.Normalize(dfout_plot['rKGE'].min(), dfout_plot['rKGE'].max())
        # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        # sm.set_array([])

        # # Remove the legend and add a colorbar
        # # ax.get_legend().remove()
        # cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
        # cbar=fig.colorbar(sm, cax=cax, orientation="horizontal")
        # cbar.ax.tick_params(labelsize=8) 
        # # cbar.ax.set_title('$rKGE$',fontsize=10)
        # cbar.ax.set_xlabel('$rKGE$',fontsize=10)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12)

        pdf.savefig()
    # metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Hydrographs of all experiments together'
    d['Author'] = u'Menaka Revel'
    d['Subject'] = 'Hydrographs of all experiments together'
    d['Keywords'] = 'Hydrographs'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()