#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
making hydrograph for give GRCD ID with confidence intervals
'''
import matplotlib
matplotlib.use('agg')  # or 'pdf' if 'agg' doesn't work
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
import datetime
import sys
import os
import re
import math
from multiprocessing import Pool
import read_grdc as grdc
from statistics import *
#====================================================================
#
#====================================================================
def make_fig(syear,eyear,station,expname,gauge):
    # read data
    # ua, la, uo, lo = read_ul_all(expname,station)
    asm, opn = read_dis([expname,station,syear,eyear])
    # org=grdc.grdc_dis(str(station),syear,eyear)
    # org=np.array(org)
    org=get_grdc(station,syear,eyear)
    #=============
    # make figure
    #=============
    plt.close()
    #labels=["GRDC","corrupted","assimilated"]
    labels=["GRDC","open-loop","assimilated"]
    fig, ax1 = plt.subplots(figsize=(12, 4))
    lines=[ax1.plot(np.arange(start,last),ma.masked_less(org,0.0),label="GRDC",color="#34495e",linewidth=3.0,zorder=101)[0]] #,marker = "o",markevery=swt[point])
    # draw mean of ensembles
    lines.append(ax1.plot(np.arange(start,last),opn,label="open-loop",color="#4dc7ec",linewidth=2.0,linestyle="--",alpha=1.0,zorder=104)[0]) #"#4dc7ec"
    lines.append(ax1.plot(np.arange(start,last),asm,label="assimilated",color="#ff8021",linewidth=1.0,alpha=1.0,zorder=106)[0])
    # ax1.fill_between(np.arange(start,last),lo,uo,color="#4dc7ec",alpha=0.2,zorder=102)
    # ax1.fill_between(np.arange(start,last),la,ua,color="#ff8021",alpha=0.2,zorder=103)
    # print ua
    # print asm
    # print la
    #    plt.ylim(ymin=)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('discharge (m$^3$/s)', color='k',fontsize=10)
    ax1.set_xlim(xmin=0,xmax=last+1)
    ax1.tick_params('y', colors='k')
    # scentific notaion
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=1,integer=True))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
    ax1.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    ax1.yaxis.major.formatter._useMathText=True 
    ax1.yaxis.offsetText.set_fontsize(10) 
    # ax1.yaxis.label.set_size(10)
    ax1.yaxis.get_offset_text().set_x(-0.05)
    #
    #xxlist=np.linspace(0,N,(eyear-syear)+1)
    #xlab=np.arange(syear,eyear+1,1)
    #xxlab=[calendar.month_name[i][:3] for i in range(1,13)]
    # if eyear-syear > 5:
    #     dtt=1
    #     dt=int(math.ceil(((eyear-syear)+1)/dtt))
    # elif eyear-syear > 10:
    #     dtt=5
    #     dt=int(math.ceil(((eyear-syear)+1)/5.0))
    # else:
    #     dtt=1
    #     dt=(eyear-syear)+1
    dtt=1
    dt=(eyear-syear)+2
    xxlist=np.linspace(0,last,dt,endpoint=True)
    #xxlab=[calendar.month_name[i][:3] for i in range(1,13)]
    xxlab=np.arange(syear,eyear+2,dtt)
    ax1.set_xticks(xxlist)
    ax1.set_xticklabels(xxlab,fontsize=10)
    # NSEval=calc_NSE([asm,opn,org])
    # ax1.set_title(str(station)+","+gauge.strip()+" NSEasm:%3.2f"%(NSEval[0])+" NSEopn:%3.2f"%(NSEval[1]))
    return 0
#====================================================================
def read_dis0(inputlist):
    experiment=str(inputlist[0])
    station=str(inputlist[1])
    syear=int(inputlist[2])
    eyear=int(inputlist[3])
    asm=[]
    opn=[]
    fname="./txt/"+experiment+"/outflow/"+station+".txt"
    # print (fname)
    with open(fname,"r") as f:
        lines=f.readlines()
    for line in lines:
        line = re.split(" ",line)
        line = list(filter(None, line))
        year=int(line[0][0:4])
        if year<syear or year>eyear:
            continue
        # print (year, syear, eyear)
        asm.append(float(line[1]))
        opn.append(float(line[2]))
    return np.array(asm), np.array(opn)
#====================================================================
def read_dis(inputlist):
    experiment=str(inputlist[0])
    station=str(inputlist[1])
    syear=int(inputlist[2])
    eyear=int(inputlist[3])
    fname="./txt/"+experiment+"/outflow/"+station+".txt"
    if not os.path.exists(fname):
        return np.ones([last],np.float32)*-9999.0, np.ones([last],np.float32)*-9999.0
    else:
        data = pd.read_csv(fname, delim_whitespace=True, names=["Date", "asm", "opn"])
        data["Date"]=pd.to_datetime(data["Date"], format='%Y%m%d')
        data=data.set_index("Date")[str(syear)+'-01-01':str(eyear)+'-12-31']
        return data["asm"].values, data["opn"].values
#====================================================================
def get_grdc(station,syear,eyear):
    grdc="/cluster/data7/menaka/GRDC_2021/"+str(station)+"_Q_Day.Cmd.txt"
    if not os.path.exists(grdc):
        return np.ones([last],np.float32)*-9999.0
    else:
        df_grdc=pd.read_csv(grdc, header=37, sep=";", names=["Date","Time","Discharge"])
        df_grdc["Date"]=pd.to_datetime(df_grdc["Date"])
        df_grdc=df_grdc.set_index("Date")[str(syear)+'-01-01':str(eyear)+'-12-31']
        idx = pd.date_range(str(syear)+'-01-01', str(eyear)+'-12-31')
        df_grdc=df_grdc.reindex(idx, fill_value=-9999.0)
        return df_grdc["Discharge"].values
#====================================================================
# Calculate Nash-Sutcliffe efficiency
def calc_NSE(inputlist):
    asm_ = inputlist[0]
    opn_ = inputlist[1]
    org_ = inputlist[2]
    NSEasm = NSE(asm_,org_)
    NSEopn = NSE(opn_,org_)
    return [NSEasm, NSEopn]
#====================================================================
syear=int(sys.argv[1])
eyear=int(sys.argv[2])
expname=sys.argv[3]
mapname=sys.argv[4]
CaMadir=sys.argv[5]
ncpus=int(sys.argv[6])
#====================================================================
start_dt=datetime.date(syear,1,1)
end_dt=datetime.date(eyear,12,31)
size=60
start=0
last=(end_dt-start_dt).days + 1
N=int(last)
#====================================================================
wth_thr=50.0
upa_thr=1.0e4 #2.5e4 #
num_thr=1
#====================================================================
obslist="./out/"+expname+"/datafile.csv"
dfobs=pd.read_csv(obslist, sep=";")
dfobs=dfobs.loc[(dfobs["RIVNUM"]<=num_thr)]
stations=dfobs["GRDC_ID"].values
gauges=dfobs["STATION"].values
#====================================================================
# pdfname="./out/"+expname+"/hydrogrphy.pdf"
# with PdfPages(pdfname) as pdf:
#     # for station, gauge in zip(stations,gauges):
#     for index in dfobs.index[0:3]:
#         print ( "making figure -->", dfobs["GRDC_ID"][index] ,dfobs["STATION"][index].strip())
#         make_fig(syear,eyear,dfobs["GRDC_ID"][index],expname,dfobs["STATION"][index])
#         ax1=plt.gca()
#         ax1.text(0.0,1.05,str(dfobs["GRDC_ID"][index])+" : "+dfobs["STATION"][index].strip(),ha='left',transform=ax1.transAxes)
#         ax1.text(1.0,1.05,"NSEasm:%3.2f"%(dfobs["NSEasm"][index])+" NSEopn:%3.2f"%(dfobs["NSEopn"][index])
#         +" CCasm:%3.2f"%(dfobs["CCasm"][index])+" CCopn:%3.2f"%(dfobs["CCopn"][index]),ha='right',transform=ax1.transAxes)
#         pdf.savefig()
#         # plt.close()
#     # metadata via the PdfPages object:
#     d = pdf.infodict()
#     d['Title'] = 'Hydrographs of '+expname
#     d['Author'] = u'Menaka Revel'
#     d['Subject'] = 'Hydrographs of '+expname
#     d['Keywords'] = 'Hydrographs'
#     d['CreationDate'] = datetime.datetime.today()
#     d['ModDate'] = datetime.datetime.today()

def generate_fig(index):
    print ( "making figure -->", dfobs["GRDC_ID"][index] , dfobs["RIVER"][index].strip(), dfobs["STATION"][index].strip())
    # Create a new figure for each iteration
    # fig=plt.figure()
    make_fig(syear,eyear,dfobs["GRDC_ID"][index],expname,dfobs["STATION"][index])
    ax1=plt.gca()
    ax1.text(0.0,1.05,str(dfobs["GRDC_ID"][index])+" : "+dfobs["STATION"][index].strip(),ha='left',transform=ax1.transAxes)
    ax1.text(1.0,1.05,"rKGE:%3.2f"%(dfobs["rKGE"][index])+" KGEasm:%3.2f"%(dfobs["KGEasm"][index])+" KGEopn:%3.2f"%(dfobs["KGEopn"][index])
    +" BRasm:%3.2f"%(dfobs["BRasm"][index])+" BRopn:%3.2f"%(dfobs["BRopn"][index])
    +" CCasm:%3.2f"%(dfobs["CCasm"][index])+" CCopn:%3.2f"%(dfobs["CCopn"][index])
    +" Sat:%d"%(dfobs["SAT_COV"][index]),ha='right',transform=ax1.transAxes)
    pdf.savefig()
    plt.close()
    # return ax1


pdfname = "./out/" + expname + "/hydrograph.pdf"
with PdfPages(pdfname) as pdf:
    # P = Pool(processes=10) #multiprocessing.cpu_count())
    # index_list = dfobs.index[0:10]
    # axlist=P.map(generate_fig, index_list)
    # P.close()
    # P.join()
    # for ax1 in axlist:
    #     pdf.savefig()
    for index in dfobs.index:
        generate_fig(index)
    # metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Hydrographs of ' + expname
    d['Author'] = u'Menaka Revel'
    d['Subject'] = 'Hydrographs of ' + expname
    d['Keywords'] = 'Hydrographs'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()

