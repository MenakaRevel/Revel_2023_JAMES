#!/opt/local/bin/python
# -*- coding: utf-8 -*-
"""
Create a dataframe with the following columns:
    GRDC_ID: GRDC ID
    RIVER: Name of the river
    STATION: Name of the GRDC station
    LON: Longitude of the GRDC station
    LAT: Latitude of the GRDC station
    IX1: X-index of the GRDC station in CaMa-Flood
    IY1: Y-index of the GRDC station in CaMa-Flood
    IX2: X-index of the GRDC station in CaMa-Flood
    IY2: Y-index of the GRDC station in CaMa-Flood
    UPAREA: Upstream area of the GRDC station
    DIS_DAM: Distance to the nearest upstream dam
    DIS_TRI: Distance to the nearest downstream connecting tributary (> 1000 km2)
    rKGE: relative KGE value for the dam
    rCC: relative correlation coefficient for the dam
    rBR: relative bias ratio for the dam
    rRV: relative relative variability for the dam
    rPFT: relative peak flow timing for the dam
"""

import numpy as np
import sys
import datetime
import re
import pandas as pd
from multiprocessing import Pool, RawArray
from statistics import *
from read_CMF import read_discharge, read_discharge_multi
import read_grdc as grdc
import read_cgls as cgls
###############################
###  data reading functions  ##
###############################
#====================================================================
def read_dis_true_multi(ix1, iy1, ix2, iy2, syear, eyear, indir):
    #print ix1,iy1
    dis = np.zeros( (len(ix1), nbdays), 'f')
    for year in range(syear, eyear+1):
        # print year
        s_days = int( (datetime.date(year , 1,1) - datetime.date(syear, 1, 1)). days)
        e_days = int( (datetime.date(year+1, 1, 1) - datetime.date(syear, 1, 1)). days)
        
        f = indir + '/outflw'+str(year)+'.bin'

        print (f)
        tmp = read_discharge_multi( ix1, iy1, ix2, iy2, e_days-s_days, f, nx, ny)

        dis[:,s_days:e_days] = tmp

    return dis
#====================================================================
def read_dis_true(ix1, iy1, ix2, iy2, syear, eyear, indir):
    """
    Read CaMa-Flood discharge
    """
    dis = np.zeros( nbdays, 'f')
    # dis_max = np.zeros( nbyears, 'f')
    for year in range(syear, eyear+1):
        s_days = int( (datetime.date(year , 1,1) - datetime.date(syear, 1, 1)). days)
        e_days = int( (datetime.date(year+1, 1, 1) - datetime.date(syear, 1, 1)). days)
        
        f = indir + '/outflw'+str(year)+'.bin'

        # print (f)
        
        tmp = read_discharge( ix1+1, iy1+1, ix2+1, iy2+1, e_days-s_days, f, nx, ny)

        #print year, e_days - s_days, s_days, e_days, outflw.shape
        dis[s_days:e_days] = tmp

    return dis
#==================================
def read_dis(inputlist):
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
#==================================
def read_wse(inputlist):
    experiment=str(inputlist[0])
    station=str(inputlist[1])
    syear=int(inputlist[2])
    eyear=int(inputlist[3])
    asm=[]
    opn=[]
    fname="./txt/"+experiment+"/sfcelv/"+station+".txt"
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
#==================================
def read_xy(line):
    line = line.split()
    damid, ix, iy = int(line[0]), int(line[4]) - 1, int(line[5]) - 1
    return damid, ix, iy
#==================================
def read_data(inputlist):
    # Read DA data
    damid   = inputlist[0]
    expname = inputlist[1]
    # line    = re.split(" ",line)
    # line    = list(filter(None, line))
    # #---------------------------------
    num     = "%07d"%(int(damid))
    # ix      = int(line[4]) - 1
    # iy      = int(line[5]) - 1
    asm, opn = read_dis(expname, num)
    return asm, opn
#==================================
def read_grdc_data(inputlist):
    grdc_id=str(inputlist[0])
    syear=int(inputlist[1])
    eyear=int(inputlist[2])
    return grdc.grdc_dis(grdc_id, syear=syear, eyear=eyear, smon=1, emon=12, sday=1, eday=31)
#==================================
# Calculate relative KGE_components
def calc_KGE_components(inputlist):
    # line,i  = inputlist
    # line    = re.split(" ",line)
    # line    = list(filter(None, line))
    # num     = "%07d"%(int(line[0]))
    # ix      = int(line[4]) - 1
    # iy      = int(line[5]) - 1
    # KGE_components
    asm_ = inputlist[0]
    opn_ = inputlist[1]
    org_ = inputlist[2]
    KGEasm,CCasm,BRasm,RVasm=KGE_components(asm_,org_)
    KGEopn,CCopn,BRopn,RVopn=KGE_components(opn_,org_)
    rKGE=(KGEasm-KGEopn)/(1.0-KGEopn+1.0e-20)
    rCC=(CCasm-CCopn)/(1.0-CCopn+1.0e-20)
    rBR=1.0-(BRasm/BRopn+1.0e-20)
    rRV=(RVasm-RVopn)/(1.0-RVopn+1.0e-20)
    # print ("asm --->",asm_[1:10])
    # print ("org --->",org_[1:10])
    # print (asm_.shape, opn_.shape, org_.shape)
    # print (KGEasm,CCasm,BRasm,RVasm)
    # print (rKGE, rCC, rBR, rRV)
    return [rKGE, rCC, rBR, rRV, KGEasm, CCasm, BRasm, RVasm, KGEopn, CCopn, BRopn, RVopn]
    # if rivwth[iy, ix] > 50.0:
    #     return [rKGE, rCC, rBR, rRV]
    # else:
    #     return None
#==================================
# Calculate normalized root mean square error
def calc_NRMSE(inputlist):
    asm_ = inputlist[0]
    opn_ = inputlist[1]
    org_ = inputlist[2]
    NRMSEasm = NRMSE(asm_,org_)
    NRMSEopn = NRMSE(opn_,org_)
    return [NRMSEasm, NRMSEopn]
#==================================
# Calculate Nash-Sutcliffe efficiency
def calc_NSE(inputlist):
    asm_ = inputlist[0]
    opn_ = inputlist[1]
    org_ = inputlist[2]
    NSEasm = NSE(asm_,org_)
    NSEopn = NSE(opn_,org_)
    return [NSEasm, NSEopn]
#==============================================================================
####  data frame related functions ####
#==============================================================================
def label_dam_type(row):
    if row["DOR_PC"] < 500:
        return "Run-of-the-river"
    else:
        return "Storage-based"
#==============================================================================
# main
#==============================================================================
argv = sys.argv
print (len(argv))
argc = len(argv)
syear = int(argv[1])
eyear = int(argv[2])
grdclist = argv[3]
damlist = argv[4]
expname = argv[5]
CaMa_dir = argv[6]
mapname = argv[7]
outname = argv[8]
ncpus = int(argv[9])
#==============================================================================
lexp=1
upz=1e09 #m2
#-----
start_dt=datetime.date(syear,1,1)
end_dt=datetime.date(eyear,12,31)
size=60
start=0
last=(end_dt-start_dt).days + 1
nbdays=int(last)
#----
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
#--------------------------------------------
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
rivnum="./dat/rivnum_"+mapname+".bin"
rivnum=np.fromfile(rivnum,np.int32).reshape(ny,nx)
# rivermap=(rivnum<=100)*1.0
rivermap=((nextxy[0]>0)*(rivwth>50.0))*1.0
#==============================================================================
# read dam list used for analysis
with open(damlist,"r") as f:
    lines=f.readlines()

# damids
damids = [int(line.split()[0]) for line in lines[1::]]
xlist  = [int(line.split()[4]) - 1 for line in lines[1::]]
ylist  = [int(line.split()[5]) - 1 for line in lines[1::]]
#==============================================================================
# read ix, iy for each dam
# p = Pool(ncpus)
# results = p.map(read_xy, lines[1::])
# p.close()
# p.join()

# xlist = {damid: ix for damid, ix, _ in results}
# ylist = {damid: iy for damid, _, iy in results}

# # get dam location
# xlist = [ix - 1 for _, ix, _ in results]
# ylist = [iy - 1 for _, _, iy in results]

damloc=np.zeros((ny,nx),np.int32)
for i in range(len(xlist)):
    if xlist[i] >= nx or ylist[i] >= ny:
        continue
    # print (ylist[i],xlist[i])
    damloc[ylist[i],xlist[i]]=1


# print (xlist)

# filename = "GRanD_dams_v1_3.csv"  # Replace with the actual filename
# read GRanD data
df = pd.read_csv(grdclist, sep=';')
# print (df.head())
# print (df.columns)
dfnew = pd.DataFrame()
dfnew["GRDC_ID"] = df[df.columns[0]]
dfnew["RIVER"] = df[df.columns[1]]
dfnew["STATION"] = df[df.columns[2]]
dfnew["IX1"] = df[df.columns[3]]
dfnew["IY1"] = df[df.columns[4]]
dfnew["IX2"] = df[df.columns[5]]
dfnew["IY2"] = df[df.columns[6]]
dfnew["LON"] = lonlat[0,dfnew["IY1"].values-1,dfnew["IX1"].values-1]
dfnew["LAT"] = lonlat[1,dfnew["IY1"].values-1,dfnew["IX1"].values-1]

dfnew = dfnew.reset_index(drop=True)

# dfnew = dfnew.reset_index(drop=True)
# dfnew = dfnew[["GRAND_ID","DOR_PC","CAP_MCM","DIS_AVG_LS","MAIN_USE"]]
# dfnew["TYPE_DAM"]=dfnew.apply(lambda row: label_dam_type(row),axis=1)

# print (dfnew.head())

# read org data
# lx = dfnew["IX1"]
# ly = dfnew["GRDC_ID"]
# dfnew["DAM_IX"]=lx
# dfnew["DAM_IY"]=ly

# org[dnum,ndays]
# org=read_dis_true_multi(dfnew["IX1"],dfnew["IY1"], dfnew["IX2"],dfnew["IY2"],syear, eyear, indir)
# org=[]
# for grdc_id in dfnew["GRDC_ID"].values:
#     org.append(grdc.grdc_dis(grdc_id,syear=syear,eyear=eyear,smon=1,emon=12,sday=1,eday=31))
# res=map(read_grdc_data,[[grdcid,str(syear),str(eyear)] for grdcid in dfnew["GRDC_ID"].values])
# read org data
p   = Pool(ncpus)
res = p.map(read_grdc_data, [[grdcid,str(syear),str(eyear)] for grdcid in dfnew["GRDC_ID"].values])
p.close()
p.join()

# res=np.array(res)
# print (res.shape)
org=np.array(res)  #[i][0] for i in range(len(res))])
print ("org --->",org.shape)

# find GRDC stations with less than 1 year data
org_available = (org!= -9999.0).sum(axis=1) >= 365
org_available = org_available.astype(int)
# org_avaliable[org_avaliable == 0] = np.nan
dfnew["OBS AVAILABILITY"] = org_available
print ("obsevation avalilable -->", np.sum(org_available))
# org_avaliable=[]
# for i in range(len(org)):
#     if np.sum((org[i]!=-9999.0)*1.0) < 365*1:
#         print ("GRDC_ID: ",dfnew["GRDC_ID"].values[i]," has no data")
#         org_avaliable.append(0)
#     else:
#         org_avaliable.append(1)

# print (org)
# read sim data
p   = Pool(ncpus)
res = p.map(read_dis, [[expname,grdcid,str(syear),str(eyear)] for grdcid in dfnew["GRDC_ID"].values])
p.close()
p.join()
# map(read_dis, [[expname,grdcid,str(syear),str(eyear)] for grdcid in dfnew["GRDC_ID"].values])
#======================================================
asm = np.array([res[i][0] for i in range(len(res))])
opn = np.array([res[i][1] for i in range(len(res))])

print ("asm --->",asm.shape)
print ("opn --->",opn.shape)
# print (asm[0,0:10])
# print (opn[0,0:10])
# print (org[0,0:10])

# print ("KGE asm --->",KGE(asm[0,:],org[0,:]))
# print ("KGE opn --->",KGE(opn[0,:],org[0,:]))
#==============================================================================
# need add columns for rKGE, rCC, rBR, rRV
# calculate rKGE, rCC, rBR, rRV
#======================================================
p       = Pool(ncpus)
results = p.map(calc_KGE_components, [[asm_,opn_,org_] for asm_,opn_,org_ in zip(asm,opn,org)])
p.close()
p.join()
# metrics = [result for result in results]   

# # # print ("metrics --->",len(metrics))
# print ("results --->",results[0])
#======================================================
rKGE   = np.array([results[i][0] for i in range(len(dfnew["GRDC_ID"].values))])
rCC    = np.array([results[i][1] for i in range(len(dfnew["GRDC_ID"].values))])
rBR    = np.array([results[i][2] for i in range(len(dfnew["GRDC_ID"].values))])
rRV    = np.array([results[i][3] for i in range(len(dfnew["GRDC_ID"].values))])
KGEasm = np.array([results[i][4] for i in range(len(dfnew["GRDC_ID"].values))])
CCasm  = np.array([results[i][5] for i in range(len(dfnew["GRDC_ID"].values))])
BRasm  = np.array([results[i][6] for i in range(len(dfnew["GRDC_ID"].values))])
RVasm  = np.array([results[i][7] for i in range(len(dfnew["GRDC_ID"].values))])
KGEopn = np.array([results[i][8] for i in range(len(dfnew["GRDC_ID"].values))])
CCopn  = np.array([results[i][9] for i in range(len(dfnew["GRDC_ID"].values))])
BRopn  = np.array([results[i][10] for i in range(len(dfnew["GRDC_ID"].values))])
RVopn  = np.array([results[i][11] for i in range(len(dfnew["GRDC_ID"].values))])

print ("rKGE --->",rKGE.shape)

print (rKGE[0:100])

dfnew["rKGE"]=rKGE
dfnew["rCC"]=rCC
dfnew["rBR"]=rBR
dfnew["rRV"]=rRV

dfnew["KGEasm"]=KGEasm
dfnew["CCasm"]=CCasm
dfnew["BRasm"]=BRasm
dfnew["RVasm"]=RVasm

dfnew["KGEopn"]=KGEopn
dfnew["CCopn"]=CCopn
dfnew["BRopn"]=BRopn
dfnew["RVopn"]=RVopn

#======================================================
# calculate normalized RMSE
p       = Pool(ncpus)
results = p.map(calc_NRMSE, [[asm_,opn_,org_] for asm_,opn_,org_ in zip(asm,opn,org)])
p.close()
p.join()
NRMSEasm  = np.array([results[i][0] for i in range(len(dfnew["GRDC_ID"].values))])
NRMSEopn  = np.array([results[i][1] for i in range(len(dfnew["GRDC_ID"].values))])

dfnew["NRMSEasm"]=NRMSEasm
dfnew["NRMSEopn"]=NRMSEopn

#======================================================
# calculate normalized RMSE
p       = Pool(ncpus)
results = p.map(calc_NSE, [[asm_,opn_,org_] for asm_,opn_,org_ in zip(asm,opn,org)])
p.close()
p.join()
NSEasm  = np.array([results[i][0] for i in range(len(dfnew["GRDC_ID"].values))])
NSEopn  = np.array([results[i][1] for i in range(len(dfnew["GRDC_ID"].values))])

dfnew["NSEasm"]=NSEasm
dfnew["NSEopn"]=NSEopn
#======================================================
# Calculate metrics realted to WSE
# calculate mean RMSE, max RMSE, weigted RMSE
# calculate mean bias, max bias, weigted bias
# calculate mean CC, max CC, weigted CC
# calculate mean BR, max BR, weigted BR
# calculate mean RV, max RV, weigted RV
#======================================================
# dfnew["RIV_WTH"]=np.array([rivwth[dfnew["IX1"][df[grdc_id]]-1,dfnew["IY1"][df[grdc_id]]-1] for grdc_id in dfnew["GRDC_ID"].values])
# River width
dfnew["RIV_WTH"]=np.array(rivwth[dfnew["IY1"]-1,dfnew["IX1"]-1])

# Uparea
dfnew["UPAREA"]=np.array(uparea[dfnew["IY1"]-1,dfnew["IX1"]-1]*1e-6)

# River number depend on largest river (1,2,...,n)
rivnum="./dat/rivnum_"+mapname+".bin"
rivnum=np.fromfile(rivnum,np.int32).reshape(ny,nx)
dfnew["RIVNUM"]=np.array(rivnum[dfnew["IY1"]-1,dfnew["IX1"]-1])

# Satellite coverage
satcov="./dat/satellite_coverage.bin"
satcov=np.fromfile(satcov,np.float32).reshape(ny,nx)
satcov[satcov<=0]=0.0
dfnew["SAT_COV"]=np.array(satcov[dfnew["IY1"]-1,dfnew["IX1"]-1])

# elevation
dfnew["ELEV"]=np.array(elevtn[dfnew["IY1"]-1,dfnew["IX1"]-1])

dfnew = dfnew[dfnew["OBS AVAILABILITY"]==1]

# dfnew = dfnew.dropna()
dfnew = dfnew.reset_index(drop=True)

# print new dataframe
print (dfnew.head())

print ("dfnew statistics --->",dfnew.describe())

dfnew.to_csv(outname, sep=';', encoding='utf-8',index=False)