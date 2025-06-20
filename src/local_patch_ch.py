#!/opt/local/bin/python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import sys
import datetime
import re
import pandas as pd
from multiprocessing import Pool, RawArray
from statistics import *
# from read_CMF import read_discharge, read_discharge_multi
import read_grdc as grdc

#==============================================================================
# main
#==============================================================================
argv = sys.argv
print (len(argv))
argc = len(argv)
syear = int(argv[1])
eyear = int(argv[2])
grdclist = argv[3]
vslist = argv[4]
patchdir = argv[5]
CaMa_dir = argv[6]
mapname = argv[7]
outdir = argv[8]
ncpus = int(argv[9])
'''
#==============================================================================
# read grid info
df = pd.read_csv(grdclist, sep=';',skipinitialspace = True)
print (df.head())
print (df.columns)
print (df[df.columns[1]].values)

for river  in df[df.columns[1]].values:
    if river=="MISSISSIPPI":
        print (river)

dfgrdc = pd.DataFrame()
df=df[(df[df.columns[1]]=="MISSISSIPPI") | (df[df.columns[1]]=="MISSOURI")]

dfgrdc["GRDC_ID"] = df[df.columns[0]]#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["RIVER"] = df[df.columns[1]]#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["STATION"] = df[df.columns[2]]#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["IX1"] = df[df.columns[3]].astype('int')#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["IY1"] = df[df.columns[4]].astype('int')#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["IX2"] = df[df.columns[5]].astype('int')#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
dfgrdc["IY2"] = df[df.columns[6]].astype('int')#[(df.columns[1]=="MISSISSIPPI") & (df.columns[1]=="MISSOURI")]
# dfgrdc["LON"] = lonlat[0,dfgrdc["IY1"].values-1,dfgrdc["IX1"].values-1]
# dfgrdc["LAT"] = lonlat[1,dfgrdc["IY1"].values-1,dfgrdc["IX1"].values-1]

dfgrdc = dfgrdc.reset_index(drop=True)
print (dfgrdc.head())
#==============================================================================
# read vslist
dfvs = pd.read_csv(vslist, sep='\s+',skipinitialspace = True)
dfvs = dfvs.reset_index(drop=True)
dfvs['ix']=dfvs['ix'].astype('int')
dfvs['iy']=dfvs['iy'].astype('int')
print (dfvs.head())

# get vs to ix,iy

# patchtype="conus_06min_ERA5_60"
# read local patch
'''
'''
for i in range(len(dfgrdc)):
    ix0 = dfgrdc["IX1"][i]
    iy0 = dfgrdc["IY1"][i]
    print ("==========================")
    print (dfgrdc["RIVER"][i], dfgrdc["STATION"][i])
    # read local patch
    patchname = patchdir+"/patch%04d%04d.txt"%(ix0,iy0)
    dflp = pd.read_csv(patchname, sep='\s+',skipinitialspace = True, header=None, names=['IX', 'IY', 'weight'])

    # include vs inside local patch
    vs= np.zeros([len(dflp)],dtype=object)
    vs[:] = 'nan'
    for j in range(len(dflp)):
        ix = dflp['IX'][j]
        iy = dflp['IY'][j]
        # print (ix,iy) 
        # print ([(dfvs['ix']==ix) & (dfvs['iy']==iy)])
        # print (len(dfvs[(dfvs['ix']==ix) & (dfvs['iy']==iy)].values)>0)
        if len(dfvs[(dfvs['ix']==ix) & (dfvs['iy']==iy)].values)>0:
            # print (dfvs["station"][(dfvs['ix']==ix) & (dfvs['iy']==iy)].values)
            vs[j]=dfvs[(dfvs['ix']==ix) & (dfvs['iy']==iy)]["station"].values[0]
            # print (vs[j])
        # for k in range(len(dfvs)):
        #     ixx=dfvs['ix'][k]
        #     iyy=dfvs['iy'][k]
        #     if (ix==ixx) & (iy==iyy):
        #         print (ix,iy,ixx, iyy, dfvs["station"][k])
        # print (ix,iy, dfvs["station"][(dfvs['ix']==ix) & (dfvs['iy']==iy)])


    # #
    dflp["vs"] = vs
    print (dflp) #.head())
    dflp.to_csv(outdir+"/"+re.split("/",patchdir)[-1]+"/"+str(dfgrdc["GRDC_ID"][i])+".txt", sep='\t', encoding='utf-8',index=False)
'''
#====================================================================
grdc    = CaMa_dir+"/map/"+mapname+"/grdc_loc.txt"
df_grdc = pd.read_csv(grdc, sep=';')
df_grdc.rename(columns=lambda x: x.strip(), inplace=True)
#====================================================================
fcgls   = "/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_conus_06min_org.txt"
df_cgls = pd.read_csv(fcgls, sep='\s+')
df_cgls.rename(columns=lambda x: x.strip(), inplace=True)
#====================================================================
for point in df_grdc.index[0::]:
    print (df_grdc["ID"][point], df_grdc["Station"][point])
    ix0=df_grdc['ix1'][point]
    iy0=df_grdc['iy1'][point]
    patchname = patchdir+"/patch%04d%04d.txt"%(ix0,iy0)
    dflp = pd.read_csv(patchname, sep='\s+',skipinitialspace = True, header=None, names=['IX', 'IY', 'weight'])
    #=====================================
    # include vs inside local patch
    vs= np.zeros([len(dflp)],dtype=object)
    vs[:] = 'nan'
    metricVals=[]
    VS_count=0
    for j in range(len(dflp)):
        ix = dflp['IX'][j]
        iy = dflp['IY'][j]
        #==================================
        if len(df_cgls[(df_cgls['ix']==ix) & (df_cgls['iy']==iy)].values)>0:
            # print (dfvs["station"][(dfvs['ix']==ix) & (dfvs['iy']==iy)].values)
            # vs[j]=dfvs[(dfvs['ix']==ix) & (dfvs['iy']==iy)]["station"].values[0]
            # vs[j]=df_cgls.loc[(df_cgls['ix']==ix) & (df_cgls['iy']==iy),"ID"].values[0]
            vs[j]=df_cgls.loc[(df_cgls['ix']==ix) & (df_cgls['iy']==iy),"station"].values[0]
            print (ix,iy,df_cgls.loc[(df_cgls['ix']==ix) & (df_cgls['iy']==iy),['ix','iy']].values[0], df_cgls.loc[(df_cgls['ix']==ix) & (df_cgls['iy']==iy),"station"].values[0])
        # dflp["vs"] = vs
        # print (dflp) #.head())
        # dflp.to_csv(outdir+"/"+re.split("/",patchdir)[-1]+"/"+str(df_grdc["ID"][point])+".txt", sep='\t', encoding='utf-8',index=False)
    print ("="*20)