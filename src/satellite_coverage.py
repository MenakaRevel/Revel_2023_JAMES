#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
Find river extent with Virtual Stations exists
'''
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
from numpy import ma
import re
#====================================================================
def along_river(ix,iy):
    xlist=[]
    ylist=[]
    while (ix>=0): #!= -9 or ix !=-10):
        xlist.append(ix)
        ylist.append(iy)
        iix=ix
        iiy=iy
        ix=nextxy[0,iiy,iix]
        iy=nextxy[1,iiy,iix]
        if (ix==-9999):
            break
        ix=ix-1
        iy=iy-1
    return np.array(xlist), np.array(ylist)
#====================================================================
# argv=sys.argv
# syear=int(argv[1])
# eyear=int(argv[2])
# CaMa_dir=argv[3]
# mapname=argv[4]
# figname=argv[5]
# ncpus=int(sys.argv[6])
ens_mem=49
seaborn_map=True
# seaborn_map=False
#==================================
# DA_dir="/cluster/data6/menaka/HydroDA"
syear=2009
eyear=2014
CaMa_dir="/cluster/data6/menaka/CaMa-Flood_v4"
mapname="conus_06min"
# ens_mem=21
# lexp=7
# seaborn_map=False
#==================================
#===================
fname=CaMa_dir+"/map/"+mapname+"/params.txt"
with open(fname,"r") as f:
    lines=f.readlines()
#-------
nx     = int(filter(None, re.split(" ",lines[0]))[0])
ny     = int(filter(None, re.split(" ",lines[1]))[0])
gsize  = float(filter(None, re.split(" ",lines[3]))[0])
lon0   = float(filter(None, re.split(" ",lines[4]))[0])
lat0   = float(filter(None, re.split(" ",lines[7]))[0])
west   = float(filter(None, re.split(" ",lines[4]))[0])
east   = float(filter(None, re.split(" ",lines[5]))[0])
south  = float(filter(None, re.split(" ",lines[6]))[0])
north  = float(filter(None, re.split(" ",lines[7]))[0])
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
rivermap=(nextxy[0]>0)*1.0 #*(rivnum==1))
#===================
# obslist="/cluster/data6/menaka/HydroDA/dat/HydroWeb_alloc_"+mapname+"_QC0.txt"
obslist="/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_"+mapname+"_org.txt"
with open(obslist,"r") as f:
    lines=f.readlines()
streams=[]
vscover=np.ones([ny,nx],np.float32)*-9999.0
for line in lines[1::]:
    line    = re.split(" ",line)
    line    = list(filter(None, line))
    num     = line[0].strip()
    station = line[1].strip()
    ix      = int(line[4])
    iy      = int(line[5])
    # stream  = station[0:-7]
    # distto  = float(station[-4::])
    #-------------------------
    if rivermap[iy-1,ix-1] !=1.0:
        continue
    # #-------------------------
    # if stream in streams:
    #     continue
    # streams.append(stream)
    lx, ly = along_river(ix, iy)
    vscover[ly,lx]=1.0

# print streams
# pname=[]
# dista=[]
# xlist=[]
# ylist=[]
# for stream0 in streams:
#     # print stream0
#     dist0=0.0
#     for line in lines[1::]:
#         line    = re.split(" ",line)
#         line    = list(filter(None, line))
#         # print line
#         num     = line[0].strip()
#         station = line[1].strip()
#         lon     = float(line[2])
#         lat     = float(line[3])
#         ix      = int(line[4])
#         iy      = int(line[5])
#         ele     = float(line[6])
#         ele_dif = float(line[7])
#         EGM08   = float(line[8])
#         EGM96   = float(line[9])
#         sat     = line[10].strip()
#         stream  = station[0:-7]
#         dstnow  = float(station[-4::])
#         # print station, stream, dstnow
#         if stream==stream0:
#             # print stream, dstnow
#             if dstnow>=dist0:
#                 dist0=dstnow
#                 loc0 =station
#                 ix0  =ix-1
#                 iy0  =iy-1
#     # print loc0, dist0,ix0,iy0
#     pname.append(loc0)
#     dista.append(dist0)
#     xlist.append(ix0)
#     ylist.append(iy0)
# #=========================
# pnum=len(pname)
# vscover=np.ones([ny,nx],np.float32)*-9999.0
# for point in np.arange(pnum):
#     print pname[point], dista[point], xlist[point], ylist[point]
#     lx, ly = along_river(xlist[point], ylist[point])
#     vscover[ly,lx]=1.0

#==== save file ===
vscover.tofile("./dat/satellite_coverage.bin")