#!/opt/local/bin/python
# -*- coding: utf-8 -*-
"""
Create list of upstream dams with distance
------------------------------------------
grdc_id | dam | distance |
------------------------------------------
"""
import numpy as np
import sys
import datetime
import re
import pandas as pd
import sys
#=====================================
damlist=sys.argv[1]
grdclist=sys.argv[2]
CaMa_dir = argv[3]
mapname = argv[4]
#=====================================
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
#=====================================
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
#=====================================
# read damloc file 
df_dam=pd.read_csv(damlist,sep='\s+')


# read grdc file 
df_grdc=pd.read_csv(grdclist,sep=';')
grdcgrid = np.zeros([ny, nx], np.int32)
# Assigning "id" values to corresponding locations in the array
for index, row in df_grdc.iterrows():
    array[row['ix1'], row['iy1']] = int(row['id'])

# for dam in df_dam.index:
    # find the downstream 
