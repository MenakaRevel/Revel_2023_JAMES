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
def downstream(ix,iy,nx,ny,nextxy,nxtdst,grdcgrid):
    grdc_id=-9999.0
    ix = ix - 1
    iy = iy - 1
    iix = ix
    iiy = iy
    dist = 0.0
    while nextxy[0,iiy - 1,iix - 1] > 0:
        ixx=iix - 1
        iyy=iiy - 1
        iix = nextxy[0,iyy,ixx]
        iiy = nextxy[1,iyy,ixx]
        if (iix<0 or iiy<0):
            grdc_id=-9999.0
            break
        if (iix > nx or iiy > ny):
            break
        grdc_id=grdcgrid[iiy - 1, iix - 1]
        if grdc_id != -9999.0:
            break
        dist = dist + nxtdst[iiy - 1, iix - 1]*1e-3
    if grdc_id == -9999.0:
       dist = -9999.0 
    return grdc_id, dist
#=====================================
damlist=sys.argv[1]
grdclist=sys.argv[2]
CaMa_dir = sys.argv[3]
mapname = sys.argv[4]
outdir = sys.argv[5]
#=====================================
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
#=====================================
nextxy = CaMa_dir+"/map/"+mapname+"/nextxy.bin"
rivwth = CaMa_dir+"/map/"+mapname+"/rivwth_gwdlr.bin"
rivhgt = CaMa_dir+"/map/"+mapname+"/rivhgt.bin"
rivlen = CaMa_dir+"/map/"+mapname+"/rivlen.bin"
elevtn = CaMa_dir+"/map/"+mapname+"/elevtn.bin"
lonlat = CaMa_dir+"/map/"+mapname+"/lonlat.bin"
uparea = CaMa_dir+"/map/"+mapname+"/uparea.bin"
nxtdst = CaMa_dir+"/map/"+mapname+"/nxtdst.bin"
nextxy = np.fromfile(nextxy,np.int32).reshape(2,ny,nx)
rivwth = np.fromfile(rivwth,np.float32).reshape(ny,nx)
rivhgt = np.fromfile(rivhgt,np.float32).reshape(ny,nx)
rivlen = np.fromfile(rivlen,np.float32).reshape(ny,nx)
elevtn = np.fromfile(elevtn,np.float32).reshape(ny,nx)
lonlat = np.fromfile(lonlat,np.float32).reshape(2,ny,nx)
uparea = np.fromfile(uparea,np.float32).reshape(ny,nx)
nxtdst = np.fromfile(nxtdst,np.float32).reshape(ny,nx)
#===================
rivnum="./dat/rivnum_"+mapname+".bin"
rivnum=np.fromfile(rivnum,np.int32).reshape(ny,nx)
# rivermap=(rivnum<=100)*1.0
rivermap=((nextxy[0]>0)*(rivwth>50.0))*1.0
#=====================================
# read damloc file 
df_dam=pd.read_csv(damlist,sep='\s+')
print (df_dam.head())

# read grdc file 
df_grdc=pd.read_csv(grdclist,sep=';')
df_grdc.rename(columns=lambda x: x.strip(),inplace=True)
print (df_grdc.head())
print (df_grdc.columns)
grdcgrid = np.ones([ny, nx], np.int32)*-9999
# Assigning "id" values to corresponding locations in the array
for index, row in df_grdc.iterrows():
    grdcgrid[row['iy1']-1, row['ix1']-1] = int(row['ID'])

gids=[]
damids=[]
damnames=[]
dists=[]
caps=[]
for point in df_dam.index[0::]:
    # find the downstream
    ix=df_dam['DamIX'][point]
    iy=df_dam['DamIY'][point]
    if (ix > nx or iy > ny):
        continue
    gid, dist= downstream(ix,iy,nx,ny,nextxy,nxtdst,grdcgrid)
    if gid != -9999.0:
        print (int(gid), df_grdc[df_grdc['ID']==int(gid)]['Station'].values[0].strip(),df_dam['Dam_ID'][point], df_dam['DamName'][point], dist)
        gids.append(int(gid))
        damids.append(df_dam['Dam_ID'][point]) 
        damnames.append(df_dam['DamName'][point])
        dists.append(dist)
        caps.append(df_dam['Capacity'][point])

df_out=pd.DataFrame()

df_out['grdcID']=gids
df_out['DamID']=damids
df_out['DamName']=damnames
df_out['Distance']=dists
df_out['Capacity']=caps


# df_1=df_out.loc[df_out.groupby(['grdcID'])['Distance'].idmax()]
# print (df_1)

# df_2=df_1.loc[df_1.groupby(['grdcID'])['Capacity'].max()]
# print (df_2)

max_distance = df_out.groupby('grdcID')['Distance'].transform(max)
max_capacity = df_out.groupby('grdcID')['Capacity'].transform(max)
print (max_distance)
print (max_capacity)

df_1 = df_out.loc[(df_out['Distance'] == max_distance) & (df_out['Capacity'] == max_capacity)]
print (df_1)

df_1.to_csv(outdir+'/dam_grdc_list.txt',index=False)

print ("save "+outdir+'/dam_grdc_list.txt')