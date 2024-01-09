#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
functions to read discharge and wse
'''
import numpy as np
import pandas as pd
import datetime
import sys
import os
#====================================================================
# read discharge
#====================================================================
def read_dis(experiment,station,syear,eyear):
    '''
    read discharge 
    '''
    fname ="./txt/"+experiment+"/outflow/"+station+".txt"
    df_dis=pd.read_csv(fname,sep='\s+',header=None, names=['date','asm','opn'])
    df_dis.rename(columns=lambda x: x.strip(), inplace=True)
    df_dis['date']=pd.to_datetime(df_dis['date'],format="%Y%m%d")
    df_dis.set_index(df_dis['date'],inplace=True)
    df_dis.dropna(axis=0,inplace=True)
    syyyymmdd='%04d-1-1'%(syear)
    eyyyymmdd='%04d-12-31'%(eyear)
    return df_dis[syyyymmdd:eyyyymmdd]['asm'].values, df_dis[syyyymmdd:eyyyymmdd]['opn'].values
#====================================================================
def read_wse(experiment,station,syear,eyear):
    asm=[]
    opn=[]
    fname ="./txt/"+experiment+"/wse/"+station+".txt"
    df_wse=pd.read_csv(fname,sep='\s+',header=None, names=['date','asm','opn'])
    df_wse.rename(columns=lambda x: x.strip(), inplace=True)
    df_wse['date']=pd.to_datetime(df_wse['date'],format="%Y%m%d")
    df_wse.set_index(df_wse['date'],inplace=True)
    df_wse.dropna(axis=0,inplace=True)
    syyyymmdd='%04d-1-1'%(syear)
    eyyyymmdd='%04d-12-31'%(eyear)
    return df_wse[syyyymmdd:eyyyymmdd]['asm'].values, df_wse[syyyymmdd:eyyyymmdd]['opn'].values
#====================================================================
