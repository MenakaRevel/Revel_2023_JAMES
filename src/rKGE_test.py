#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
rKGE - relative KGE values
'''
# '''
# # delta r cumulative distribution
# # NSEAI boxplot 
# # relative sharpness boxplot
# # delta reliability boxplot
# # rISS boxplot
# '''
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.cbook import boxplot_stats
# from matplotlib.colors import LogNorm,Normalize,ListedColormap
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
#==============================================================================
exlist="./experiment_list.nam" 
with open(exlist,"r") as exf:
    linesexf=exf.readlines()
experiments=[]
labels=[]
for lineexf in linesexf:
    lineexf = re.split(":",lineexf)
    lineexf = list(filter(None, lineexf))
    labels.append(lineexf[0])
    experiments.append(lineexf[1].strip())
#==============================================================================
upa_thr=1e2
riv_thr=100
df11 = pd.DataFrame()
df22 = pd.DataFrame()
for exp,label in zip(experiments,labels):
    # print (exp)
    df1="./out/"+exp+"/datafile.csv"
    # print (df1)
    df1=pd.read_csv(df1, sep=';')
    df2="./out/"+exp+"/datafile_2016-2020.csv"
    # print (df2)
    df2=pd.read_csv(df2, sep=';')
    df11["rKGE"]=df1["rKGE"].loc[(df1["SAT_COV"]==1.0) & (df1["UPAREA"]>=upa_thr) & (df1["RIVNUM"]<=riv_thr)]
    df22["rKGE"]=df2["rKGE"].loc[(df2["SAT_COV"]==1.0) & (df2["UPAREA"]>=upa_thr) & (df2["RIVNUM"]<=riv_thr)]
    print (exp, (sum((df11["rKGE"].values>0.0)*1.0)/len(df11["rKGE"]))*100.0,
    (sum((df22["rKGE"].values>0.0)*1.0)/len(df22["rKGE"]))*100.0)