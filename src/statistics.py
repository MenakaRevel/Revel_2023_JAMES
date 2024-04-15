#!/opt/local/bin/python
# -*- coding: utf-8 -*-
'''
Statistics for streamflow caclculations
'''
import numpy as np
from numpy import ma
import datetime
import calendar
import warnings;warnings.filterwarnings('ignore')

#========================================
#====  functions calculating stats  ====
#========================================
def filter_nan(s,o):
    """
    this functions removed the data  from simulated and observed data
    where ever the observed data contains nan
    """
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]

    return data[:,0],data[:,1]
#====================================================================
def NSE(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        NSE: Nash Sutcliffe efficient coefficient
    """
    s,o = filter_nan(s,o)
    o=ma.masked_where(o<=0.0,o).filled(0.0)
    s=ma.masked_where(o<=0.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s) 
    return 1 - sum((s-o)**2)/(sum((o-np.mean(o))**2)+1e-20)
#====================================================================
def correlation(s,o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    s,o = filter_nan(s,o)
    o=ma.masked_where(o<=0.0,o).filled(0.0)
    s=ma.masked_where(o<=0.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    if s.size == 0:
        corr = 0.0 #np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
        
    return corr
#========================================
def KGE(s,o):
    """
	Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0,1]
    return 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)
#========================================
def KGE2009(s,o):
    """
	Kling Gupta Efficiency (Kling et al., 2009)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    B = np.mean(s) / np.mean(o)
    y = np.std(s) / np.std(o)
    r = np.corrcoef(o, s)[0,1]
    return 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)
#========================================
def KGED2012(s,o):
    """
	Kling Gupta Efficiency Deviation (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0,1]
    return 1 - np.sqrt((r - 1) ** 2 + (y - 1) ** 2)
#========================================
def KGED2009(s,o):
    """
	Kling Gupta Efficiency Deviation (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    B = np.mean(s) / np.mean(o)
    y = np.std(s) / np.std(o)
    r = np.corrcoef(o, s)[0,1]
    return 1 - np.sqrt((r - 1) ** 2 + (y - 1) ** 2)
#========================================
def KGE_components(s,o):
    """
	Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    BR = np.mean(s) / np.mean(o)
    RV = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    CC = np.corrcoef(o, s)[0,1]
    val=1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
    return val, CC, BR, RV
#========================================
def KGE_components_2009(s,o):
    """
	Kling Gupta Efficiency (Kling et al., 2009)
	input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    BR = np.mean(s) / np.mean(o)
    RV = np.std(s) / np.std(o)
    CC = np.corrcoef(o, s)[0,1]
    val=1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
    return val, CC, BR, RV
#==========================================================
def RMSE(s,o):
    """
    Root Mean Square Error
    input:
        s: simulated
        o: observed
    output:
        RMSE: Root Mean Square Error
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    # return np.sqrt(np.mean((s-o)**2))
    return np.sqrt(np.ma.mean(np.ma.masked_where(o<=0.0,(s-o)**2)))
#==========================================================
def NRMSE(s,o):
    """
    Normalized Root Mean Square Error
    input:
        s: simulated
        o: observed
    output:
        NRMSE: Normalized Root Mean Square Error
    """
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    s,o = filter_nan(s,o)
    # return np.sqrt(np.mean((s-o)**2))
    return np.sqrt(np.ma.mean(np.ma.masked_where(o<=0.0,(s-o)**2)))/np.mean(o)
#==========================================================
def pBIAS(s,o):
    """
    Percentage Bias
    input:
        s: simulated
        o: observed
    output:
        pBias: Percentage Bias
    """
    s,o = filter_nan(s,o)
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    return abs((np.mean(s)-np.mean(o))/np.mean(o))
#==========================================================
def BIAS(s,o):
    """
    Bias
    input:
        s: simulated
        o: observed
    output:
        Bias: Bias
    """
    s,o = filter_nan(s,o)
    o=ma.masked_where(o==-9999.0,o).filled(0.0)
    s=ma.masked_where(o==-9999.0,s).filled(0.0)
    o=np.compress(o>0.0,o)
    s=np.compress(o>0.0,s)
    # o=np.compress(o==-9999.0,o)
    # s=np.compress(o==-9999.0,s)
    return abs((np.mean(s)-np.mean(o)))
#====================================================================
def Amplitude(s,o,syear=2009,eyear=2014):
    """
    Amplitude 
    input:
        s: simulated
        o: observed
    output:
        Amplitude: Amplitudes of simulated and observed
    """ 
    s,o = filter_nan(s,o)
    # o=ma.masked_where(o==-9999.0,o).filled(0.0)
    # s=ma.masked_where(o==-9999.0,s).filled(0.0)
    # o=np.compress(o>0.0,o)
    # s=np.compress(o>0.0,s)
    amps=[]
    ampo=[]
    for year in np.arange(syear,eyear+1):
        date1 = datetime.date(year, 1, 1) # month and day are 1-base
        date2 = datetime.date(year, 12, 31)
        days  = (date2-date1).days + 1
        if year == syear:
            st_dt = 0
            ed_dt = days
        elif year == eyear:
            st_dt = ed_dt
            ed_dt = -1
        else:
            st_dt = ed_dt
            ed_dt = ed_dt + days
        # print st_dt, ed_dt
        maxloc = np.argmax(s[st_dt:ed_dt])
        minloc = np.argmin(s[st_dt:ed_dt])
        smax   = np.amax(s[st_dt:ed_dt])
        smin   = np.amin(s[st_dt:ed_dt])
        maxloc1=max(maxloc-15,0)
        maxloc2=min(maxloc+15,len(s)-1)
        minloc1=max(minloc-15,0)
        minloc2=min(minloc+15,len(s)-1)
        maxarry=ma.masked_equal(o[st_dt:ed_dt],-9999.0).filled(-9999.0)
        minarry=ma.masked_equal(o[st_dt:ed_dt],-9999.0).filled(9999.0)
        omax   = np.amax(maxarry[maxloc1:maxloc2])
        omin   = np.amin(minarry[minloc1:minloc2])
        if omax == -9999.0 or omin == 9999.0:
            continue
        amps.append(smax-smin)
        ampo.append(omax-omin)
    return np.mean(np.array(amps)),np.mean(np.array(ampo))
#====================================================================
def rAmplitude(s,o,syear=2009,eyear=2014):
    """
    Relative Amplitude Difference
    input:
        s: simulated
        o: observed
    output:
        rAmplitude: Realtive Amplitude Difference
    """ 
    s,o = filter_nan(s,o)
    # o=ma.masked_where(o==-9999.0,o).filled(0.0)
    # s=ma.masked_where(o==-9999.0,s).filled(0.0)
    # o=np.compress(o>0.0,o)
    # s=np.compress(o>0.0,s)
    alti_diff=[]
    for year in np.arange(syear,eyear+1):
        date1 = datetime.date(year, 1, 1) # month and day are 1-base
        date2 = datetime.date(year, 12, 31)
        days  = (date2-date1).days + 1
        if year == syear:
            st_dt = 0
            ed_dt = days
        elif year == eyear:
            st_dt = ed_dt
            ed_dt = -1
        else:
            st_dt = ed_dt
            ed_dt = ed_dt + days
        # print st_dt, ed_dt
        maxloc = np.argmax(s[st_dt:ed_dt])
        minloc = np.argmin(s[st_dt:ed_dt])
        smax   = np.amax(s[st_dt:ed_dt])
        smin   = np.amin(s[st_dt:ed_dt])
        maxloc1=max(maxloc-15,0)
        maxloc2=min(maxloc+15,len(s)-1)
        minloc1=max(minloc-15,0)
        minloc2=min(minloc+15,len(s)-1)
        maxarry=ma.masked_equal(o[st_dt:ed_dt],-9999.0).filled(-9999.0)
        minarry=ma.masked_equal(o[st_dt:ed_dt],-9999.0).filled(9999.0)
        omax   = np.amax(maxarry[maxloc1:maxloc2])
        omin   = np.amin(minarry[minloc1:minloc2])
        if omax == -9999.0 or omin == 9999.0:
            continue
        alti_diff.append((smax-smin)-(omax-omin))
    return np.mean(alti_diff)
#====================================================================
def dpeak_timing(s,o,syear=2001,eyear=2010):
    """
    Mean Peak Discharge Timing Difference
    input:
        s: simulated
        o: observed
    output:
        dpeak_timing: Mean Peak Discharge Timing Difference
    """ 
    s,o = filter_nan(s,o)
    peak_diff=[]
    for year in np.arange(syear,eyear+1):
        date1 = datetime.date(year, 1, 1) # month and day are 1-base
        date2 = datetime.date(year, 12, 31)
        days  = (date2-date1).days + 1
        if year == syear:
            st_dt = 0
            ed_dt = days
        elif year == eyear:
            st_dt = ed_dt
            ed_dt = -1
        else:
            st_dt = ed_dt
            ed_dt = ed_dt + days
        # print st_dt, ed_dt
        if sum(o[st_dt:ed_dt]) < 0.0:
            continue
        maxloco = np.argmax(o[st_dt:ed_dt])
        maxloc1=max(maxloco-15,0)
        maxloc2=min(maxloco+15,len(s)-1)
        maxlocs=maxloc1 + np.argmax(s[st_dt:ed_dt][maxloc1:maxloc2+1])
        # print (maxloco, maxlocs)
        peak_diff.append(float(maxlocs)-float(maxloco))
    return np.mean(np.abs(peak_diff))
#====================================================================
def peak_timing(s,o,syear=2001,eyear=2010):
    """
    Mean Peak Discharge Timing Difference
    input:
        s: simulated
        o: observed
    output:
        dpeak_timing: Mean Peak Discharge Timing Difference
    """ 
    s,o = filter_nan(s,o)
    lmaxlocs=[]
    lmaxloco=[]
    for year in np.arange(syear,eyear+1):
        date1 = datetime.date(year, 1, 1) # month and day are 1-base
        date2 = datetime.date(year, 12, 31)
        days  = (date2-date1).days + 1
        if year == syear:
            st_dt = 0
            ed_dt = days
        elif year == eyear:
            st_dt = ed_dt
            ed_dt = -1
        else:
            st_dt = ed_dt
            ed_dt = ed_dt + days
        # print st_dt, ed_dt
        if sum(o[st_dt:ed_dt]) < 0.0:
            continue
        maxloco = np.argmax(o[st_dt:ed_dt])
        maxloc1=max(maxloco-15,0)
        maxloc2=min(maxloco+15,len(s)-1)
        maxlocs=maxloc1 + np.argmax(s[st_dt:ed_dt][maxloc1:maxloc2+1])
        # print (maxloco, maxlocs)
        lmaxloco.append(float(maxloco))
        lmaxlocs.append(float(maxlocs))
    return np.mean(np.array(lmaxlocs)), np.mean(np.array(lmaxloco))
#====================================================================