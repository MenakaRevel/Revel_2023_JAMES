# experiment names
## Years to use for this analysis 
Depend on the availability of the satellite observation 2017.1.1-2019.12.31

## Direct DA
1. [done] DIR_WSE_ERA5_CGLS_001 - Use all observation, Empirical local patch
2. [done] DIR_WSE_ERA5_CGLS_002 - Use observation thinning, Empirical local patch
3. [done] DIR_WSE_ERA5_CGLS_003 - Use observation thinning, Dam constrained empirical local patch
4. [done] DIR_WSE_ERA5_CGLS_004 - Use all observation, Dam constrained empirical local patch
5. DIR_WSE_ERA5_CGLS_005 - Use all observation, Distance-based local patch
6. DIR_WSE_ERA5_CGLS_006 - Use observation thinning, Distance-based local patch
7. [done] DIR_WSE_ERA5_CGLS_007 - Use observation thinning, Dam constrained Distance-based local patch
8. DIR_WSE_ERA5_CGLS_008 - Use all observation, Dam constrained Distance-based local patch

## Normalized DA
1. [done] NOM_WSE_ERA5_CGLS_001 - Use all observation, Empirical local patch
2. [done] NOM_WSE_ERA5_CGLS_002 - Use all observation, Dam constrained empirical local patch
3. [done] NOM_WSE_ERA5_CGLS_003 - Use observation thinning, Empirical local patch
4. NOM_WSE_ERA5_CGLS_004 - Use observation thinning, Dam constrained empirical local patch
5. NOM_WSE_ERA5_CGLS_005 - Use all observation, Distance-based local patch
6. NOM_WSE_ERA5_CGLS_006 - Use observation thinning, Distance-based local patch
7. NOM_WSE_ERA5_CGLS_007 - Use observation thinning, Dam constrained Distance-based local patch
8. [done] NOM_WSE_ERA5_CGLS_008 - Use all observation, Dam constrained Distance-based 

## Anomaly DA
1. [done] ANO_WSE_ERA5_CGLS_001 - Use all observation, Empirical local patch
2. [done] ANO_WSE_ERA5_CGLS_002 - Use all observation, Dam constrained empirical local patch
3. ANO_WSE_ERA5_CGLS_003 - Use observation thinning, Empirical local patch
4. ANO_WSE_ERA5_CGLS_004 - Use observation thinning, Dam constrained empirical local patch
5. ANO_WSE_ERA5_CGLS_005 - Use all observation, Distance-based local patch
6. ANO_WSE_ERA5_CGLS_006 - Use observation thinning, Distance-based local patch
7. ANO_WSE_ERA5_CGLS_007 - Use observation thinning, Dam constrained Distance-based local patch
8. [done] ANO_WSE_ERA5_CGLS_008 - Use all observation, Dam constrained Distance-based


### Best for each DA method
DIR - DIR_ThnEmpDam    52.083333    0.000
ANO - ANO_AllEmp       63.541667    0.017
NOM - NOM_AllEmpDam    57.291667    0.002

Exp            | % Positive | Median
DIR_AllEmp       36.458333    -0.024332
DIR_ThnEmp       50.000000    1.376911e-07  
DIR_ThnEmpDam    52.083333    6.217769e-07
DIR_AllEmpDam    41.666667    -0.008299
ANO_AllEmp       63.541667    0.017212
ANO_AllEmpDam    63.541667    0.007820
ANO_AllDstDam    56.250000    0.004152
NOM_AllEmp       53.125000    0.004269
NOM_AllEmpDam    57.291667    0.002354
NOM_ThnEmp       57.291667    8.043279e-07
NOM_AllDstDam    51.041667    0.000084


## Sensitivity experiments [for NOM all obs - without Dam] - Same DA settings
1. threshold empirical 0.2, 0.4 ,0.6, 0.8 --> DA experiment numbers 41-44 (0.6 is already done - NOM_WSE_ERA5_CGLS_001)
    [done] NOM_WSE_ERA5_CGLS_041 - All obs, Emplp threshold=0.2 := NOM_All_Emp_020
    [done] NOM_WSE_ERA5_CGLS_042 - All obs, Emplp threshold=0.4 := NOM_All_Emp_040
    [done] NOM_WSE_ERA5_CGLS_001 - All obs, Emplp threshold=0.6 := NOM_All_Emp_040
    [done] NOM_WSE_ERA5_CGLS_044 - All obs, Emplp threshold=0.8 := NOM_All_Emp_080

    [done] NOM_WSE_ERA5_CGLS_002 - All obs, Emplp threshold=0.6 Dam := NOM_All_Emp_060_Dam
    NOM_WSE_ERA5_CGLS_046 - All obs, Emplp threshold=0.8 Dam := NOM_All_Emp_080_Dam

       NOM_All_Emp_020  NOM_All_Emp_040  NOM_All_Emp_060  NOM_All_Emp_080
count        96.000000        96.000000        96.000000        96.000000
mean         -0.039929        -0.026172         0.006318        -0.002592
std           0.339827         0.316301         0.213752         0.236093
min          -1.756259        -1.649259        -0.760735        -1.289753
25%          -0.090682        -0.051054        -0.021385        -0.022164
50%          -0.004060        -0.004424         0.004269         0.002567
75%           0.087391         0.061109         0.060716         0.039503
max           0.548387         0.542267         0.519508         0.563579

NOM_All_Emp_020    -0.004060     47.916667
NOM_All_Emp_040    -0.004424     47.916667
NOM_All_Emp_060     0.004269     53.125000
NOM_All_Emp_080     0.002567     54.166667


	           median (UPAREA>=1.0e4)	%positive (UPAREA>=1.0e4)	median (UPAREA>=2.5e4) %positive (UPAREA>=2.5e4)
NOM_All_Emp_020	    -0.0041	         47.92	        -0.0041	              46.55
NOM_All_Emp_040	    -0.0044	         47.92	        -0.0086	              48.28
NOM_All_Emp_060	    -0.0015	         47.92	        -0.0038	              48.28
## NOM_All_Emp_080	     0.0026	         54.17	         0.0136	              56.89


2. distance distance-based 50, 100, 500, 1000km --> DA experiment numbers 51-54
    [done] NOM_WSE_ERA5_CGLS_051 - All obs, Dst  50KM := NOM_All_Dst_50KM
    [done] NOM_WSE_ERA5_CGLS_052 - All obs, Dst 100KM := NOM_All_Dst_100KM
    [done] NOM_WSE_ERA5_CGLS_053 - All obs, Dst 500KM := NOM_All_Dst_500KM
    [done] NOM_WSE_ERA5_CGLS_005 - All obs, Dst 1000KM:= NOM_All_Dst_1000KM

    NOM_WSE_ERA5_CGLS_055 - All obs, Dst 500KM Dam  := NOM_All_Dst_100KM_Dam
    NOM_WSE_ERA5_CGLS_056 - All obs, Dst 1000KM Dam := NOM_All_Dst_500KM_Dam

       NOM_All_Dst_50KM  NOM_All_Dst_100KM  NOM_All_Dst_500KM
count      9.600000e+01          96.000000          96.000000
mean      -2.257346e-07           0.011020          -0.028640
std        2.848014e-06           0.110506           0.284552
min       -1.334076e-05          -0.620686          -1.377462
25%       -6.259339e-07          -0.006121          -0.060551
50%       -9.190886e-08           0.001389          -0.003995
75%        6.287784e-07           0.022212           0.052835
max        6.419335e-06           0.297572           0.593160
NOM_All_Dst_50KM     46.875000
NOM_All_Dst_100KM    57.291667
NOM_All_Dst_500KM    48.958333

  median (UPAREA>=1.0e4)	%positive (UPAREA>=1.0e4)	median (UPAREA>=2.5e4) %positive (UPAREA>=2.5e4)
NOM_All_Dst_50KM	   -0.0000	  46.88	   -0.0000	46.55
### NOM_All_Dst_100KM	    0.0014	  57.29	    0.0025	60.34
NOM_All_Dst_500KM	   -0.0040	  48.96	   -0.0048	48.28
NOM_All_Dst_1000KM	   -0.0027	  47.92	   -0.0017	48.28

3. temporal empirical 1, 5, 10, 20yrs ---> DA experiment numbers 61-64
4. ensemble size ?? --> need more time and computational efficiency (may be 20, 50, 100 perturbations)

*** Ensemble simulations --> 20,  50, 100, 1000 (??)
[done] NOM_All_Emp_080_020: NOM_WSE_ERA5_CGLS_044 / 061
[done] NOM_All_Emp_080_050: NOM_WSE_ERA5_CGLS_062 - 50 ensembles (keep the openloop files spinup and assim_out/open)
NOM_All_Emp_080_100: NOM_WSE_ERA5_CGLS_063

NOM_All_Emp_060_050: NOM_WSE_ERA5_CGLS_072 - 50 ensembles (keep the openloop files spinup and assim_out/open)

**** ERA5 seems to biased in WSE according to Xudong (2023) --> VIC_BC can be a better option as Runoff.

## --> **** {Does the ensemble size is not large enough to have give a difference in DA improvement ??? }


--> use the data at: /home/yamadai/data/Runoff/VIC_BC_day
dimensions:
	time = UNLIMITED ; // (365 currently)
	bnds = 2 ;
	lon = 7200 ;
	lat = 3000 ;

Orientation: StoN

nc0=xr.open_dataset("RUNOFF_2019.nc", chunks={'time':1}) 

# check the assimilation results with VIC_BC
==> as ERA5 WSE has large bias than the VIC BC according to Xudong Benchmarking system.


# VIC BC Experiments
## 1. Sensitivity of ensemble size
NOM_All_Emp_080_020: NOM_WSE_VICBC_CGLS_011
NOM_All_Emp_080_050: NOM_WSE_VICBC_CGLS_012
NOM_All_Emp_080_100: NOM_WSE_VICBC_CGLS_013
NOM_All_Emp_080_1000: NOM_WSE_VICBC_CGLS_014