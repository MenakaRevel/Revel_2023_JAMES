#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q F10
#PBS -l select=1:ncpus=10:mem=100gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N multi_read_dis

# OMP Settings
NCPUS=10
export OMP_NUM_THREADS=$NCPUS

USER=`whoami`

# cd $PBS_O_WORKDIR
cd "/work/a06/menaka/Revel_2023_JAMES"

# Input Dir
# indir="/cluster/data7/menaka/HydroDA/out"
indir="/cluster/data8/menaka/HydroDA/out" # for sensitivty experiments NOM 41-44

# out dir
outdir="/work/a06/menaka/Revel_2023_JAMES/txt"


#CaMA-Flood directory
# CaMa_dir="/cluster/data6/menaka/CaMa-Flood_v396a_20200514"
CaMa_dir="/cluster/data6/menaka/CaMa-Flood_v4"

# Map
mapname="conus_06min"

# syear, smon, sday
syear=2016
smon=1
sday=1

# eyear, emon, eday
eyear=2019
emon=12
eday=31

N=`python src/calc_days.py $syear $smon $sday $eyear $emon $eday`

ens_mem=50

for expname in "NOM_WSE_ERA5_CGLS_072"; #"NOM_WSE_ERA5_CGLS_051" "NOM_WSE_ERA5_CGLS_052" "NOM_WSE_ERA5_CGLS_053"; #"NOM_WSE_ERA5_CGLS_044"; #"ANO_WSE_ERA5_CGLS_004"; #"DIR_WSE_ERA5_CGLS_007"; # "ANO_WSE_ERA5_CGLS_008"; #"NOM_WSE_ERA5_CGLS_008"; #"ANO_WSE_ERA5_CGLS_008"; #"DIR_WSE_ERA5_CGLS_002" "DIR_WSE_ERA5_CGLS_003" "DIR_WSE_ERA5_CGLS_004" "NOM_WSE_ERA5_CGLS_001" "NOM_WSE_ERA5_CGLS_002"; #"DIR_WSE_ERA5_CGLS_001"; #
do
    mkdir -p $outdir/$expname/outflow
    ##=========================================================
    echo ./src/read_discharge $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir
    time ./src/read_discharge $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir &
    ## for parallel computation using multiple CPUs 
    NUM=`ps -U $USER | grep ./src/read_discharge | wc -l | awk '{print $1}'`
    echo $NUM
    while [ $NUM -gt $NCPUS ];
    do
        sleep 1
        NUM=`ps -U $USER | grep ./src/read_discharge | wc -l | awk '{print $1}'`
    done
done

wait