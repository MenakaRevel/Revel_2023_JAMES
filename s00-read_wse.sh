#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q F10
#PBS -l select=1:ncpus=10:mem=40gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N read_wse

# OMP Settings
NCPUS=10
export OMP_NUM_THREADS=$NCPUS

# cd $PBS_O_WORKDIR
cd "/work/a06/menaka/Revel_2023_JAMES"

# Input Dir
indir="/cluster/data8/menaka/HydroDA/out"

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
eyear=2020
emon=12
eday=31

N=`python src/calc_days.py $syear $smon $sday $eyear $emon $eday`
echo $N
# expname="NOM_WSE_E2O_HWEB_001"
# expname="NOM_WSE_E2O_HWEB_002"
# expname="NOM_WSE_E2O_HWEB_003"
# expname="NOM_WSE_E2O_HWEB_004"
# expname="NOM_WSE_E2O_HWEB_005"
# expname="NOM_WSE_E2O_HWEB_006"
# expname="ANO_WSE_E2O_HWEB_001"
# expname="ANO_WSE_E2O_HWEB_002"
# expname="ANO_WSE_E2O_HWEB_003"
# expname="ANO_WSE_E2O_HWEB_004"
expname="ANO_WSE_E2O_HWEB_005"
# expname="DIR_WSE_E2O_HWEB_001"
# expname="DIR_WSE_E2O_HWEB_002"

ens_mem=50

# simulatiion
obstype="simulation"
# obslist="/cluster/data6/menaka/HydroDA/dat/HydroWeb_alloc_amz_06min_QC0_simulation.txt"
obslist="/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_conus_06min_org.txt"

for expname in "NOM_WSE_ERA5_CGLS_062"; #"NOM_WSE_ERA5_CGLS_041" "NOM_WSE_ERA5_CGLS_062"; #"NOM_WSE_ERA5_CGLS_072"; #"ANO_WSE_ERA5_CGLS_001" "ANO_WSE_ERA5_CGLS_002"; #"NOM_WSE_ERA5_CGLS_042" "NOM_WSE_ERA5_CGLS_044"; #"ANO_WSE_ERA5_CGLS_004"; #"DIR_WSE_ERA5_CGLS_007"; # "ANO_WSE_ERA5_CGLS_008"; #"NOM_WSE_ERA5_CGLS_008"; #"ANO_WSE_ERA5_CGLS_008"; #"DIR_WSE_ERA5_CGLS_002" "DIR_WSE_ERA5_CGLS_003" "DIR_WSE_ERA5_CGLS_004" "NOM_WSE_ERA5_CGLS_001" "NOM_WSE_ERA5_CGLS_002"; #"DIR_WSE_ERA5_CGLS_001"; #
do
    mkdir -p $outdir/$expname/wse

    echo ./src/read_WSE $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir $obslist
    time ./src/read_WSE $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir $obslist
    ## for parallel computation using multiple CPUs 
    NUM=`ps -U $USER | grep ./src/read_WSE | wc -l | awk '{print $1}'`
    echo $NUM
    while [ $NUM -gt $NCPUS ];
    do
        sleep 1
        NUM=`ps -U $USER | grep ./src/read_WSE | wc -l | awk '{print $1}'`
    done
done

wait


# # validation
# obstype="validation"
# obslist="/cluster/data6/menaka/HydroDA/dat/HydroWeb_alloc_amz_06min_QC0_validation.txt"

# mkdir -p $outdir/$expname/wse.$obstype

# echo ./src/read_WSE $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir $obstype $obslist
# time ./src/read_WSE $expname $mapname $syear $smon $sday $eyear $emon $eday $ens_mem $N $CaMa_dir $indir $outdir $obstype $obslist