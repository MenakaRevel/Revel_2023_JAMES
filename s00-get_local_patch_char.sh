#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q E20
#PBS -l select=1:ncpus=20:mem=100gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N get_local_patch_char

#source ~/.bashrc
# import virtual environment
# source ~/.bashrc
# source ~/.bash_conda

# source activate pydef

which python

# OMP Settings
NCPUS=20
export OMP_NUM_THREADS=$NCPUS

# cd $PBS_O_WORKDIR
cd "/work/a06/menaka/Revel_2023_JAMES"

#CaMa-Flood directory
CaMa_dir="/cluster/data6/menaka/CaMa-Flood_v4"
# CaMa_dir="/cluster/data6/menaka/CaMa-Flood_v410"

# Map
# mapname="glb_15min"
mapname="conus_06min"

# syear, smon, sday
syear=2016
smon=1
sday=1

# eyear, emon, eday
eyear=2020
emon=12
eday=31

# GRDC list
grdclist="$CaMa_dir/map/$mapname/grdc_loc.txt"

# Dam list
vslist="/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_${mapname}_org.txt"

patchtype="conus_06min_ERA5_60"
patchdir="/cluster/data6/menaka/Empirical_LocalPatch/local_patch/${patchtype}"

outdir="./local_patch_char"  #_$syear-$eyear
mkdir -p $outdir/$patchtype

python src/local_patch_ch.py $syear $eyear $grdclist $vslist $patchdir $CaMa_dir $mapname $outdir $NCPUS