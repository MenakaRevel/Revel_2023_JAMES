#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q F40
#PBS -l select=1:ncpus=40:mem=100gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N read_wse

# OMP Settings
NCPUS=40
export OMP_NUM_THREADS=$NCPUS

# cd $PBS_O_WORKDIR
cd "/work/a06/menaka/Revel_2023_JAMES"

# # Input Dir
# indir="/cluster/data7/menaka/HydroDA/out"

# # out dir
# outdir="/work/a06/menaka/Revel_2023_JAMES/txt"

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

# expname="NOM_WSE_ERA5_CGLS_062"
expname="NOM_WSE_ERA5_CGLS_044"
# expname="ANO_WSE_ERA5_CGLS_008"
# expname="NOM_WSE_ERA5_CGLS_054"

mkdir -p ./out/$expname

echo "python src/hydrograph_pdf.py $syear $eyear $expname $mapname $CaMa_dir $NCPUS"
python src/hydrograph_pdf.py $syear $eyear $expname $mapname $CaMa_dir $NCPUS

wait