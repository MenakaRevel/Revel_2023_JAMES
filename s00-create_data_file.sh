#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q E20
#PBS -l select=1:ncpus=20:mem=100gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N create_datafile

#source ~/.bashrc
# import virtual environment
source ~/.bashrc
source ~/.bash_conda

source activate pydef

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
eyear=2019
emon=12
eday=31

# GRDC list
grdclist="$CaMa_dir/map/$mapname/grdc_loc.txt"

# Dam list
damlist="./dat/damloc_$mapname.txt"

# experiment name
# expname="DIR_WSE_ERA5_CGLS_001"

for expname in "DIR_WSE_ERA5_CGLS_001" "DIR_WSE_ERA5_CGLS_002" "DIR_WSE_ERA5_CGLS_003" "DIR_WSE_ERA5_CGLS_007" "NOM_WSE_ERA5_CGLS_001" "NOM_WSE_ERA5_CGLS_002" "NOM_WSE_ERA5_CGLS_003" "ANO_WSE_ERA5_CGLS_001" "ANO_WSE_ERA5_CGLS_002" "ANO_WSE_ERA5_CGLS_004" "ANO_WSE_ERA5_CGLS_008"; #"NOM_WSE_ERA5_CGLS_072"; #"NOM_WSE_ERA5_CGLS_062"; #"ANO_WSE_ERA5_CGLS_008"; #"NOM_WSE_ERA5_CGLS_054" "NOM_WSE_ERA5_CGLS_051" "NOM_WSE_ERA5_CGLS_052" "NOM_WSE_ERA5_CGLS_053" "NOM_WSE_ERA5_CGLS_041" "NOM_WSE_ERA5_CGLS_042" "NOM_WSE_ERA5_CGLS_044" "NOM_WSE_ERA5_CGLS_001"; #"NOM_WSE_ERA5_CGLS_051"; # #"ANO_WSE_ERA5_CGLS_004"; #"DIR_WSE_ERA5_CGLS_007"; #"ANO_WSE_ERA5_CGLS_008" "NOM_WSE_ERA5_CGLS_008" "DIR_WSE_ERA5_CGLS_001" "DIR_WSE_ERA5_CGLS_002" "DIR_WSE_ERA5_CGLS_003" "DIR_WSE_ERA5_CGLS_004" "NOM_WSE_ERA5_CGLS_001" "NOM_WSE_ERA5_CGLS_002" "NOM_WSE_ERA5_CGLS_003" "ANO_WSE_ERA5_CGLS_001" "ANO_WSE_ERA5_CGLS_002";
do
    # output file name
    outname="./out/$expname/datafile.csv"  #_$syear-$eyear

    mkdir -p ./out/$expname
    # 
    echo "python src/create_data_file.py $syear $eyear $grdclist $damlist $expname $CaMa_dir $mapname $outname $NCPUS" 

    python src/create_data_file.py $syear $eyear $grdclist $damlist $expname $CaMa_dir $mapname $outname $NCPUS
done

wait

conda deactivate