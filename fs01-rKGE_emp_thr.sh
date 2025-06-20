#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q E10
#PBS -l select=1:ncpus=10:mem=10gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N Figure07

#source ~/.bashrc
# import virtual environment
# source ~/.bashrc
# source ~/.bash_conda

# source activate py38

# which python

# OMP Settings
NCPUS=10
export OMP_NUM_THREADS=$NCPUS

# got to working dirctory
# cd $PBS_O_WORKDIR
# cd "/cluster/data7/menaka/Reveletal2022"
cd "/work/a06/menaka/Revel_2023_JAMES"

mkdir -p figures
mkdir -p data

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

# figname="fig07-NSE_boxplot"
# figname="fig-KGE_boxplot"
# figname="fig-KGEAI_boxplot"
# figname="fig07-rISS_boxplot"
# figname="fig-DCORR_boxplot"
figname="figs1-rKGE_boxplot_threshold_20230804"

#*** 0. experiment list
EXLIST="./Figs1-experiment_list.nam"
rm -r $EXLIST
cat >> ${EXLIST} << EOF
NOM_All_Emp_020: NOM_WSE_ERA5_CGLS_041
NOM_All_Emp_040: NOM_WSE_ERA5_CGLS_042
NOM_All_Emp_060: NOM_WSE_ERA5_CGLS_043
NOM_All_Emp_080: NOM_WSE_ERA5_CGLS_044
EOF

# NOM_All_Emp_060Dam: NOM_WSE_ERA5_CGLS_002

python src/localization_sensitivity.py $syear $eyear $CaMa_dir $mapname $EXLIST $figname $NCPUS &

wait

rm -r $EXLIST

# conda deactivate