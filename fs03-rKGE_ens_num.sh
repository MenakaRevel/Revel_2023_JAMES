#! /bin/bash

### SET "mool PBS" @ IIS U-Tokyo
#PBS -q E10
#PBS -l select=1:ncpus=10:mem=10gb
#PBS -l place=scatter
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N FigureS02

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
eyear=2019
emon=12
eday=31

# figname="fig07-NSE_boxplot"
# figname="fig-KGE_boxplot"
# figname="fig-KGEAI_boxplot"
# figname="fig07-rISS_boxplot"
# figname="fig-DCORR_boxplot"
figname="figs3-rKGE_boxplot_ensemble_number"

#*** 0. experiment list
EXLIST="./Figs3-experiment_list.nam"
rm -r $EXLIST
cat >> ${EXLIST} << EOF
NOM_All_Emp_020: NOM_WSE_ERA5_CGLS_044
NOM_All_Emp_050_ERA5_60: NOM_WSE_ERA5_CGLS_072
NOM_All_Emp_050_VICBC: NOM_WSE_VICBC_CGLS_012
NOM_All_Emp_050_VICBC_60: NOM_WSE_VICBC_CGLS_022
EOF
# 
#NOM_All_Emp_050: NOM_WSE_ERA5_CGLS_062
python src/localization_sensitivity.py $syear $eyear $CaMa_dir $mapname $EXLIST $figname $NCPUS &

wait

rm -r $EXLIST

# conda deactivate