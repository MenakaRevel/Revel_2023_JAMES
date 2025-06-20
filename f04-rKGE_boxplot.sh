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
figname="fig04-rKGE_boxplot_dist_ALL"

#*** 0. experiment list
EXLIST="./Fig04-experiment_list.nam"
rm -r $EXLIST
cat >> ${EXLIST} << EOF
DIR_All_Emp:     DIR_WSE_ERA5_CGLS_001
DIR_Thn_Emp:     DIR_WSE_ERA5_CGLS_002
DIR_Thn_Emp_Dam: DIR_WSE_ERA5_CGLS_003
DIR_Thn_Dst_Dam: DIR_WSE_ERA5_CGLS_007
ANO_All_Emp:     ANO_WSE_ERA5_CGLS_001
ANO_All_Emp_Dam: ANO_WSE_ERA5_CGLS_002
ANO_Thn_Emp_Dam: ANO_WSE_ERA5_CGLS_008
ANO_All_Dst_Dam: ANO_WSE_ERA5_CGLS_008
NOM_All_Emp:     NOM_WSE_ERA5_CGLS_001
NOM_All_Emp_Dam: NOM_WSE_ERA5_CGLS_002
NOM_Thn_Emp:     NOM_WSE_ERA5_CGLS_003
NOM_All_Dst_Dam: NOM_WSE_ERA5_CGLS_008
EOF

#DIR_AllEmpDam: DIR_WSE_ERA5_CGLS_004
# EXLIST="./Fig01-experiment_list.nam"
# rm -r $EXLIST
# cat >> ${EXLIST} << EOF
# NOM_AllObs: NOM_WSE_ERA5_CGLS_001
# NOM_AllDam: NOM_WSE_ERA5_CGLS_002
# NOM_ThnObs: NOM_WSE_ERA5_CGLS_003
# EOF

# NOM_ThnDam: NOM_WSE_ERA5_CGLS_004

# 
python src/relative_statistics_plot.py $syear $eyear $CaMa_dir $mapname $EXLIST $figname $NCPUS &

wait

rm -r $EXLIST

# conda deactivate