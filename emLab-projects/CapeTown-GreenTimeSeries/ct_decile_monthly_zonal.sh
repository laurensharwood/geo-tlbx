#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o Zdeci.%N.%a.%j.out # STDOUT
#SBATCH -e Zdeci.%N.%a.%j.err # STDERR
#SBATCH --job-name="Zdeci"

####Set permissions of output files:
umask 002
####################################

###SHPS=( "/home/sandbox-cel/capeTown/vector/deci16_mask1420_32734.shp" "/home/sandbox-cel/capeTown/vector/deci16_mask1418_32734.shp" "/home/sandbox-cel/capeTown/vector/deci16_mask18_32734.shp")

SHPS=("/home/l_sharwood/code/deci16_mask2_18_32734.shp")
VIs=("evi" "ndvi" "ndmi1" "ndmi2" "savi") ## ("ndmi2") ###  "evi" "ndvi" "ndmi1" "ndmi2" "savi"

startYr=2016 ## 2014 for landsat. ## 2016 for sentinel
##month_dir="/home/sandbox-cel/capeTown/monthly/landsat/vrts" 
month_dir="/home/sandbox-cel/capeTown/monthly/sentinel2_brdf/vrts" 
## month_dir="/home/sandbox-cel/capeTown/monthly/sentinel2_brdf/vrts" 
endYr=2021

####################################

source ~/.bashrc
cd ~/
conda activate .helpers38

for inShp in "${SHPS[@]}"
do
    for VI in "${VIs[@]}"
    do
    python code/bash/ct_decile_monthly_zonal.py $inShp $month_dir $VI $startYr $endYr
    done
done
conda deactivate



