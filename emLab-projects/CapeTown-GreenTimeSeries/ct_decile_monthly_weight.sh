#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o Wdeci.%N.%a.%j.out # STDOUT
#SBATCH -e Wdeci.%N.%a.%j.err # STDERR
#SBATCH --job-name="Wdeci"

####Set permissions of output files:
umask 002
####################################

SHPS=("/home/sandbox-cel/capeTown/vector/deci16_mask1418_32734.shp" "/home/sandbox-cel/capeTown/vector/deci16_mask1820_32734.shp" "/home/sandbox-cel/capeTown/vector/deci16_mask1420_32734.shp")

startYr=2014 ## 2014 for landsat. ## 2017 for sentinel
month_dir="/home/sandbox-cel/capeTown/monthly/landsat/vrts" 
## month_dir="/home/sandbox-cel/capeTown/monthly/sentinel2_refl/vrts" 
## month_dir="/home/sandbox-cel/capeTown/monthly/sentinel2_brdf/vrts" 
endYr=2021
VIs=("evi" "ndvi" "ndmi1" "ndmi2") ## ("ndmi2") ### 

source ~/.bashrc
cd ~/
conda activate .helpers38

for VI in "${VIs[@]}"
do
    for inShp in "${SHPS[@]}"
    do
    python code/bash/ct_decile_monthly_weight.py $inShp $month_dir $VI $startYr $endYr
    done
done
conda deactivate

