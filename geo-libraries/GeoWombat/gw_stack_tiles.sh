#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-12:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stack.%N.%a.%j.out # STDOUT
#SBATCH -e stack.%N.%a.%j.err # STDERR
#SBATCH --job-name="stackVars"
#SBATCH --array=[2]%2


####Set permissions of output files:
umask 002
##############################################

GRID_ID="$(($SLURM_ARRAY_TASK_ID))"

INPUT_DIR="/home/downspout-cel/DL/raster/S1/02_diff_vars/refined_lee_filter/"

OUTPUT_DIR="/home/downspout-cel/DL/raster/S1/04_stacks/top36_full/"

FEATURE_LIST=['S2_max_NDMI_082017', 'S2_avg_SWIR2_052017', 'S1A_VV_2017_Mar_avg_foc5', 'S2_avg_SWIR2_062017', 'S2_avg_SAVI_052017', 'S2_max_SWIR2_062017', 'S2_avg_SWIR1_052017', 'S2_avg_NDWI_042017', 'S2_max_SWIR2_052017', 'S2_max_NDMI_062017', 'NASA_dem_foc3', 'S2_avg_NDWI_052017', 'S1_VV_2017_EarlyWet_StDev_foc5', 'S2_avg_SWIR1_102017', 'S2_max_NDWI_032017', 'S2_avg_SWIR1_072017', 'S2_max_NDWI_062017', 'S1A_VHVV_20170224_divide_foc5', 'NASA_dem', 'S2_max_NDMI_072017', 'S1A_VV_2017_May_avg', 'S2_avg_NDWI_032017', 'S1_VH_2017_BonusDry_Min_foc3', 'S2_max_NDMI_112017', 'S1A_VHVV_20170531_divide_foc5', 'S1_VV_2017_EarlyDry_Min_foc5', 'Palsar_17_sl_HV_F02DAR_UTM32', 'Palsar_17_sl_HH_F02DAR_UTM32', 'S1_VV_2017_EarlyDry_StDev_foc5', 'S1A_VHVV_20170401_divide_foc3', 'S1_VV_2017_EarlyDry_Min_foc3', 'S2_avg_NDMI_092017', 'S2_avg_SWIR2_072017', 'NASA_dem_foc3_slope', 'S2_max_NDWI_042017', 'NASA_dem_slope']


##############################################
cd ~/
source .bashrc
conda activate .helpers38
cd /home/lsharwood/code/DL/bash_scripts/

python gw_stack_tiles.py "${GRID_ID}" "${INPUT_DIR}" "${OUTPUT_DIR}" "${FEATURE_LIST}"
