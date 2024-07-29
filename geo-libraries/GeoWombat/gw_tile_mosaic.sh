#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-12:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o S1tile.%N.%a.%j.out # STDOUT
#SBATCH -e S1tile.%N.%a.%j.err # STDERR
#SBATCH --job-name="tilMos"
#SBATCH --array=[2,3,4,5,6,7,8,10,11]%2



####Set permissions of output files:
umask 002
##############################################

###### grid number ... array=[2,3,4,5,7,8,10,11]
GRID_ID="$(($SLURM_ARRAY_TASK_ID))"

###### 01_mosaics/temporal_filter/, 01_mosaics/refined_lee_filter/, 01_mosaics/lee_filter/, or just 01_mosaics/
INPUT_DIR="/home/downspout-cel/DL/raster/S1/01_mosaics/TOPO"

###### "NASA_dem" for elevation/topo,"Palsar",  "mosaic.tif" for S1 
KEYWORD="NASA_dem" 

OUTPUT_DIR="/home/downspout-cel/DL/raster/S1/02_diff_vars/refined_lee_filter/"

##############################################
cd ~/
source .bashrc
conda activate .helpers38
cd /home/lsharwood/code/DL/bash_scripts/

python gw_tile_mosaic.py "${GRID_ID}" "${INPUT_DIR}" "${KEYWORD}" "${OUTPUT_DIR}"
