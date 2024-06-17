#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-12:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o S1resiz.%N.%a.%j.out # STDOUT
#SBATCH -e S1resiz.%N.%a.%j.err # STDERR
#SBATCH --job-name="S1resiz"
#SBATCH --array=[2]%2



####Set permissions of output files:
umask 002
##############################################

GRID_ID="$(($SLURM_ARRAY_TASK_ID))"

INPUT_DIR="/home/downspout-cel/DL/raster/S1/" 

##############################################
cd ~/
source .bashrc
conda activate .helpers38
cd /home/lsharwood/code/DL/bash_scripts/

python gw_resize.py "${GRID_ID}" "${INPUT_DIR}"
