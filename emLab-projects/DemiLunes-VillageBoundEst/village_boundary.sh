#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o vilBnd.%N.%a.%j.out # STDOUT
#SBATCH -e vilBnd.%N.%a.%j.err # STDERR
#SBATCH --job-name="vilBnd"

####################################
####Set permissions of output files:
umask 002
####################################

ogPts="/home/l_sharwood/code/demilunes/vilBoundEst/DL_field_pts.gpkg"
orig_overlap_pctile_4KDE=0.75 ## 0.25 0.50 0.75
orig_size_pctile_4KDE=0.75 ## 0.25 0.50 0.75
out_dir="/home/l_sharwood/code/demilunes/vilBoundEst/"
cd ~/
conda activate .helpers38
python code/bash/village_boundary_funcs.py $ogPts $orig_overlap_pctile_4KDE $orig_size_pctile_4KDE $out_dir

### 
## Download "NewBounds-version-wOlap_UTM32.shp" files in outDir locally
## Remove Overlap (batch) in Arc
## Renamer: replace 'wOlap_UTM32' -> 'noOlap'

## remove multiparts (keep largest) *** 
## max(multipolygon, key=lambda a: a.area)

## Accuracy assessment (param_assess)
