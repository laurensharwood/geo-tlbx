#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RF22.%N.%a.%j.out # STDOUT
#SBATCH -e RF22.%N.%a.%j.err # STDERR
#SBATCH --job-name="RF22"

######################

## STEPS=steps to create glacis prediction raster
## all: ['splitTrainHoldout','stack','extract','append','train','tune','saveAccuracy','savePredRasts','saveFeatImp']  
##STEPS=['stack','extract','append','train','saveAccuracy','savePredRasts']
STEPS=['savePredRasts']


#### TRAIN/VAL VECTOR PARAMS #### 
## TV_SHAPE=first col name, class_code or code_2, will be used as the label ## 2019_GlacisNot ## 2019_8cl
## TV_FIELD=Training / VALIDATION partition (75/25) field to use 
## Overwrites if 'splitTrainHoldout' is in STEPS and the field exists 
TV_SHAPE="/home/downspout-cel/DL/glacisMod/models/8cl_stac_mean/GlacisNOT.shp"
TV_FIELD="TV_01" #"TV_01" or "TV06"

#### REMOTE SENSING FEATURES PARAMS ####
## GRID_DIR=folder to look for input raster features to stack /grids/###/Feat_UNQ###.tif
## STACK_DIR=folder to save raster stacks ## make sure "Class_Lookup.csv" exists in this folder
## YR_PREFIX=prefix to add to raster stack-- must be 'Yr##' 
### model results saved into a subdir as TV_SHAPE
YR_PREFIX="Yr22"
GRID_DIR="/home/downspout-cel/DL/raster/stac_grids/"
STACK_DIR="/home/downspout-cel/DL/glacisMod/stac_mean_stacks/"

#### MODEL PARAMS #### 
## SEED=random seed for RF model training 
## PRED_THRESH=glacis probability threshold for prediction rasters
## PRED_GRIDS=UNQ grid cells to predict, saved in TV_SHAPE subfolder 

### "discrete" "continuous" 0.7
PRED_THRESH=0.7

PRED_GRIDS=[45,46,58,59,60,61,73,74,75,87,88,89]
##[21,22,23,24,35,36,45,46,58,59,60,61,64,65,66,72,73,74,75,76,77,78,79,86,87,88,89,91,92,93,100,101,105,106,107,108,119,120,121,122,123,124,133,134,135,136,137,138,160,174,175,188,189,201]

SEED=615

#####################

cd ~/
source .bashrc
conda activate .helpers38
python ~/code/bash/DL_RandomForest.py $STEPS $YR_PREFIX $TV_SHAPE $TV_FIELD $SEED $PRED_THRESH $PRED_GRIDS $GRID_DIR $STACK_DIR
conda deactivate

