#!/bin/bash

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import geowombat as gw
from datetime import date 
import json
import time
from shapely.geometry import Point
import rasterio as rio
import xarray as xr
import rioxarray
import tqdm
from itertools import chain
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sys import argv

start = time.time()

###########################

folder_num = argv[1]
print(folder_num)

base_image = '/home/downspout-cel/DL/raster/grids2/'+ str(folder_num).zfill(6) + '/brdf_ts/ms/ndmi/2017020.tif'
out_folder = "UNQ_" + str(folder_num).zfill(2)

    
input_dir = argv[2]

top_S1 = ['S1_2017_MeanLateDryVV_foc3.tif', 'S1_2017_StdevEarlyWetVH_foc3.tif', 'S1_2017_MinEarlyDryVV_foc3.tif', 'S1_2017_StdevEarlyDryVV_foc3.tif', 'S1_2017_StdevEarlyWetVV_foc3.tif', 'S1_2017_MeanEarlyDryVV_foc3.tif', 'S1_2017_MaxBonusDryVH_foc3.tif', 'S1_2017_MeanBonusDryVH_foc3.tif']

raster_list =  [r for r in os.listdir(input_dir) if r.endswith(".tif") and (r in top_S1 or "NASA" in r)] 

out_dir = os.path.join(input_dir, out_folder)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

###########################
with gw.open("/home/sandbox-cel/DL/PlanetScope/2017_harmonized_aoi2/20171009_091419_0f52_3B_AnalyticMS_SR_harmonized.tif") as tar_res:
    new_res = tar_res.res
    
with gw.open(base_image) as target:
    new_crs = target.crs
    new_extent = target.gw.bounds
    for rast in raster_list:
        if int(folder_num) < 10:
            out_rast = rast.split(".")[0] + "_UNQ0" + str(folder_num) + "_UTM32.tif" 
        if int(folder_num) >= 10:
            out_rast = rast.split(".")[0] + "_UNQ" + str(folder_num) + "_UTM32.tif"     
        with gw.config.update(nodata=0, ref_crs=new_crs,  ref_res=new_res, ref_bounds=new_extent):
            with gw.open(os.path.join(input_dir, rast)) as ref:
                attrs = ref.attrs.copy()
                ref = (ref).assign_attrs(**attrs, count=1, nodata=0, verbose=1)
                ref = (ref).assign_attrs(res=new_res)
                ref.gw.to_raster(os.path.join(out_dir, out_rast))
                print(os.path.join(out_dir, out_rast))   

###########################
end = time.time()
minutes = (end - start)/60
print('time this took ' , str(minutes), ' minutes to complete')
sys.stdout.flush()
